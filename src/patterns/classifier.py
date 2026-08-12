"""Classificador multiclasse (16 classes) do proximo padrao, condicionado no
padrao actual E no contexto medido em que ele se formou.

Onde entraria na cadeia de 4 passos -- e porque nao entra em todos:

  passo 1     -> este classificador. O padrao actual ja' se formou, portanto a
                 sua qualidade, duracao, volume e regime de mercado sao factos
                 medidos.
  passos 2-4  -> cadeia de Markov (src/patterns/sequence.py). O padrao de
                 partida desses passos e' um padrao PREVISTO que ainda nao
                 existe: nao tem qualidade nem volume medidos. Atribuir-lhe
                 valores tipicos daria numeros com ar mais informado sem
                 informacao nova nenhuma por tras. A cadeia de Markov e'
                 precisamente a versao marginalizada sobre esse contexto, que
                 e' a coisa honesta a usar quando o contexto e' desconhecido.

PROTOCOLO DE AVALIACAO -- tres particoes, nao duas.
Cronologicas por ticker: 60% treino, 20% validacao, 20% teste. O algoritmo e o
peso do ensemble sao escolhidos na VALIDACAO; o teste so' e' tocado para
reportar. Com split a dois (escolhendo no proprio teste) o ensemble a 5m
aparentava +2.9pp de ganho; com o protocolo correcto sobraram +0.3pp, dentro
de um erro-padrao. E' facil enganarmo-nos aqui, dai a separacao ser explicita.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.patterns import context, sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_FRACTION, VALIDATION_FRACTION = 0.6, 0.2
MIN_TRAIN_ROWS = 200
ENSEMBLE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _make_logistic() -> Pipeline:
    # SEM class_weight="balanced" de proposito. Com 16 classes muito
    # desequilibradas (BEAR_FLAG ~29%, DIAMOND_BOTTOM ~0.3%), balancear
    # amplifica as raras ~100x e o modelo passa a prever classes raras quase
    # sempre: medido, deu 4.9% de accuracy, abaixo do acaso (1/16 = 6.25%).
    # Balancear serve para recall em minorias; aqui a metrica e' accuracy.
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])


def _make_boosting() -> HistGradientBoostingClassifier:
    # Deliberadamente pequeno: 16 classes com ~1000-2000 exemplos overfitam
    # com muita facilidade se se deixar a arvore crescer.
    return HistGradientBoostingClassifier(
        max_depth=4, max_iter=150, learning_rate=0.06,
        min_samples_leaf=25, l2_regularization=1.0, random_state=42,
    )


def _split_three_ways(frame: pd.DataFrame):
    """Cronologico por ticker. Nunca se mistura o futuro de um ticker com o
    passado de outro, e dentro de cada ticker o teste e' sempre o periodo mais
    recente."""
    parts = ([], [], [])
    for _, group in frame.groupby("ticker", sort=False):
        group = group.sort_values("confirmed_ts")
        n = len(group)
        first, second = int(n * TRAIN_FRACTION), int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))
        parts[0].append(group.iloc[:first])
        parts[1].append(group.iloc[first:second])
        parts[2].append(group.iloc[second:])
    return tuple(
        pd.concat(p).sort_values("confirmed_ts").reset_index(drop=True) if p else pd.DataFrame()
        for p in parts
    )


def _markov_from(frame: pd.DataFrame) -> dict:
    counts = {}
    for from_pattern, to_pattern in zip(frame["from_pattern"], frame["next_pattern"]):
        counts[(from_pattern, to_pattern)] = counts.get((from_pattern, to_pattern), 0) + 1
    return {key: {"count": value, "median_bars": None} for key, value in counts.items()}


def _blended_probabilities(model, transitions, frame, weight: float) -> np.ndarray:
    """Matriz (n_linhas x 16) com w*classificador + (1-w)*Markov, na ordem
    canonica de config.PATTERN_TYPES."""
    order = config.PATTERN_TYPES
    predicted = model.predict_proba(frame[context.FEATURE_NAMES])
    classes = list(model.classes_)

    out = np.zeros((len(frame), len(order)))
    for row, (index, from_pattern) in enumerate(enumerate(frame["from_pattern"])):
        by_class = dict(zip(classes, predicted[index]))
        classifier_vector = np.array([by_class.get(p, 1e-4) for p in order])
        markov = {p: prob for p, prob, _, _ in sequence.transition_distribution(transitions, from_pattern)}
        markov_vector = np.array([markov[p] for p in order])
        out[row] = weight * classifier_vector + (1 - weight) * markov_vector
    return out


def _accuracy_at_k(probabilities: np.ndarray, actual, k: int = 1) -> float:
    order = np.array(config.PATTERN_TYPES)
    top_k = np.argsort(probabilities, axis=1)[:, -k:]
    hits = sum(1 for row, truth in zip(top_k, actual) if truth in order[row])
    return hits / len(actual) if len(actual) else 0.0


def train(conn, timeframe: str) -> dict | None:
    frame = context.build_dataset(conn, timeframe)
    if frame.empty:
        return None

    train_df, validation_df, test_df = _split_three_ways(frame)
    if len(train_df) < MIN_TRAIN_ROWS or validation_df.empty or test_df.empty:
        database.log_run(
            conn, datetime.now(timezone.utc).strftime("%Y-%m-%d"), "pattern_classifier", "WARN",
            f"{timeframe}: apenas {len(train_df)} linhas de treino (minimo {MIN_TRAIN_ROWS}) -- adiado",
        )
        return None

    features = context.FEATURE_NAMES
    X_train, y_train = train_df[features], train_df["next_pattern"]
    transitions = _markov_from(train_df)

    # --- escolha do algoritmo: na VALIDACAO ---
    candidates = {"logistic_regression": _make_logistic(), "hist_gradient_boosting": _make_boosting()}
    validation_scores = {}
    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        probabilities = _blended_probabilities(estimator, transitions, validation_df, weight=1.0)
        validation_scores[name] = _accuracy_at_k(probabilities, list(validation_df["next_pattern"]))

    best_name = max(validation_scores, key=validation_scores.get)
    model = candidates[best_name]

    # --- escolha do peso do ensemble: tambem na VALIDACAO ---
    weight_scores = {
        w: _accuracy_at_k(
            _blended_probabilities(model, transitions, validation_df, w),
            list(validation_df["next_pattern"]),
        )
        for w in ENSEMBLE_GRID
    }
    best_weight = max(weight_scores, key=weight_scores.get)

    # --- reportar: so' agora se toca no TESTE ---
    y_test = list(test_df["next_pattern"])
    probabilities_classifier = _blended_probabilities(model, transitions, test_df, 1.0)
    probabilities_markov = _blended_probabilities(model, transitions, test_df, 0.0)
    probabilities_ensemble = _blended_probabilities(model, transitions, test_df, best_weight)

    accuracy = _accuracy_at_k(probabilities_classifier, y_test)
    top3_accuracy = _accuracy_at_k(probabilities_classifier, y_test, k=3)
    markov_accuracy = _accuracy_at_k(probabilities_markov, y_test)
    markov_top3_accuracy = _accuracy_at_k(probabilities_markov, y_test, k=3)
    ensemble_accuracy = _accuracy_at_k(probabilities_ensemble, y_test)

    most_common = y_train.value_counts().index[0]
    baseline_accuracy = float((pd.Series(y_test) == most_common).mean())
    standard_error = float(np.sqrt(max(accuracy, 1e-9) * (1 - accuracy) / len(y_test)))

    version = f"clf_{timeframe}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": features, "version": version,
         "algorithm": best_name, "timeframe": timeframe, "ensemble_weight": best_weight},
        config.MODELS_DIR / f"pattern_clf_{timeframe}.joblib",
    )

    conn.execute(
        """INSERT INTO pattern_model_versions
             (version, timeframe, algorithm, trained_at, n_train, n_validation, n_test, n_features,
              accuracy, top3_accuracy, markov_accuracy, markov_top3_accuracy,
              baseline_frequency_accuracy, ensemble_weight, ensemble_accuracy, standard_error,
              hyperparams_json, feature_names_json, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (version, timeframe, best_name, datetime.now(timezone.utc).isoformat(),
         len(train_df), len(validation_df), len(test_df), len(features),
         accuracy, top3_accuracy, markov_accuracy, markov_top3_accuracy, baseline_accuracy,
         best_weight, ensemble_accuracy, standard_error,
         json.dumps({}), json.dumps(features),
         f"validacao algoritmos={ {k: round(v, 4) for k, v in validation_scores.items()} }; "
         f"validacao pesos={ {k: round(v, 4) for k, v in weight_scores.items()} }; "
         f"baseline preve sempre {most_common}"),
    )
    conn.commit()

    return {
        "version": version, "timeframe": timeframe, "algorithm": best_name,
        "n_train": len(train_df), "n_validation": len(validation_df), "n_test": len(test_df),
        "n_features": len(features), "accuracy": accuracy, "top3_accuracy": top3_accuracy,
        "markov_accuracy": markov_accuracy, "markov_top3_accuracy": markov_top3_accuracy,
        "baseline_accuracy": baseline_accuracy, "baseline_pattern": most_common,
        "ensemble_weight": best_weight, "ensemble_accuracy": ensemble_accuracy,
        "standard_error": standard_error,
        "validation_algorithms": validation_scores, "validation_weights": weight_scores,
    }


_MODEL_CACHE = {}


def load_model(timeframe: str, refresh: bool = False):
    if refresh or timeframe not in _MODEL_CACHE:
        path = config.MODELS_DIR / f"pattern_clf_{timeframe}.joblib"
        _MODEL_CACHE[timeframe] = joblib.load(path) if path.exists() else None
    return _MODEL_CACHE[timeframe]


def predict_distribution(timeframe: str, features: dict) -> list | None:
    """[(pattern_type, probabilidade)] ordenado, ou None se nao houver
    classificador treinado. Classes nunca vistas no treino recebem uma
    probabilidade residual em vez de zero -- nunca vista nao e' impossivel."""
    bundle = load_model(timeframe)
    if bundle is None:
        return None

    X = pd.DataFrame([features])[bundle["feature_names"]]
    probabilities = bundle["model"].predict_proba(X)[0]
    known = dict(zip(bundle["model"].classes_, probabilities))
    full = {p: known.get(p, 1e-4) for p in config.PATTERN_TYPES}
    total = sum(full.values())
    return sorted(((p, v / total) for p, v in full.items()), key=lambda item: -item[1])


if __name__ == "__main__":
    connection = database.get_connection()
    for tf in config.PATTERN_TIMEFRAMES:
        result = train(connection, tf)
        if result is None:
            print(f"{tf}: treino adiado (dados insuficientes)")
            continue
        print(f"=== {tf} ===")
        print(f"  treino={result['n_train']} validacao={result['n_validation']} teste={result['n_test']} "
              f"features={result['n_features']}")
        print(f"  VALIDACAO -> algoritmos { {k: f'{v:.1%}' for k, v in result['validation_algorithms'].items()} }")
        print(f"               pesos      { {k: f'{v:.1%}' for k, v in result['validation_weights'].items()} }")
        print(f"               escolhidos: {result['algorithm']}, peso do ensemble w={result['ensemble_weight']}")
        print(f"  TESTE (nunca usado para escolher nada, erro-padrao ~{result['standard_error']*100:.2f}pp)")
        print(f"    {'':22s} {'top-1':>8s} {'top-3':>8s}")
        print(f"    {'classificador':22s} {result['accuracy']:>7.1%} {result['top3_accuracy']:>8.1%}")
        print(f"    {'cadeia de Markov':22s} {result['markov_accuracy']:>7.1%} {result['markov_top3_accuracy']:>8.1%}")
        print(f"    {'ensemble':22s} {result['ensemble_accuracy']:>7.1%}        -")
        print(f"    {'baseline frequencia':22s} {result['baseline_accuracy']:>7.1%}        -  "
              f"(prever sempre {result['baseline_pattern']})")
        delta = (result["accuracy"] - result["markov_accuracy"]) * 100
        significant = abs(delta) > 2 * result["standard_error"] * 100
        print(f"    contexto vs Markov: {delta:+.1f} pp -> "
              f"{'significativo' if significant else 'DENTRO DO RUIDO (< 2 erros-padrao)'}")
    connection.close()
