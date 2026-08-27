"""Treino das arvores de decisao (XGBoost) para os dois modelos.

METRICA QUE INTERESSA -- e nao e' a accuracy global.
Com ESPERAR a ser ~42% das linhas, um modelo que responda sempre ESPERAR
acerta 42% e e' completamente inutil: nunca actua. O numero que decide e' a
PRECISAO NAS DECISOES ACCIONAVEIS -- quando o modelo diz SOBE ou DESCE, com
que frequencia acerta -- junto com a fatia de vezes em que se atreve a decidir.
Um modelo que decide em 5% dos casos com 60% de precisao pode valer mais do
que um que decide sempre com 45%.

VARRIMENTO DE PROFUNDIDADE: treina-se a varias profundidades para testar com
numeros a intuicao "mais detalhado = mais lento mas mais preciso". Mede-se
accuracy, precisao accionavel, tempo de treino e latencia de inferencia.

Split cronologico com embargo, e os erros-padrao sobre DIAS distintos: as 43
accoes movem-se juntas, portanto linhas do mesmo minuto nao sao observacoes
independentes.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.patterns.ingest_intraday import load_bars
from src.trees import features as F

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

T = config.TREES
MODELS = ("indicators", "microstructure")


def build_dataset(conn, timeframe: str, model: str, tickers=None) -> pd.DataFrame:
    """Junta features + rotulo para todos os tickers de um timeframe."""
    tickers = tickers or config.PATTERN_TICKERS
    builder = F.indicator_features if model == "indicators" else F.microstructure_features

    frames = []
    for ticker in tickers:
        bars = load_bars(conn, ticker, timeframe)
        if len(bars) < 200:
            continue
        matrix = builder(bars)
        labels = F.label_three_class(bars)
        frame = matrix.join(labels)
        frame["ticker"] = ticker
        frame["ts"] = frame.index
        # O indice vem de load_bars ja' chamado "ts"; sem isto ficam um nivel
        # de indice e uma coluna com o mesmo nome e o sort_values e' ambiguo.
        frame.index.name = None
        frames.append(frame.dropna())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values(["ts", "ticker"]).reset_index(drop=True)


def split_by_time(frame: pd.DataFrame, horizon: int):
    """Split cronologico GLOBAL (por timestamp, nao por ticker).

    A 1m todos os tickers partilham os mesmos minutos, portanto cortar por
    indice dentro de cada ticker deixaria o teste de um ticker a sobrepor-se
    ao treino de outro no mesmo instante -- fuga de informacao entre accoes
    correlacionadas. Cortar pelo tempo global evita isso.

    O embargo remove as linhas cuja janela futura atravessa a fronteira."""
    timestamps = np.sort(frame["ts"].unique())
    first = timestamps[int(len(timestamps) * T["train_fraction"])]
    second = timestamps[int(len(timestamps) * (T["train_fraction"] + T["validation_fraction"]))]

    step = pd.Series(timestamps).diff().median()
    embargo = step * horizon if pd.notna(step) else pd.Timedelta(0)

    train = frame[frame["ts"] < first - embargo]
    validation = frame[(frame["ts"] >= first) & (frame["ts"] < second - embargo)]
    test = frame[frame["ts"] >= second]
    return train, validation, test


def actionable_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precisao quando o modelo se atreve a decidir (SOBE ou DESCE)."""
    acted = y_pred != F.WAIT
    n_acted = int(acted.sum())
    if n_acted == 0:
        return {"action_rate": 0.0, "action_precision": None,
                "action_precision_baseline": None, "action_precision_lift": None,
                "up_precision": None, "down_precision": None,
                "up_base_rate": None, "down_base_rate": None, "reversal_rate": None}

    correct = (y_pred[acted] == y_true[acted]).sum()
    up = y_pred == F.UP
    down = y_pred == F.DOWN
    precision = float(correct / n_acted)

    # Baseline justo para a precisao: se o modelo dissesse SOBE ao calhas,
    # acertaria a taxa-base de SOBE. Sem esta comparacao, "acerta 35%" nao se
    # sabe se e' bom ou mau -- com tres classes desequilibradas, 35% pode ser
    # otimo (se SOBE for 28% das barras) ou pessimo.
    base_up = float((y_true == F.UP).mean())
    base_down = float((y_true == F.DOWN).mean())
    base_acted = ((base_up * up.sum() + base_down * down.sum()) / n_acted) if n_acted else 0.0

    return {
        "action_rate": n_acted / len(y_pred),
        "action_precision": precision,
        "action_precision_baseline": base_acted,
        "action_precision_lift": precision - base_acted,
        "up_precision": float((y_true[up] == F.UP).mean()) if up.any() else None,
        "down_precision": float((y_true[down] == F.DOWN).mean()) if down.any() else None,
        "up_base_rate": base_up,
        "down_base_rate": base_down,
        # Erro grave: dizer SOBE quando desceu (ou o contrario). Confundir com
        # ESPERAR custa uma oportunidade; isto custa dinheiro.
        "reversal_rate": float(
            ((y_pred[acted] != F.WAIT) & (y_true[acted] != F.WAIT)
             & (y_pred[acted] != y_true[acted])).mean()
        ),
    }


def train_one(conn, timeframe: str, model: str) -> dict | None:
    frame = build_dataset(conn, timeframe, model)
    if frame.empty:
        return None

    feature_names = (F.INDICATOR_FEATURES if model == "indicators"
                     else F.microstructure_feature_names())
    train_df, validation_df, test_df = split_by_time(frame, T["horizon_bars"])
    if len(train_df) < T["min_train_rows"] or validation_df.empty or test_df.empty:
        return None

    X_train, y_train = train_df[feature_names], train_df["label"].astype(int)
    X_validation, y_validation = validation_df[feature_names], validation_df["label"].astype(int)
    X_test, y_test = test_df[feature_names], test_df["label"].astype(int)

    # --- varrimento de profundidade, escolhido na VALIDACAO ---
    sweep = []
    for depth in T["depth_sweep"]:
        params = dict(T["xgb_params"], max_depth=depth)
        estimator = XGBClassifier(**params)

        started = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predictions = estimator.predict(X_validation)
        predict_seconds = time.perf_counter() - started

        metrics = actionable_metrics(y_validation.values, predictions)
        sweep.append({
            "max_depth": depth,
            "accuracy": float((predictions == y_validation.values).mean()),
            "action_precision": metrics["action_precision"],
            "action_rate": metrics["action_rate"],
            "train_seconds": train_seconds,
            "predict_ms_per_row": 1000 * predict_seconds / max(len(X_validation), 1),
            "n_nodes": int(sum(t.count("\n") for t in estimator.get_booster().get_dump())),
        })

    # Escolhe pela precisao accionavel, nao pela accuracy: a accuracy premeia
    # responder sempre ESPERAR, que e' precisamente o modelo inutil.
    usable = [s for s in sweep if s["action_precision"] is not None and s["action_rate"] > 0.02]
    best = max(usable or sweep, key=lambda s: s["action_precision"] or 0)

    final = XGBClassifier(**dict(T["xgb_params"], max_depth=best["max_depth"]))
    final.fit(pd.concat([X_train, X_validation]), pd.concat([y_train, y_validation]))

    started = time.perf_counter()
    test_predictions = final.predict(X_test)
    latency_ms = 1000 * (time.perf_counter() - started) / max(len(X_test), 1)

    accuracy = float((test_predictions == y_test.values).mean())
    metrics = actionable_metrics(y_test.values, test_predictions)

    majority = int(y_train.mode().iloc[0])
    baseline = float((y_test.values == majority).mean())
    effective_days = int(pd.Series(test_df["ts"]).dt.normalize().nunique())
    standard_error = float(np.sqrt(0.25 / max(effective_days, 1)))

    importances = sorted(
        zip(feature_names, final.feature_importances_), key=lambda item: -item[1]
    )[:10]

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final, "feature_names": feature_names, "kind": model,
         "timeframe": timeframe, "max_depth": best["max_depth"]},
        config.MODELS_DIR / f"tree_{model}_{timeframe}.joblib",
    )

    return {
        "model": model, "timeframe": timeframe, "max_depth": best["max_depth"],
        "n_train": len(train_df), "n_validation": len(validation_df), "n_test": len(test_df),
        "n_features": len(feature_names), "effective_days": effective_days,
        "accuracy": accuracy, "baseline_majority": baseline,
        "baseline_class": F.CLASS_NAMES[majority],
        "standard_error": standard_error,
        "significant": abs(accuracy - baseline) > 2 * standard_error,
        "latency_ms": latency_ms,
        "class_distribution": {F.CLASS_NAMES[k]: float(v) for k, v in
                               y_test.value_counts(normalize=True).sort_index().items()},
        "top_features": [(name, float(value)) for name, value in importances],
        "sweep": sweep,
        **metrics,
    }


def store(conn, result: dict) -> str:
    version = (f"tree_{result['model']}_{result['timeframe']}_"
               f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}")
    conn.execute(
        """INSERT INTO tree_models
             (version, model_kind, timeframe, max_depth, trained_at, n_train, n_validation,
              n_test, n_features, effective_days, accuracy, baseline_majority, baseline_class,
              action_rate, action_precision, action_precision_baseline, action_precision_lift,
              up_precision, down_precision, up_base_rate, down_base_rate, reversal_rate,
              standard_error, significant, latency_ms, class_distribution_json,
              top_features_json, sweep_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (version, result["model"], result["timeframe"], result["max_depth"],
         datetime.now(timezone.utc).isoformat(), result["n_train"], result["n_validation"],
         result["n_test"], result["n_features"], result["effective_days"], result["accuracy"],
         result["baseline_majority"], result["baseline_class"], result["action_rate"],
         result["action_precision"], result["action_precision_baseline"],
         result["action_precision_lift"], result["up_precision"], result["down_precision"],
         result["up_base_rate"], result["down_base_rate"],
         result["reversal_rate"], result["standard_error"], int(result["significant"]),
         result["latency_ms"], json.dumps(result["class_distribution"]),
         json.dumps(result["top_features"]), json.dumps(result["sweep"])),
    )
    conn.commit()
    return version


def _report(result: dict) -> None:
    print(f"\n=== {result['model'].upper()} @ {result['timeframe']} ===")
    print(f"  treino={result['n_train']:,} validacao={result['n_validation']:,} "
          f"teste={result['n_test']:,} | {result['n_features']} features | "
          f"{result['effective_days']} dias efectivos no teste")
    print(f"  profundidade escolhida: {result['max_depth']}   latencia: "
          f"{result['latency_ms']:.4f} ms/linha")
    print(f"  distribuicao no teste: {result['class_distribution']}")
    print(f"  accuracy 3 classes: {result['accuracy']:.1%}  "
          f"(baseline responder sempre {result['baseline_class']}: {result['baseline_majority']:.1%}, "
          f"+-{result['standard_error']*100:.1f}pp)")
    precision = result["action_precision"]
    if precision is None:
        print("  nunca decide")
    else:
        lift = result["action_precision_lift"] * 100
        print(f"  DECIDE em {result['action_rate']:.1%} das barras; quando decide acerta "
              f"{precision:.1%}  (ao calhas seria {result['action_precision_baseline']:.1%} "
              f"-> lift {lift:+.1f}pp)")
        print(f"     SOBE: {(result['up_precision'] or 0):.1%} vs base {result['up_base_rate']:.1%}   "
              f"DESCE: {(result['down_precision'] or 0):.1%} vs base {result['down_base_rate']:.1%}   "
              f"inversoes graves: {result['reversal_rate']:.1%}")
    print(f"  top features: {', '.join(n for n, _ in result['top_features'][:5])}")
    print(f"  {'profundidade':>12} {'accuracy':>9} {'precisao':>9} {'decide':>8} "
          f"{'treino(s)':>10} {'ms/linha':>9} {'nos':>8}")
    for step in result["sweep"]:
        precision_text = f"{step['action_precision']:.1%}" if step["action_precision"] else "--"
        print(f"  {step['max_depth']:>12} {step['accuracy']:>8.1%} {precision_text:>9} "
              f"{step['action_rate']:>7.1%} {step['train_seconds']:>10.1f} "
              f"{step['predict_ms_per_row']:>9.4f} {step['n_nodes']:>8,}")


if __name__ == "__main__":
    connection = database.get_connection()
    for timeframe in T["timeframes"]:
        for model in MODELS:
            outcome = train_one(connection, timeframe, model)
            if outcome is None:
                print(f"{model} @ {timeframe}: dados insuficientes")
                continue
            store(connection, outcome)
            _report(outcome)
    connection.close()
