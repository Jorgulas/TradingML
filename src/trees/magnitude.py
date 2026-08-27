"""Previsao de MAGNITUDE do proximo movimento (nao da direccao).

Porque este alvo. A decomposicao da precisao do modelo de direccao deu:
  P(houve movimento | modelo decidiu) = 82.0%  contra 46.0% de taxa-base
  P(direccao certa | houve movimento) = 53.5%  contra 50% de acaso
O sinal que o modelo tem e' de magnitude. Medir direccao era atribuir-lhe o
merito de uma coisa que ele nao faz.

Dois alvos sobre exactamente as mesmas features:
  CLASSIFICACAO  big_move -- o movimento sai da banda de indiferenca?
  REGRESSAO      magnitude_ratio -- vale quantos desvios-padrao recentes?

O de regressao e' o mais util: diz QUANTO, nao apenas SE.

DUAS BASELINES OBRIGATORIAS, senao o resultado nao se sabe ler:
  classe maioritaria   -- responder sempre o mesmo
  persistencia de vol  -- um unico numero, vol_10/vol_20 (a volatilidade curta
                          esta' a subir?). A volatilidade e' fortemente
                          autocorrelacionada, portanto qualquer modelo tem
                          obrigacao de bater isto; se nao bater, as 40 features
                          nao estao a acrescentar nada a uma regra de uma linha.
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
from src.trees import train as tree_train

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

T = config.TREES


def build_dataset(conn, timeframe: str, model: str, tickers=None) -> pd.DataFrame:
    """Mesmas features do modelo de direccao, alvo diferente."""
    tickers = tickers or config.TREE_TICKERS
    builder = F.indicator_features if model == "indicators" else F.microstructure_features
    benchmark = load_bars(conn, config.BENCHMARK["ticker"], timeframe)

    frames = []
    for ticker in tickers:
        bars = load_bars(conn, ticker, timeframe)
        if len(bars) < 200:
            continue
        matrix = builder(bars).join(F.market_context_features(bars, benchmark))
        frame = matrix.join(F.label_magnitude(bars))
        frame["ticker"] = ticker
        frame["ts"] = frame.index
        frame.index.name = None
        frames.append(frame.dropna())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values(["ts", "ticker"]).reset_index(drop=True)


def decile_lift(predictions: np.ndarray, realised: np.ndarray) -> list:
    """Ordena por magnitude prevista e mede a magnitude REAL em cada decil.

    E' a metrica mais decisiva para uso pratico: se as barras que o modelo poe
    no decil de cima nao se mexerem mais do que as do decil de baixo, o modelo
    nao serve para dimensionar posicoes nem para nada."""
    order = np.argsort(predictions)
    buckets = np.array_split(order, 10)
    return [
        {"decile": i + 1,
         "mean_predicted": float(predictions[bucket].mean()),
         "mean_realised": float(realised[bucket].mean()),
         "n": len(bucket)}
        for i, bucket in enumerate(buckets)
    ]


def volatility_persistence_baseline(train_df, test_df, target: str):
    """Baseline de uma linha: so' o racio entre volatilidade curta e longa."""
    feature = "vol_10" if "vol_10" in train_df.columns else "std_return"
    if feature not in train_df.columns:
        return None
    X_train = train_df[[feature]].values
    X_test = test_df[[feature]].values

    if target == "big_move":
        model = LogisticRegression(max_iter=1000).fit(X_train, train_df[target])
        probabilities = model.predict_proba(X_test)[:, 1]
        return {
            "accuracy": float(((probabilities >= 0.5).astype(int) == test_df[target].values).mean()),
            "auc": float(roc_auc_score(test_df[target], probabilities))
            if test_df[target].nunique() > 1 else None,
        }
    correlation = stats.spearmanr(X_test.ravel(), test_df[target].values).statistic
    return {"spearman": float(correlation)}


def train_one(conn, timeframe: str, model: str) -> dict | None:
    frame = build_dataset(conn, timeframe, model)
    if frame.empty:
        return None

    feature_names = tree_train.feature_names_for(model)
    train_df, validation_df, test_df = tree_train.split_by_time(frame, T["horizon_bars"])
    if len(train_df) < T["min_train_rows"] or validation_df.empty or test_df.empty:
        return None

    X_train, X_validation, X_test = (d[feature_names] for d in (train_df, validation_df, test_df))
    effective_days = int(pd.Series(test_df["ts"]).dt.normalize().nunique())
    standard_error = float(np.sqrt(0.25 / max(effective_days, 1)))

    # ---------------- classificacao: vem movimento grande? ----------------
    y_train = train_df["big_move"].astype(int)
    y_test = test_df["big_move"].astype(int)

    sweep = []
    for depth in T["depth_sweep"]:
        estimator = XGBClassifier(**dict(
            T["xgb_params"], max_depth=depth, objective="binary:logistic", num_class=None,
        ))
        started = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_seconds = time.perf_counter() - started
        probabilities = estimator.predict_proba(X_validation)[:, 1]
        sweep.append({
            "max_depth": depth,
            "auc": float(roc_auc_score(validation_df["big_move"], probabilities)),
            "accuracy": float(((probabilities >= 0.5).astype(int)
                               == validation_df["big_move"].values).mean()),
            "train_seconds": train_seconds,
        })

    best = max(sweep, key=lambda s: s["auc"])
    classifier = XGBClassifier(**dict(
        T["xgb_params"], max_depth=best["max_depth"], objective="binary:logistic", num_class=None,
    ))
    classifier.fit(pd.concat([X_train, X_validation]),
                   pd.concat([y_train, validation_df["big_move"].astype(int)]))

    started = time.perf_counter()
    test_probabilities = classifier.predict_proba(X_test)[:, 1]
    latency_ms = 1000 * (time.perf_counter() - started) / max(len(X_test), 1)

    accuracy = float(((test_probabilities >= 0.5).astype(int) == y_test.values).mean())
    auc = float(roc_auc_score(y_test, test_probabilities))
    base_rate = float(y_test.mean())
    majority_baseline = max(base_rate, 1 - base_rate)

    coverage_curve = []
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        selected = test_probabilities >= threshold
        if selected.sum() < 30:
            coverage_curve.append({"threshold": threshold, "coverage": float(selected.mean()),
                                   "precision": None, "lift": None, "n": int(selected.sum())})
            continue
        precision = float(y_test.values[selected].mean())
        coverage_curve.append({
            "threshold": threshold, "coverage": float(selected.mean()),
            "precision": precision, "lift": precision - base_rate, "n": int(selected.sum()),
        })

    # ---------------- regressao: quao grande? ----------------
    regressor = XGBRegressor(**{k: v for k, v in T["xgb_params"].items()
                                if k not in ("objective", "num_class")},
                             max_depth=best["max_depth"], objective="reg:squarederror")
    regressor.fit(pd.concat([X_train, X_validation]),
                  pd.concat([train_df["magnitude_ratio"], validation_df["magnitude_ratio"]]))
    predicted_magnitude = regressor.predict(X_test)
    realised = test_df["magnitude_ratio"].values

    spearman = float(stats.spearmanr(predicted_magnitude, realised).statistic)
    deciles = decile_lift(predicted_magnitude, realised)
    top_bottom = deciles[-1]["mean_realised"] / deciles[0]["mean_realised"] \
        if deciles[0]["mean_realised"] else None

    baselines = {
        "classification": volatility_persistence_baseline(train_df, test_df, "big_move"),
        "regression": volatility_persistence_baseline(train_df, test_df, "magnitude_ratio"),
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": classifier, "regressor": regressor,
                 "feature_names": feature_names, "kind": model, "timeframe": timeframe,
                 "max_depth": best["max_depth"]},
                config.MODELS_DIR / f"magnitude_{model}_{timeframe}.joblib")

    importances = sorted(zip(feature_names, classifier.feature_importances_),
                         key=lambda item: -item[1])[:10]

    return {
        "model": model, "timeframe": timeframe, "max_depth": best["max_depth"],
        "n_train": len(train_df), "n_validation": len(validation_df), "n_test": len(test_df),
        "effective_days": effective_days, "standard_error": standard_error,
        "accuracy": accuracy, "auc": auc, "base_rate": base_rate,
        "majority_baseline": majority_baseline,
        "significant": abs(accuracy - majority_baseline) > 2 * standard_error,
        "spearman": spearman, "top_bottom_ratio": top_bottom,
        "latency_ms": latency_ms,
        "coverage_curve": coverage_curve, "deciles": deciles, "sweep": sweep,
        "baselines": baselines,
        "top_features": [(name, float(value)) for name, value in importances],
    }


def store(conn, result: dict) -> str:
    version = (f"mag_{result['model']}_{result['timeframe']}_"
               f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}")
    baseline_auc = (result["baselines"]["classification"] or {}).get("auc")
    baseline_spearman = (result["baselines"]["regression"] or {}).get("spearman")
    conn.execute(
        """INSERT INTO magnitude_models
             (version, model_kind, timeframe, max_depth, trained_at, n_train, n_validation,
              n_test, effective_days, accuracy, auc, base_rate, majority_baseline,
              baseline_vol_auc, spearman, baseline_vol_spearman, top_bottom_ratio,
              standard_error, significant, latency_ms, coverage_curve_json, deciles_json,
              sweep_json, top_features_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (version, result["model"], result["timeframe"], result["max_depth"],
         datetime.now(timezone.utc).isoformat(), result["n_train"], result["n_validation"],
         result["n_test"], result["effective_days"], result["accuracy"], result["auc"],
         result["base_rate"], result["majority_baseline"], baseline_auc, result["spearman"],
         baseline_spearman, result["top_bottom_ratio"], result["standard_error"],
         int(result["significant"]), result["latency_ms"],
         json.dumps(result["coverage_curve"]), json.dumps(result["deciles"]),
         json.dumps(result["sweep"]), json.dumps(result["top_features"])),
    )
    conn.commit()
    return version


def _report(r: dict) -> None:
    print(f"\n=== MAGNITUDE / {r['model'].upper()} @ {r['timeframe']} ===")
    print(f"  treino={r['n_train']:,} teste={r['n_test']:,} | {r['effective_days']} dias efectivos "
          f"| profundidade {r['max_depth']} | {r['latency_ms']:.4f} ms/linha")

    baseline_auc = (r["baselines"]["classification"] or {}).get("auc")
    print(f"  CLASSIFICACAO (vem movimento grande?)")
    print(f"    accuracy {r['accuracy']:.1%}  vs classe maioritaria {r['majority_baseline']:.1%} "
          f"(+-{r['standard_error']*100:.1f}pp) -> "
          f"{'SIGNIFICATIVO' if r['significant'] else 'dentro do ruido'}")
    print(f"    AUC {r['auc']:.4f}  vs baseline de persistencia de volatilidade "
          f"{baseline_auc:.4f}" if baseline_auc else f"    AUC {r['auc']:.4f}")
    print(f"    taxa-base de movimento grande: {r['base_rate']:.1%}")
    print(f"    {'limiar':>7} {'cobertura':>10} {'precisao':>9} {'lift':>9} {'n':>8}")
    for point in r["coverage_curve"]:
        if point["precision"] is None:
            print(f"    {point['threshold']:>7.2f} {point['coverage']:>9.1%} {'--':>9} {'--':>9} {point['n']:>8,}")
        else:
            print(f"    {point['threshold']:>7.2f} {point['coverage']:>9.1%} "
                  f"{point['precision']:>8.1%} {100*point['lift']:>+8.1f}pp {point['n']:>8,}")

    baseline_spearman = (r["baselines"]["regression"] or {}).get("spearman")
    print(f"  REGRESSAO (quao grande?)")
    print(f"    Spearman {r['spearman']:.4f}" +
          (f"  vs baseline de volatilidade {baseline_spearman:.4f}" if baseline_spearman else ""))
    print(f"    decil de topo move-se {r['top_bottom_ratio']:.2f}x o decil de baixo")
    print(f"    {'decil':>6} {'previsto':>10} {'real':>9} {'n':>8}")
    for d in r["deciles"]:
        print(f"    {d['decile']:>6} {d['mean_predicted']:>10.3f} {d['mean_realised']:>9.3f} {d['n']:>8,}")
    print(f"  top features: {', '.join(n for n, _ in r['top_features'][:6])}")


if __name__ == "__main__":
    connection = database.get_connection()
    for timeframe in T["timeframes"]:
        for model in tree_train.MODELS:
            outcome = train_one(connection, timeframe, model)
            if outcome is None:
                print(f"magnitude {model} @ {timeframe}: dados insuficientes")
                continue
            store(connection, outcome)
            _report(outcome)
    connection.close()
