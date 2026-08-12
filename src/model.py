"""Treino/load/predict por horizonte.

Modelo principal: LogisticRegression(L2, class_weight='balanced') -- com so'
8 tickers, uma arvore isola facilmente as poucas vezes que um boolean de
noticias e' True e trata isso como regra forte (overfitting em categorias
raras). L2 encolhe coeficientes para perto de zero a menos que o sinal seja
consistente em muitas linhas -- e' a robustez que os booleanos esparsos
precisam, e da' coeficientes interpretaveis. RandomForest fica como modelo
secundario so' de comparacao (cv_accuracy registada em notes), nunca decide
trades.

Validacao sempre TimeSeriesSplit (walk-forward), nunca k-fold aleatorio --
os dados sao series temporais, um shuffle "vaza" o futuro para o treino.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from db import database
from src import features

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_training_frame(conn, horizon: str) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT ticker, date, direction FROM outcomes "
        "WHERE horizon = ? AND direction IS NOT NULL ORDER BY ticker, date",
        (horizon,),
    ).fetchall()
    records = []
    for r in rows:
        vec = features.build_feature_vector(conn, r["ticker"], r["date"], horizon)
        if vec is None:
            continue
        vec["ticker"] = r["ticker"]
        vec["date"] = r["date"]
        vec["direction"] = r["direction"]
        records.append(vec)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["date", "ticker"]).reset_index(drop=True)


def _make_logistic_pipeline() -> Pipeline:
    # penalty='l2' e' o default do LogisticRegression -- omitido de proposito
    # (passa-lo explicitamente ficou deprecated no sklearn 1.8+ a favor de l1_ratio/C).
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        max_depth=5, min_samples_leaf=20, n_estimators=200, random_state=42, class_weight="balanced"
    )


def train(conn, horizon: str) -> dict | None:
    frame = build_training_frame(conn, horizon)
    if len(frame) < config.MIN_TRAIN_ROWS:
        database.log_run(conn, datetime.now(timezone.utc).strftime("%Y-%m-%d"), "model_train", "WARN",
                          f"{horizon}: apenas {len(frame)} linhas rotuladas, abaixo do minimo "
                          f"({config.MIN_TRAIN_ROWS}) -- treino adiado")
        return None

    feat_cols = features.feature_names(horizon)
    X = frame[feat_cols]
    y = frame["direction"]

    majority_class = y.mode().iloc[0]
    baseline_majority_acc = float((y == majority_class).mean())

    persistence_pred = frame.groupby("ticker")["direction"].shift(1)
    valid = persistence_pred.notna()
    baseline_persistence_acc = (
        float((persistence_pred[valid] == y[valid]).mean()) if valid.any() else None
    )

    # Embargo entre treino e teste de cada split: os horizontes com
    # predict_ahead_days > 1 usam labels de janela sobreposta (o outcome de D
    # e o de D+1 partilham quase toda a janela de 20 sessoes), o que sem gap
    # deixa a fronteira treino/teste otimista (quase-leakage). gap aproxima
    # "ahead sessoes" em numero de linhas, dado que o frame intercala tickers.
    ahead = config.HORIZON_PARAMS[horizon]["predict_ahead_days"]
    rows_per_date = max(1, int(frame.groupby("date").size().median()))
    gap = ahead * rows_per_date if ahead > 1 else 0
    n_splits = min(5, max(2, len(frame) // max(30, gap + 10)))
    try:
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        splits = list(tscv.split(X))
    except ValueError:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(X))

    logistic_scores, rf_scores = [], []
    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        logistic = _make_logistic_pipeline()
        logistic.fit(X_train, y_train)
        logistic_scores.append(logistic.score(X_test, y_test))

        rf = _make_rf()
        rf.fit(X_train, y_train)
        rf_scores.append(rf.score(X_test, y_test))

    cv_accuracy_logistic = float(np.mean(logistic_scores))
    cv_accuracy_rf = float(np.mean(rf_scores))

    final_model = _make_logistic_pipeline()
    final_model.fit(X, y)

    version = f"{horizon}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / f"model_{horizon}.joblib"
    joblib.dump({"model": final_model, "feature_names": feat_cols, "version": version}, model_path)

    conn.execute(
        """INSERT INTO model_versions (version, horizon, trained_at, algorithm, hyperparams_json,
             train_start_date, train_end_date, n_train_rows, cv_accuracy,
             baseline_majority_accuracy, baseline_persistence_accuracy, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            version, horizon, datetime.now(timezone.utc).isoformat(), "logistic_regression",
            json.dumps({"penalty": "l2", "class_weight": "balanced"}),
            str(frame["date"].min()), str(frame["date"].max()), len(frame), cv_accuracy_logistic,
            baseline_majority_acc, baseline_persistence_acc,
            f"random_forest_comparison_cv_accuracy={cv_accuracy_rf:.4f} (nao decide trades)",
        ),
    )
    conn.commit()

    return {
        "version": version, "n_rows": len(frame), "n_splits": n_splits,
        "cv_accuracy_logistic": cv_accuracy_logistic, "cv_accuracy_rf": cv_accuracy_rf,
        "baseline_majority": baseline_majority_acc, "baseline_persistence": baseline_persistence_acc,
    }


def load_model(horizon: str):
    path = config.MODELS_DIR / f"model_{horizon}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def predict(conn, ticker: str, date: str, horizon: str):
    """Devolve {predicted_direction, confidence, model_version, feature_snapshot}
    ou None se nao houver modelo treinado ou faltarem features para esta data."""
    bundle = load_model(horizon)
    if bundle is None:
        return None
    vec = features.build_feature_vector(conn, ticker, date, horizon)
    if vec is None:
        return None
    X = pd.DataFrame([vec])[bundle["feature_names"]]
    model = bundle["model"]
    proba = model.predict_proba(X)[0]
    classes = list(model.named_steps["clf"].classes_)
    p_up = proba[classes.index(1)]
    predicted_direction = 1 if p_up >= 0.5 else 0
    confidence = p_up if predicted_direction == 1 else 1 - p_up
    return {
        "predicted_direction": predicted_direction,
        "confidence": float(confidence),
        "model_version": bundle["version"],
        "feature_snapshot": vec,
    }


def record_prediction(conn, ticker: str, date: str, horizon: str, result: dict) -> int:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO predictions (ticker, date, horizon, predicted_direction, confidence, model_version, feature_snapshot, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(ticker, date, horizon) DO UPDATE SET
             predicted_direction = excluded.predicted_direction, confidence = excluded.confidence,
             model_version = excluded.model_version, feature_snapshot = excluded.feature_snapshot,
             created_at = excluded.created_at""",
        (
            ticker, date, horizon, result["predicted_direction"], result["confidence"],
            result["model_version"], json.dumps(result["feature_snapshot"]), now,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM predictions WHERE ticker = ? AND date = ? AND horizon = ?",
        (ticker, date, horizon),
    ).fetchone()
    return row["id"]


def predict_and_record_all(conn, date: str, horizon: str, tickers=None) -> dict:
    tickers = tickers or config.TICKERS
    results = {}
    for ticker in tickers:
        result = predict(conn, ticker, date, horizon)
        if result is None:
            results[ticker] = None
            continue
        record_prediction(conn, ticker, date, horizon, result)
        results[ticker] = result
    return results


if __name__ == "__main__":
    connection = database.get_connection()
    for hz in config.HORIZONS:
        outcome = train(connection, hz)
        if outcome is None:
            print(f"{hz}: treino adiado (dados insuficientes)")
        else:
            print(
                f"{hz}: n={outcome['n_rows']} splits={outcome['n_splits']} "
                f"logistic_cv={outcome['cv_accuracy_logistic']:.4f} rf_cv={outcome['cv_accuracy_rf']:.4f} "
                f"baseline_majority={outcome['baseline_majority']:.4f} "
                f"baseline_persistence={outcome['baseline_persistence']}"
            )
    connection.close()
