"""Inferencia ao vivo das duas arvores de decisao.

Devolve, para um ticker, a decisao de cada modelo (SOBE / DESCE / ESPERAR)
com a probabilidade das tres classes, e diz se os dois concordam.

O desacordo entre eles e' informacao util e nao um defeito: os dois modelos
veem a mesma janela de tempo mas descrevem-na de maneiras diferentes (um por
indicadores, outro pela forma do percurso), portanto concordarem e' um sinal
mais forte do que qualquer um deles sozinho.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.patterns.ingest_intraday import load_bars
from src.trees import features as F

import joblib
import pandas as pd

T = config.TREES
_CACHE = {}


def load_tree(model: str, timeframe: str, refresh: bool = False):
    key = (model, timeframe)
    if refresh or key not in _CACHE:
        path = config.MODELS_DIR / f"tree_{model}_{timeframe}.joblib"
        _CACHE[key] = joblib.load(path) if path.exists() else None
    return _CACHE[key]


def predict(conn, ticker: str, timeframe: str = None, model: str = None) -> dict | None:
    """Decisao do modelo para a barra mais recente disponivel."""
    timeframe = timeframe or T["timeframes"][0]
    bundle = load_tree(model, timeframe)
    if bundle is None:
        return None

    # Janela curta: chega para todas as features (a mais longa olha 60 barras
    # para tras) e mantem a inferencia em milissegundos.
    bars = load_bars(conn, ticker, timeframe, limit=200)
    if len(bars) < 80:
        return None

    started = time.perf_counter()
    builder = F.indicator_features if model == "indicators" else F.microstructure_features
    matrix = builder(bars).dropna()
    if matrix.empty:
        return None

    row = matrix.iloc[[-1]][bundle["feature_names"]]
    probabilities = bundle["model"].predict_proba(row)[0]
    latency_ms = 1000 * (time.perf_counter() - started)

    decision = int(probabilities.argmax())
    return {
        "model": model, "timeframe": timeframe, "ticker": ticker,
        "as_of_ts": matrix.index[-1].isoformat(),
        "last_price": float(bars["close"].iloc[-1]),
        "decision": decision,
        "decision_name": F.CLASS_NAMES[decision],
        "confidence": float(probabilities[decision]),
        "probabilities": {F.CLASS_NAMES[i]: float(p) for i, p in enumerate(probabilities)},
        "max_depth": bundle["max_depth"],
        "latency_ms": latency_ms,
    }


def predict_both(conn, ticker: str, timeframe: str = None) -> dict:
    timeframe = timeframe or T["timeframes"][0]
    indicators = predict(conn, ticker, timeframe, "indicators")
    microstructure = predict(conn, ticker, timeframe, "microstructure")

    agreement = None
    if indicators and microstructure:
        agreement = indicators["decision"] == microstructure["decision"]

    return {
        "ticker": ticker, "timeframe": timeframe,
        "indicators": indicators, "microstructure": microstructure,
        "agree": agreement,
        # Consenso: so' se atreve quando os dois dizem o mesmo E nao e'
        # ESPERAR. Duas leituras diferentes do mesmo grafico a concordarem
        # sao mais do que qualquer uma delas sozinha.
        "consensus": (indicators["decision_name"]
                      if agreement and indicators["decision"] != F.WAIT else "ESPERAR"),
    }


def model_summary(conn) -> list:
    rows = conn.execute(
        """SELECT * FROM tree_models WHERE version IN (
             SELECT version FROM tree_models t2
             WHERE t2.model_kind = tree_models.model_kind
               AND t2.timeframe = tree_models.timeframe
             ORDER BY trained_at DESC LIMIT 1)
           ORDER BY timeframe, model_kind"""
    ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    connection = database.get_connection()
    timeframe = T["timeframes"][0]
    print(f"decisoes ao vivo ({timeframe}):\n")
    print(f"{'TICKER':8s} {'INDICADORES':>26s} {'FORMA DA JANELA':>26s}   CONSENSO")
    for ticker in config.TICKERS:
        result = predict_both(connection, ticker, timeframe)
        a, b = result["indicators"], result["microstructure"]
        if not a or not b:
            print(f"{ticker:8s} modelo por treinar")
            continue
        print(f"{ticker:8s} {a['decision_name']:>12s} {a['confidence']:>6.1%}     "
              f"{b['decision_name']:>12s} {b['confidence']:>6.1%}     {result['consensus']}"
              f"{'  <- concordam' if result['agree'] else ''}")
    connection.close()
