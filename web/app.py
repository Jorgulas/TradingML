"""Dashboard web do TradingML -- Flask, estritamente so-leitura (todas as
escritas acontecem via os scripts em src/, nunca por uma rota Flask, o que
evita problemas de concorrencia de escrita no SQLite).

Uso: py web/app.py   (depois abrir http://localhost:5000)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from db import database
from src.patterns import live as patterns_live
from src.patterns.ingest_intraday import load_bars

from flask import Flask, g, jsonify, render_template, request

app = Flask(__name__)


def get_db():
    if "db" not in g:
        g.db = database.get_connection()
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def _latest_date(conn, table: str) -> str:
    return conn.execute(f"SELECT MAX(date) FROM {table}").fetchone()[0]


@app.route("/")
def index():
    return render_template(
        "index.html",
        tickers=config.TICKERS,
        horizons=config.HORIZONS,
        currency_symbol=config.CURRENCY_SYMBOL,
        starting_cash=config.STARTING_CASH,
    )


@app.route("/api/summary")
def api_summary():
    conn = get_db()
    latest_equity_date = _latest_date(conn, "equity_curve")

    horizons_data = {}
    for horizon in config.HORIZONS:
        row = conn.execute(
            "SELECT * FROM equity_curve WHERE horizon = ? ORDER BY date DESC LIMIT 1", (horizon,)
        ).fetchone()
        model_row = conn.execute(
            """SELECT version, trained_at, cv_accuracy, baseline_majority_accuracy,
                      baseline_persistence_accuracy, n_train_rows, algorithm
               FROM model_versions WHERE horizon = ? ORDER BY trained_at DESC LIMIT 1""",
            (horizon,),
        ).fetchone()
        horizons_data[horizon] = {
            "cash": row["cash"] if row else config.STARTING_CASH,
            "positions_value": row["positions_value"] if row else 0.0,
            "total_value": row["total_value"] if row else config.STARTING_CASH,
            "pnl_abs": row["pnl_abs"] if row else 0.0,
            "pnl_pct": row["pnl_pct"] if row else 0.0,
            "num_positions": row["num_positions"] if row else 0,
            "as_of": row["date"] if row else None,
            "model": dict(model_row) if model_row else None,
            "params": config.HORIZON_PARAMS[horizon],
        }

    last_run = conn.execute("SELECT * FROM run_log ORDER BY created_at DESC LIMIT 1").fetchone()
    recent_errors = conn.execute(
        "SELECT * FROM run_log WHERE status = 'ERROR' ORDER BY created_at DESC LIMIT 5"
    ).fetchall()

    missing_news = []
    if latest_equity_date:
        missing_news = [
            r["ticker"] for r in conn.execute(
                """SELECT w.ticker FROM watchlist w WHERE w.active = 1 AND NOT EXISTS
                   (SELECT 1 FROM news_features nf WHERE nf.ticker = w.ticker AND nf.date = ?)""",
                (latest_equity_date,),
            ).fetchall()
        ]

    return jsonify({
        "horizons": horizons_data,
        "latest_date": latest_equity_date,
        "starting_cash": config.STARTING_CASH,
        "currency_symbol": config.CURRENCY_SYMBOL,
        "last_run": dict(last_run) if last_run else None,
        "recent_errors": [dict(r) for r in recent_errors],
        "tickers_missing_news_today": missing_news,
    })


@app.route("/api/equity_curve")
def api_equity_curve():
    conn = get_db()
    rows = conn.execute(
        "SELECT date, horizon, total_value, pnl_pct FROM equity_curve ORDER BY date"
    ).fetchall()
    series = {h: [] for h in config.HORIZONS}
    for r in rows:
        if r["horizon"] in series:
            series[r["horizon"]].append({"date": r["date"], "total_value": r["total_value"], "pnl_pct": r["pnl_pct"]})
    return jsonify(series)


@app.route("/api/positions")
def api_positions():
    conn = get_db()
    result = {}
    for horizon in config.HORIZONS:
        items = []
        for p in conn.execute("SELECT * FROM positions WHERE horizon = ? ORDER BY ticker", (horizon,)).fetchall():
            price_row = conn.execute(
                "SELECT close FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT 1", (p["ticker"],)
            ).fetchone()
            latest_price = price_row["close"] if price_row else p["avg_price"]
            unrealized_pnl = (latest_price - p["avg_price"]) * p["qty"]
            items.append({
                "ticker": p["ticker"],
                "qty": p["qty"],
                "avg_price": p["avg_price"],
                "latest_price": latest_price,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": (latest_price / p["avg_price"] - 1) if p["avg_price"] else 0.0,
                "opened_date": p["opened_date"],
            })
        result[horizon] = items
    return jsonify(result)


@app.route("/api/trades")
def api_trades():
    conn = get_db()
    rows = conn.execute(
        """SELECT ticker, date, horizon, side, qty, price, reason, created_at
           FROM trades ORDER BY created_at DESC LIMIT 30"""
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/signals")
def api_signals():
    conn = get_db()
    latest_date = _latest_date(conn, "predictions")
    if latest_date is None:
        return jsonify({"date": None, "tickers": []})

    news = {
        r["ticker"]: dict(r)
        for r in conn.execute("SELECT * FROM news_features WHERE date = ?", (latest_date,)).fetchall()
    }

    tickers_out = []
    for ticker in config.TICKERS:
        entry = {"ticker": ticker, "news": news.get(ticker), "signals": {}}
        for horizon in config.HORIZONS:
            pred = conn.execute(
                """SELECT predicted_direction, confidence FROM predictions
                   WHERE ticker = ? AND date = ? AND horizon = ?""",
                (ticker, latest_date, horizon),
            ).fetchone()
            entry["signals"][horizon] = dict(pred) if pred else None
        tickers_out.append(entry)

    return jsonify({"date": latest_date, "tickers": tickers_out, "boolean_features": config.BOOLEAN_FEATURES})


# ---------------------------------------------------------------------------
# Padroes graficos (subsistema independente)
# ---------------------------------------------------------------------------

@app.route("/patterns")
def patterns_page():
    return render_template(
        "patterns.html",
        tickers=config.TICKERS,
        timeframes=list(config.PATTERN_TIMEFRAMES),
        default_timeframe=config.PATTERN_LIVE_TIMEFRAME,
        pattern_types=config.PATTERN_TYPES,
    )


@app.route("/api/patterns/<ticker>")
def api_patterns(ticker):
    ticker = ticker.upper()
    if ticker not in config.TICKERS:
        return jsonify({"error": f"ticker desconhecido: {ticker}"}), 404
    timeframe = request.args.get("timeframe", config.PATTERN_LIVE_TIMEFRAME)
    if timeframe not in config.PATTERN_TIMEFRAMES:
        return jsonify({"error": f"timeframe desconhecido: {timeframe}"}), 400
    return jsonify(patterns_live.analyse(get_db(), ticker, timeframe))


@app.route("/api/patterns/<ticker>/bars")
def api_pattern_bars(ticker):
    """Barras recentes para desenhar o grafico (so' o necessario, nao o
    historico todo -- a pagina refresca de poucos em poucos segundos)."""
    ticker = ticker.upper()
    if ticker not in config.TICKERS:
        return jsonify({"error": f"ticker desconhecido: {ticker}"}), 404
    timeframe = request.args.get("timeframe", config.PATTERN_LIVE_TIMEFRAME)
    if timeframe not in config.PATTERN_TIMEFRAMES:
        return jsonify({"error": f"timeframe desconhecido: {timeframe}"}), 400
    limit = min(int(request.args.get("limit", 300)), 1000)

    bars = load_bars(get_db(), ticker, timeframe, limit=limit)
    if bars.empty:
        return jsonify({"bars": []})
    return jsonify({
        "bars": [
            {"ts": ts.isoformat(), "open": row["open"], "high": row["high"],
             "low": row["low"], "close": row["close"]}
            for ts, row in bars.iterrows()
        ]
    })


@app.route("/api/patterns/model")
def api_pattern_model():
    """Estado do modelo de sequencia: densidade da matriz e -- sobretudo -- a
    comparacao honesta contra o baseline de frequencia."""
    from src.patterns import sequence

    conn = get_db()
    out = {}
    for timeframe in config.PATTERN_TIMEFRAMES:
        transitions = sequence.load_transitions(conn, timeframe)
        n_patterns = conn.execute(
            "SELECT COUNT(*) FROM detected_patterns WHERE timeframe = ?", (timeframe,)
        ).fetchone()[0]
        metrics = sequence.evaluate(conn, timeframe)
        out[timeframe] = {
            "n_patterns": n_patterns,
            "n_transition_cells": len(transitions),
            "total_cells": len(config.PATTERN_TYPES) ** 2,
            "markov_top1_accuracy": metrics.get("markov_top1_accuracy"),
            "baseline_frequency_accuracy": metrics.get("baseline_frequency_accuracy"),
            "baseline_pattern": metrics.get("baseline_pattern"),
            "n_test": metrics.get("n_test"),
        }
    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
