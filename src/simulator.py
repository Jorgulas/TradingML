"""Motor de simulacao de trading (paper trading, sem dinheiro real) por
horizonte. Le a previsao do dia (ja gravada em `predictions` por src/model.py)
e decide compra/venda simulada por ticker, nesta ordem de prioridade
(identica todos os dias, por ticker):

  1. stop-loss do horizonte disparado (posicao existente)      -> SELL
  2. modelo confiante em baixa + posicao existente              -> SELL
  3. modelo confiante em alta + sem posicao + caixa suficiente  -> BUY
  4. senao                                                      -> HOLD

Idempotente por (date, horizon): se ja existe uma linha em equity_curve para
essa combinacao, simulate_day() e' um no-op que devolve o resultado ja'
gravado -- assim correr o pipeline duas vezes no mesmo dia nunca duplica
trades nem volta a comprar algo que ja' foi vendido nesse dia.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from db import database


def _insert_trade(conn, ticker, date, horizon, side, qty, price, cash_after, reason, prediction_id, created_at):
    conn.execute(
        """INSERT INTO trades (ticker, date, horizon, side, qty, price, cash_after, reason, prediction_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ticker, date, horizon, side, qty, price, cash_after, reason, prediction_id, created_at),
    )


def simulate_day(conn, date: str, horizon: str) -> dict:
    existing = conn.execute(
        "SELECT * FROM equity_curve WHERE date = ? AND horizon = ?", (date, horizon)
    ).fetchone()
    if existing is not None:
        return {
            "date": date, "horizon": horizon, "trades": [], "skipped": "already simulated",
            "cash": existing["cash"], "positions_value": existing["positions_value"],
            "total_value": existing["total_value"], "pnl_pct": existing["pnl_pct"],
        }

    params = config.HORIZON_PARAMS[horizon]

    state = conn.execute("SELECT cash FROM portfolio_state WHERE horizon = ?", (horizon,)).fetchone()
    if state is None:
        raise RuntimeError(f"portfolio_state em falta para horizon={horizon} -- corre database.seed_portfolio_state()")
    cash = state["cash"]

    positions = {
        r["ticker"]: {"qty": r["qty"], "avg_price": r["avg_price"], "opened_date": r["opened_date"]}
        for r in conn.execute("SELECT * FROM positions WHERE horizon = ?", (horizon,)).fetchall()
    }
    prices_today = {
        r["ticker"]: r["close"]
        for r in conn.execute("SELECT ticker, close FROM prices WHERE date = ?", (date,)).fetchall()
    }
    predictions_today = {
        r["ticker"]: {"predicted_direction": r["predicted_direction"], "confidence": r["confidence"], "id": r["id"]}
        for r in conn.execute(
            "SELECT id, ticker, predicted_direction, confidence FROM predictions WHERE date = ? AND horizon = ?",
            (date, horizon),
        ).fetchall()
    }

    positions_value_start = sum(p["qty"] * prices_today.get(t, p["avg_price"]) for t, p in positions.items())
    equity_at_start = cash + positions_value_start

    now = datetime.now(timezone.utc).isoformat()
    trades_made = []

    for ticker in config.TICKERS:
        price = prices_today.get(ticker)
        if price is None:
            continue  # sem barra hoje para este ticker -- nao mexe

        held = positions.get(ticker)
        pred = predictions_today.get(ticker)

        if held is not None:
            stop_price = held["avg_price"] * (1 - params["stop_loss_pct"])
            if price <= stop_price:
                cash += held["qty"] * price
                _insert_trade(conn, ticker, date, horizon, "SELL", held["qty"], price, cash, "stop_loss", None, now)
                trades_made.append({"ticker": ticker, "side": "SELL", "reason": "stop_loss", "qty": held["qty"], "price": price})
                del positions[ticker]
                continue
            if pred and pred["predicted_direction"] == 0 and pred["confidence"] >= params["confidence_threshold"]:
                cash += held["qty"] * price
                _insert_trade(conn, ticker, date, horizon, "SELL", held["qty"], price, cash, "signal_flip", pred["id"], now)
                trades_made.append({"ticker": ticker, "side": "SELL", "reason": "signal_flip", "qty": held["qty"], "price": price})
                del positions[ticker]
            continue

        if pred and pred["predicted_direction"] == 1 and pred["confidence"] >= params["confidence_threshold"]:
            target_value = equity_at_start * params["position_size_pct"]
            if target_value > cash:
                database.log_run(conn, date, "simulator", "WARN",
                                  f"{horizon} {ticker}: sem caixa suficiente para nova posicao "
                                  f"({target_value:.2f} > {cash:.2f})")
                continue
            qty = target_value / price
            cash -= target_value
            positions[ticker] = {"qty": qty, "avg_price": price, "opened_date": date}
            _insert_trade(conn, ticker, date, horizon, "BUY", qty, price, cash, "signal_entry", pred["id"], now)
            trades_made.append({"ticker": ticker, "side": "BUY", "reason": "signal_entry", "qty": qty, "price": price})

    conn.execute("DELETE FROM positions WHERE horizon = ?", (horizon,))
    for ticker, pos in positions.items():
        conn.execute(
            "INSERT INTO positions (ticker, horizon, qty, avg_price, opened_date) VALUES (?, ?, ?, ?, ?)",
            (ticker, horizon, pos["qty"], pos["avg_price"], pos["opened_date"]),
        )
    conn.execute(
        "UPDATE portfolio_state SET cash = ?, updated_at = ? WHERE horizon = ?", (cash, now, horizon)
    )

    positions_value_end = sum(p["qty"] * prices_today.get(t, p["avg_price"]) for t, p in positions.items())
    total_value = cash + positions_value_end
    pnl_abs = total_value - config.STARTING_CASH
    pnl_pct = pnl_abs / config.STARTING_CASH
    conn.execute(
        """INSERT INTO equity_curve (date, horizon, cash, positions_value, total_value, pnl_abs, pnl_pct, num_positions)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(date, horizon) DO UPDATE SET
             cash = excluded.cash, positions_value = excluded.positions_value, total_value = excluded.total_value,
             pnl_abs = excluded.pnl_abs, pnl_pct = excluded.pnl_pct, num_positions = excluded.num_positions""",
        (date, horizon, cash, positions_value_end, total_value, pnl_abs, pnl_pct, len(positions)),
    )
    conn.commit()

    return {
        "date": date, "horizon": horizon, "trades": trades_made, "cash": cash,
        "positions_value": positions_value_end, "total_value": total_value, "pnl_pct": pnl_pct,
    }


if __name__ == "__main__":
    connection = database.get_connection()
    latest_date = connection.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    for hz in config.HORIZONS:
        result = simulate_day(connection, latest_date, hz)
        print(hz, result)
    connection.close()
