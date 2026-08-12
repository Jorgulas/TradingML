import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src import simulator

HORIZON = "SHORT"  # regras identicas para LONG, so' muda stop_loss_pct/threshold


def _set_price(conn, ticker, date, close):
    conn.execute(
        "INSERT INTO prices (ticker, date, close, ingested_at) VALUES (?, ?, ?, ?)",
        (ticker, date, close, datetime.now(timezone.utc).isoformat()),
    )


def _set_prediction(conn, ticker, date, horizon, direction, confidence):
    conn.execute(
        """INSERT INTO predictions (ticker, date, horizon, predicted_direction, confidence, model_version, feature_snapshot, created_at)
           VALUES (?, ?, ?, ?, ?, 'test', '{}', ?)""",
        (ticker, date, horizon, direction, confidence, datetime.now(timezone.utc).isoformat()),
    )


def _set_position(conn, ticker, horizon, qty, avg_price, opened_date):
    conn.execute(
        "INSERT INTO positions (ticker, horizon, qty, avg_price, opened_date) VALUES (?, ?, ?, ?, ?)",
        (ticker, horizon, qty, avg_price, opened_date),
    )


def _set_cash(conn, horizon, cash):
    conn.execute("UPDATE portfolio_state SET cash = ? WHERE horizon = ?", (cash, horizon))
    conn.commit()


def test_fresh_entry_buy_sizes_8pct_of_equity(conn):
    _set_price(conn, "AAPL", "2024-06-03", 100.0)
    _set_prediction(conn, "AAPL", "2024-06-03", HORIZON, direction=1, confidence=0.8)
    conn.commit()

    result = simulator.simulate_day(conn, "2024-06-03", HORIZON)

    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade == {"ticker": "AAPL", "side": "BUY", "reason": "signal_entry", "qty": pytest.approx(80.0), "price": 100.0}

    pos = conn.execute("SELECT * FROM positions WHERE ticker='AAPL' AND horizon=?", (HORIZON,)).fetchone()
    assert pos["qty"] == pytest.approx(80.0)
    assert pos["avg_price"] == pytest.approx(100.0)

    cash = conn.execute("SELECT cash FROM portfolio_state WHERE horizon=?", (HORIZON,)).fetchone()["cash"]
    assert cash == pytest.approx(100_000.0 - 8_000.0)

    # uma BUY nao muda o equity total (so' converte cash em posicao, sem fees)
    assert result["total_value"] == pytest.approx(100_000.0)
    assert result["pnl_pct"] == pytest.approx(0.0)


def test_low_confidence_signal_does_not_trigger_entry(conn):
    _set_price(conn, "AAPL", "2024-06-03", 100.0)
    _set_prediction(conn, "AAPL", "2024-06-03", HORIZON, direction=1, confidence=0.55)  # < 0.6 threshold
    conn.commit()

    result = simulator.simulate_day(conn, "2024-06-03", HORIZON)

    assert result["trades"] == []
    assert conn.execute("SELECT COUNT(*) FROM positions WHERE horizon=?", (HORIZON,)).fetchone()[0] == 0


def test_stop_loss_triggers_at_exactly_minus_10_pct(conn):
    _set_position(conn, "AAPL", HORIZON, qty=80.0, avg_price=100.0, opened_date="2024-06-01")
    _set_cash(conn, HORIZON, 92_000.0)
    _set_price(conn, "AAPL", "2024-06-04", 90.0)  # exatamente -10%
    # sinal diz para manter/comprar mais -- stop-loss tem de ganhar prioridade na mesma
    _set_prediction(conn, "AAPL", "2024-06-04", HORIZON, direction=1, confidence=0.9)
    conn.commit()

    result = simulator.simulate_day(conn, "2024-06-04", HORIZON)

    assert result["trades"] == [{"ticker": "AAPL", "side": "SELL", "reason": "stop_loss", "qty": pytest.approx(80.0), "price": 90.0}]
    assert conn.execute("SELECT COUNT(*) FROM positions WHERE horizon=?", (HORIZON,)).fetchone()[0] == 0
    cash = conn.execute("SELECT cash FROM portfolio_state WHERE horizon=?", (HORIZON,)).fetchone()["cash"]
    assert cash == pytest.approx(92_000.0 + 80.0 * 90.0)


def test_price_just_above_stop_loss_does_not_sell(conn):
    _set_position(conn, "AAPL", HORIZON, qty=80.0, avg_price=100.0, opened_date="2024-06-01")
    _set_cash(conn, HORIZON, 92_000.0)
    _set_price(conn, "AAPL", "2024-06-04", 90.01)  # mesmo mesmo por 1 cent acima do stop
    conn.commit()

    result = simulator.simulate_day(conn, "2024-06-04", HORIZON)

    assert result["trades"] == []
    assert conn.execute("SELECT COUNT(*) FROM positions WHERE horizon=?", (HORIZON,)).fetchone()[0] == 1


def test_signal_flip_sells_existing_position(conn):
    _set_position(conn, "AAPL", HORIZON, qty=80.0, avg_price=100.0, opened_date="2024-06-01")
    _set_cash(conn, HORIZON, 92_000.0)
    _set_price(conn, "AAPL", "2024-06-04", 98.0)  # acima do stop-loss (90), nao dispara por ai
    _set_prediction(conn, "AAPL", "2024-06-04", HORIZON, direction=0, confidence=0.7)
    conn.commit()

    result = simulator.simulate_day(conn, "2024-06-04", HORIZON)

    assert result["trades"] == [{"ticker": "AAPL", "side": "SELL", "reason": "signal_flip", "qty": pytest.approx(80.0), "price": 98.0}]
    cash = conn.execute("SELECT cash FROM portfolio_state WHERE horizon=?", (HORIZON,)).fetchone()["cash"]
    assert cash == pytest.approx(92_000.0 + 80.0 * 98.0)


def test_insufficient_cash_skips_entry_without_crashing(conn):
    _set_position(conn, "MSFT", HORIZON, qty=1000.0, avg_price=95.0, opened_date="2024-06-01")
    _set_cash(conn, HORIZON, 500.0)
    _set_price(conn, "MSFT", "2024-06-04", 95.0)
    _set_price(conn, "AAPL", "2024-06-04", 50.0)
    _set_prediction(conn, "AAPL", "2024-06-04", HORIZON, direction=1, confidence=0.9)
    conn.commit()

    # equity_at_start = 500 (cash) + 1000*95 (MSFT) = 95500 -> target 8% = 7640 > cash(500)
    result = simulator.simulate_day(conn, "2024-06-04", HORIZON)

    assert result["trades"] == []  # nenhuma entrada nova, mas nao rebenta
    assert conn.execute("SELECT COUNT(*) FROM positions WHERE ticker='AAPL' AND horizon=?", (HORIZON,)).fetchone()[0] == 0
    warn = conn.execute(
        "SELECT * FROM run_log WHERE stage='simulator' AND status='WARN' AND message LIKE '%AAPL%'"
    ).fetchone()
    assert warn is not None


def test_simulate_day_is_idempotent_when_rerun_same_date(conn):
    _set_price(conn, "AAPL", "2024-06-03", 100.0)
    _set_prediction(conn, "AAPL", "2024-06-03", HORIZON, direction=1, confidence=0.8)
    conn.commit()

    first = simulator.simulate_day(conn, "2024-06-03", HORIZON)
    second = simulator.simulate_day(conn, "2024-06-03", HORIZON)

    assert first["trades"] != []
    assert second.get("skipped") == "already simulated"
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM equity_curve").fetchone()[0] == 1
    assert second["total_value"] == pytest.approx(first["total_value"])


def test_equity_curve_written_even_without_any_trade(conn):
    # dia sem sinal nenhum -- ainda assim tem de haver mark-to-market diario
    _set_price(conn, "AAPL", "2024-06-03", 100.0)
    conn.commit()

    simulator.simulate_day(conn, "2024-06-03", HORIZON)

    row = conn.execute("SELECT * FROM equity_curve WHERE date='2024-06-03' AND horizon=?", (HORIZON,)).fetchone()
    assert row is not None
    assert row["total_value"] == pytest.approx(config.STARTING_CASH)


# --- LONG horizon: mesma logica, stop-loss mais largo (-20% em vez de -10%) ---

def test_long_horizon_stop_loss_triggers_at_exactly_minus_20_pct(conn):
    _set_position(conn, "AAPL", "LONG", qty=80.0, avg_price=100.0, opened_date="2024-06-01")
    _set_cash(conn, "LONG", 92_000.0)
    _set_price(conn, "AAPL", "2024-06-04", 80.0)  # exatamente -20%
    conn.commit()

    result = simulator.simulate_day(conn, "2024-06-04", "LONG")

    assert result["trades"] == [{"ticker": "AAPL", "side": "SELL", "reason": "stop_loss", "qty": pytest.approx(80.0), "price": 80.0}]


def test_long_horizon_minus_10pct_alone_does_not_trigger_its_wider_stop(conn):
    # -10% dispararia o stop de SHORT mas NAO o de LONG (-20%) -- confirma
    # que os dois horizontes usam mesmo parametros diferentes, nao partilhados.
    _set_position(conn, "AAPL", "LONG", qty=80.0, avg_price=100.0, opened_date="2024-06-01")
    _set_cash(conn, "LONG", 92_000.0)
    _set_price(conn, "AAPL", "2024-06-04", 90.0)
    conn.commit()

    result = simulator.simulate_day(conn, "2024-06-04", "LONG")

    assert result["trades"] == []
    assert conn.execute("SELECT COUNT(*) FROM positions WHERE horizon='LONG'").fetchone()[0] == 1


def test_short_and_long_portfolios_are_fully_independent(conn):
    # a mesma ticker pode estar comprada a LONG e de fora a SHORT ao mesmo tempo
    _set_position(conn, "AAPL", "LONG", qty=80.0, avg_price=100.0, opened_date="2024-06-01")
    _set_price(conn, "AAPL", "2024-06-04", 100.0)
    _set_prediction(conn, "AAPL", "2024-06-04", "SHORT", direction=1, confidence=0.8)
    conn.commit()

    simulator.simulate_day(conn, "2024-06-04", "SHORT")

    short_cash = conn.execute("SELECT cash FROM portfolio_state WHERE horizon='SHORT'").fetchone()["cash"]
    long_cash = conn.execute("SELECT cash FROM portfolio_state WHERE horizon='LONG'").fetchone()["cash"]
    assert short_cash == pytest.approx(100_000.0 - 8_000.0)  # SHORT comprou
    assert long_cash == pytest.approx(100_000.0)  # LONG nao foi tocado
    assert conn.execute("SELECT COUNT(*) FROM positions WHERE horizon='SHORT'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM positions WHERE horizon='LONG'").fetchone()[0] == 1
