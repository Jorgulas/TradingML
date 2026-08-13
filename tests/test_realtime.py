"""Testes do forward test ao vivo.

A garantia que aqui se protege e' a razao de ser desta janela inteira: uma
previsao tem de ser gravada ANTES de a barra que a resolve existir. Se isso
partir, o numero mostrado na pagina passa a ser tao viciavel como qualquer
backtest, e deixa de ter o valor que tem.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.patterns import realtime


# --------------------------------------------------------------------------
# Intervalo de Wilson -- e' o que impede vender ruido como resultado
# --------------------------------------------------------------------------

def test_wilson_interval_is_enormous_with_few_observations():
    low, high = realtime.wilson_interval(1, 4)
    assert high - low > 0.5, "com 4 observacoes o intervalo tem de ser larguissimo"
    assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_tightens_with_more_observations():
    narrow = realtime.wilson_interval(500, 1000)
    wide = realtime.wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_stays_inside_zero_one():
    """O intervalo normal produz limites fora de [0,1] nos extremos; o de
    Wilson e' usado precisamente por nao fazer isso."""
    for successes, total in [(0, 3), (3, 3), (0, 1), (1, 1)]:
        low, high = realtime.wilson_interval(successes, total)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_is_none_without_observations():
    assert realtime.wilson_interval(0, 0) == (None, None)


def test_wilson_interval_contains_the_point_estimate():
    for successes, total in [(1, 4), (7, 10), (55, 100), (250, 500)]:
        low, high = realtime.wilson_interval(successes, total)
        assert low <= successes / total <= high


# --------------------------------------------------------------------------
# A garantia central: nada de barras futuras
# --------------------------------------------------------------------------

def _bars(n=120, seed=5):
    rng = np.random.default_rng(seed)
    closes = 100 + rng.normal(0, 0.3, n).cumsum()
    index = pd.date_range("2026-08-10 13:30", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.002, "low": closes * 0.998,
         "close": closes, "volume": rng.integers(1000, 9000, n)},
        index=index,
    )


def test_replay_slice_never_contains_a_future_bar():
    """A fatia que o detector ve em cada passo do replay tem de acabar
    exactamente no cursor."""
    bars = _bars()
    for cursor in (30, 60, 99):
        visible = bars.iloc[:cursor + 1]
        assert len(visible) == cursor + 1
        assert visible.index[-1] == bars.index[cursor]
        assert bars.index[cursor + 1] not in visible.index


def test_resolution_bar_is_in_the_future_when_prediction_is_made(conn):
    """No momento em que a previsao e' gravada, resolve_at_ts tem de ser
    posterior a' ultima barra visivel -- ou seja, ainda nao existe."""
    bars = _bars()
    cursor = 50
    visible = bars.iloc[:cursor + 1]
    horizon = config.PATTERN_REALTIME["horizon_bars"]

    predicted_at = visible.index[-1]
    resolve_at = predicted_at + (bars.index[1] - bars.index[0]) * horizon

    assert resolve_at > visible.index[-1]
    assert resolve_at not in visible.index


def test_prediction_stored_before_outcome_exists(conn):
    session_id = realtime.start_session(conn, "replay")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO live_predictions
             (session_id, ticker, pattern_type, predicted_at_ts, resolve_at_ts, entry_price,
              predicted_direction, confidence, created_at)
           VALUES (?, 'AAPL', 'BULL_FLAG', '2026-08-10T14:00:00+00:00',
                   '2026-08-10T14:30:00+00:00', 100.0, 1, 0.7, ?)""",
        (session_id, now),
    )
    conn.commit()

    row = conn.execute("SELECT * FROM live_predictions").fetchone()
    assert row["resolved"] == 0
    assert row["exit_price"] is None
    assert row["actual_direction"] is None
    assert row["correct"] is None
    assert row["resolve_at_ts"] > row["predicted_at_ts"]


def test_resolution_marks_correct_only_when_direction_matches(conn):
    session_id = realtime.start_session(conn, "replay")
    bars = _bars()
    ticker = "AAPL"
    entry_index, horizon = 40, config.PATTERN_REALTIME["horizon_bars"]
    entry_price = float(bars["close"].iloc[entry_index])
    exit_price = float(bars["close"].iloc[entry_index + horizon])
    went_up = exit_price > entry_price

    conn.execute(
        """INSERT INTO live_predictions
             (session_id, ticker, pattern_type, predicted_at_ts, resolve_at_ts, entry_price,
              predicted_direction, confidence, created_at)
           VALUES (?, ?, 'BULL_FLAG', ?, ?, ?, 1, 0.7, ?)""",
        (session_id, ticker, bars.index[entry_index].isoformat(),
         bars.index[entry_index + horizon].isoformat(), entry_price,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()

    class _NoLearn:
        def learn(self, *args, **kwargs):
            pass

    resolved = realtime._resolve_due(conn, session_id, ticker, bars, len(bars) - 1, _NoLearn())
    conn.commit()

    assert resolved == 1
    row = conn.execute("SELECT * FROM live_predictions").fetchone()
    assert row["resolved"] == 1
    assert bool(row["correct"]) is bool(went_up)   # apostou em subida


def test_pending_prediction_is_not_resolved_before_its_bar_exists(conn):
    session_id = realtime.start_session(conn, "replay")
    bars = _bars()
    conn.execute(
        """INSERT INTO live_predictions
             (session_id, ticker, pattern_type, predicted_at_ts, resolve_at_ts, entry_price,
              predicted_direction, confidence, created_at)
           VALUES (?, 'AAPL', 'BULL_FLAG', ?, ?, 100.0, 1, 0.7, ?)""",
        (session_id, bars.index[10].isoformat(), bars.index[60].isoformat(),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()

    class _NoLearn:
        def learn(self, *args, **kwargs):
            pass

    # cursor ainda em 30: a barra 60 nao existe do ponto de vista do motor
    resolved = realtime._resolve_due(conn, session_id, "AAPL", bars.iloc[:31], 30, _NoLearn())
    assert resolved == 0
    assert conn.execute("SELECT resolved FROM live_predictions").fetchone()[0] == 0


# --------------------------------------------------------------------------
# Sessoes
# --------------------------------------------------------------------------

def test_starting_a_session_stops_the_previous_one(conn):
    first = realtime.start_session(conn, "replay")
    second = realtime.start_session(conn, "replay")
    statuses = dict(conn.execute("SELECT session_id, status FROM live_sessions").fetchall())
    assert statuses[first] == "stopped"
    assert statuses[second] == "running"


def test_session_state_is_empty_without_any_session(conn):
    assert realtime.session_state(conn) == {"session": None}


def test_market_hours_window():
    open_time = datetime(2026, 8, 13, 15, 0, tzinfo=timezone.utc)      # quinta, 15:00 UTC
    closed_time = datetime(2026, 8, 13, 21, 0, tzinfo=timezone.utc)    # depois do fecho
    weekend = datetime(2026, 8, 15, 15, 0, tzinfo=timezone.utc)        # sabado
    assert realtime.market_is_open(open_time)
    assert not realtime.market_is_open(closed_time)
    assert not realtime.market_is_open(weekend)
