"""Testes da previsao de direccao pos-padrao.

O que se protege aqui e' sobretudo o desenho anti-leakage. Tres armadilhas
concretas, cada uma capaz de produzir um backtest bonito e sem significado:
o preco de entrada ser o do fim do padrao em vez do da sua confirmacao; a
janela futura de treino invadir a particao de teste; e o erro-padrao ser
calculado sobre padroes em vez de dias.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.patterns import direction


def _seed_bars(conn, ticker, timeframe, closes, start="2026-01-05 14:30"):
    index = pd.date_range(start, periods=len(closes), freq="h", tz="UTC")
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """INSERT INTO intraday_prices (ticker, timeframe, ts, open, high, low, close, volume, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(ticker, timeframe, ts.isoformat(), c, c * 1.001, c * 0.999, float(c), 10_000, now)
         for ts, c in zip(index, closes)],
    )
    conn.commit()
    return index


def _seed_pattern(conn, ticker, timeframe, pattern_type, start_ts, end_ts, confirmed_ts):
    conn.execute(
        """INSERT INTO detected_patterns (ticker, timeframe, pattern_type, start_ts, end_ts,
             confirmed_ts, start_idx, end_idx, quality, meta_json, detected_at)
           VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0.8, '{}', ?)""",
        (ticker, timeframe, pattern_type, start_ts.isoformat(), end_ts.isoformat(),
         confirmed_ts.isoformat(), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


# --------------------------------------------------------------------------
# O rotulo tem de partir da CONFIRMACAO, nao do fim do padrao
# --------------------------------------------------------------------------

def test_reference_price_is_taken_at_confirmation_not_at_pattern_end(conn):
    """A armadilha central: o padrao acaba em end_ts, mas so' se SABE que
    existe quando o ultimo pivot fica confirmado, algumas barras depois.
    Entrar ao preco do fim do padrao e' negociar com informacao que ainda nao
    existia."""
    closes = [100.0] * 10 + [200.0] * 10 + [400.0] * 10  # tres patamares distintos
    index = _seed_bars(conn, "AAPL", "1h", closes)
    # padrao acaba no indice 5 (preco 100) mas so' confirma no indice 12 (preco 200)
    _seed_pattern(conn, "AAPL", "1h", "BULL_FLAG", index[0], index[5], index[12])

    direction.build_outcomes(conn, "1h", horizons=[10], tickers=["AAPL"])
    row = conn.execute(
        "SELECT ref_close, target_close, forward_return FROM pattern_outcomes"
    ).fetchone()

    # 200 (indice 12) e nao 100 (indice 5): entrou na confirmacao, nao no fim
    assert row["ref_close"] == pytest.approx(200.0), "entrou ao preco do fim do padrao (lookahead)"
    # 400 (indice 22): o horizonte tambem conta a partir da confirmacao
    assert row["target_close"] == pytest.approx(400.0)
    assert row["forward_return"] == pytest.approx(1.0)


def test_direction_label_matches_the_sign_of_the_forward_return(conn):
    index = _seed_bars(conn, "AAPL", "1h", list(np.linspace(100, 130, 40)))
    _seed_pattern(conn, "AAPL", "1h", "BULL_FLAG", index[0], index[5], index[10])
    direction.build_outcomes(conn, "1h", horizons=[5], tickers=["AAPL"])
    row = conn.execute("SELECT forward_return, direction FROM pattern_outcomes").fetchone()
    assert row["forward_return"] > 0 and row["direction"] == 1

    conn.execute("DELETE FROM pattern_outcomes")
    conn.execute("DELETE FROM intraday_prices")
    conn.execute("DELETE FROM detected_patterns")
    index = _seed_bars(conn, "AAPL", "1h", list(np.linspace(130, 100, 40)))
    _seed_pattern(conn, "AAPL", "1h", "BEAR_FLAG", index[0], index[5], index[10])
    direction.build_outcomes(conn, "1h", horizons=[5], tickers=["AAPL"])
    row = conn.execute("SELECT forward_return, direction FROM pattern_outcomes").fetchone()
    assert row["forward_return"] < 0 and row["direction"] == 0


def test_outcome_is_skipped_when_the_forward_window_does_not_exist_yet(conn):
    """Um padrao confirmado ha' 2 barras nao pode ter rotulo a 20 barras."""
    index = _seed_bars(conn, "AAPL", "1h", list(np.linspace(100, 110, 20)))
    _seed_pattern(conn, "AAPL", "1h", "PENNANT", index[0], index[10], index[18])
    direction.build_outcomes(conn, "1h", horizons=[20], tickers=["AAPL"])
    assert conn.execute("SELECT COUNT(*) FROM pattern_outcomes").fetchone()[0] == 0


def test_excess_return_subtracts_the_benchmark_over_the_same_window(conn):
    closes = list(np.linspace(100, 120, 40))          # accao +20%
    index = _seed_bars(conn, "AAPL", "1h", closes)
    _seed_bars(conn, "SPY", "1h", list(np.linspace(100, 120, 40)))  # indice igual
    _seed_pattern(conn, "AAPL", "1h", "BULL_FLAG", index[0], index[5], index[10])

    direction.build_outcomes(conn, "1h", horizons=[5], tickers=["AAPL"])
    row = conn.execute("SELECT forward_return, benchmark_return, excess_return FROM pattern_outcomes").fetchone()

    assert row["forward_return"] > 0            # subiu em bruto
    assert row["excess_return"] == pytest.approx(0.0, abs=1e-9)  # mas nada face ao mercado


# --------------------------------------------------------------------------
# Embargo: a janela futura do treino nao pode invadir a validacao/teste
# --------------------------------------------------------------------------

def _synthetic_frame(n=100, horizon=10):
    return pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "confirmed_ts": [f"2026-03-{1 + i // 24:02d}T{i % 24:02d}:00:00+00:00" for i in range(n)],
        "confirm_idx": list(range(n)),
        "label": [i % 2 for i in range(n)],
        "forward_return": [0.01] * n,
        "day": pd.to_datetime([f"2026-03-{1 + i // 24:02d}" for i in range(n)]),
    })


def test_embargo_drops_observations_whose_future_crosses_the_boundary():
    horizon = 10
    train, validation, test = direction.split_with_embargo(_synthetic_frame(100, horizon), horizon)

    validation_start = validation["confirm_idx"].min()
    test_start = test["confirm_idx"].min()
    assert (train["confirm_idx"] + horizon < validation_start).all()
    assert (validation["confirm_idx"] + horizon < test_start).all()


def test_embargo_actually_removes_rows():
    """Se o embargo nao removesse nada, nao estaria a fazer nada."""
    frame = _synthetic_frame(100, 10)
    train, validation, test = direction.split_with_embargo(frame, horizon=10)
    assert len(train) + len(validation) + len(test) < len(frame)


def test_larger_horizon_embargoes_more():
    frame = _synthetic_frame(200, 0)
    small = sum(len(p) for p in direction.split_with_embargo(frame, horizon=5))
    large = sum(len(p) for p in direction.split_with_embargo(frame, horizon=40))
    assert large < small


def test_partitions_never_overlap_in_time():
    train, validation, test = direction.split_with_embargo(_synthetic_frame(120, 10), horizon=10)
    assert train["confirm_idx"].max() < validation["confirm_idx"].min()
    assert validation["confirm_idx"].max() < test["confirm_idx"].min()


# --------------------------------------------------------------------------
# Configuracao
# --------------------------------------------------------------------------

def test_only_hourly_is_used_for_statistics():
    """A 5m ha' so' 61 dias distintos de historico; com os 43 tickers a
    moverem-se juntos, a amostra efectiva e' 61 e nada e' mensuravel."""
    assert config.PATTERN_DIRECTION["timeframes"] == ["1h"]


def test_benchmark_is_not_scanned_for_patterns():
    """O SPY existe para servir de denominador, nao para gerar padroes."""
    assert config.BENCHMARK["ticker"] not in config.PATTERN_TICKERS
    assert config.BENCHMARK["ticker"] in [i["ticker"] for i in config.ALL_INSTRUMENTS]
