"""Testes da experiencia de sobrevivencia e das defesas que ela obrigou a criar.

O que protegem:

  1. O GUARDA DE DADOS IMPOSSIVEIS. Foi o que faltou e custou uma conclusao
     errada: a BYND entrou no painel com um desdobramento inverso aplicado de
     forma inconsistente (0,55 -> 17,85 -> 0,56 dolares em dias seguidos) e
     esse unico ticker produziu 99% de um colapso que quase levou a concluir
     que o modelo nao generalizava. O guarda tem de apanhar essa assinatura e
     tem de deixar passar movimentos violentos mas possiveis.

  2. A DECOMPOSICAO POR PERNA. Se a atribuicao de retorno as pernas comprada e
     vendida estiver trocada, a conclusao sobre o vies inverte-se.

  3. A ESTRATIFICACAO nao pode usar o drawdown para decidir, so' para agrupar.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.decision import panel as P
from src.research import survivorship as K


def _bars(closes, start="2025-01-02"):
    index = pd.date_range(start, periods=len(closes), freq="B", tz="UTC")
    closes = np.asarray(closes, dtype=float)
    # Volume TEM de variar: `volume_z20` divide pelo desvio-padrao movel, e com
    # volume constante fica NaN em todas as linhas, o que esvazia o painel
    # inteiro sem erro nenhum visivel.
    volume = 10_000 + (np.arange(len(closes)) * 137) % 5_000
    return pd.DataFrame({"open": closes, "high": closes * 1.01, "low": closes * 0.99,
                         "close": closes, "volume": volume}, index=index)


# --------------------------------------------------------------------------
# 1. O guarda de dados impossiveis
# --------------------------------------------------------------------------

def test_reverse_split_artefact_is_caught():
    """A assinatura exacta da BYND: ida e volta entre dois niveis de preco
    separados por um factor de trinta."""
    prices = [0.59, 17.85, 18.30, 17.91, 0.56, 0.55, 0.54, 16.98, 0.61]
    assert K_is_corrupt(prices)


def test_a_violent_but_possible_move_survives():
    """Uma biotecnologica pode cair 60% num dia com um ensaio falhado. Isso e'
    mercado, nao erro, e excluir esses casos seria censurar a cauda que este
    projecto passou seis rondas a estudar."""
    prices = [100.0, 98.0, 40.0, 42.0, 41.0, 43.0]
    assert not K_is_corrupt(prices)


def test_a_normal_series_survives():
    rng = np.random.default_rng(4)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.02, 300))
    assert not K_is_corrupt(prices)


def _seed_daily(conn, ticker, closes):
    from datetime import datetime, timezone

    conn.execute("INSERT OR IGNORE INTO watchlist (ticker, name, sector, active, "
                 "added_date) VALUES (?, ?, 'Test', 0, '2025-01-01')", (ticker, ticker))
    bars = _bars(closes)
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO intraday_prices (ticker, timeframe, ts, open, high, "
        "low, close, volume, ingested_at) VALUES (?, '1d', ?, ?, ?, ?, ?, ?, ?)",
        [(ticker, index.isoformat(), r["open"], r["high"], r["low"], r["close"],
          r["volume"], now) for index, r in bars.iterrows()])
    conn.commit()


def test_corrupt_ticker_is_reported_not_silently_dropped(conn):
    """Perder um ticker em silencio e' pior do que o incluir: ninguem repara que
    o universo mudou entre duas execucoes, e a comparacao deixa de ser sobre o
    que se pensa que e'."""
    rng = np.random.default_rng(1)
    clean = 100 * np.cumprod(1 + rng.normal(0, 0.015, 320))
    _seed_daily(conn, "GOOD", clean)
    _seed_daily(conn, "BADD", np.tile([1.0, 30.0], 160))
    _seed_daily(conn, "SPY", clean * 0.9)

    excluded = []
    frame = P.build(conn, tickers=["GOOD", "BADD"], min_days=100,
                    timeframe="1d", excluded=excluded)

    assert excluded == ["BADD"]
    assert set(frame["ticker"].unique()) == {"GOOD"}


def K_is_corrupt(prices):
    return P.is_corrupt(_bars(prices))


# --------------------------------------------------------------------------
# 2. Decomposicao por perna
# --------------------------------------------------------------------------

def _one_day(returns, confidences):
    return pd.DataFrame({
        "ts": [pd.Timestamp("2025-01-02", tz="UTC")] * len(returns),
        "ticker": [f"T{i}" for i in range(len(returns))],
        "forward_return": returns,
    }), np.asarray(confidences, dtype=float)


def test_legs_have_the_right_sign():
    """A perna vendida e' MENOS a media do decil de baixo. Se o sinal estiver
    trocado, um modelo mau parece bom."""
    n = 40
    returns = list(np.linspace(-0.10, 0.10, n))
    frame, confidence = _one_day(returns, np.arange(n))   # confianca = ordem real
    result = K.legs(frame, confidence, fraction=0.1)
    assert result["comprada"].iloc[0] > 0     # comprou os que subiram
    assert result["vendida"].iloc[0] > 0      # vendeu os que cairam


def test_legs_sum_to_the_long_short_return():
    from src.decision.short_term import long_short_return

    n = 40
    rng = np.random.default_rng(9)
    returns = list(rng.normal(0, 0.05, n))
    frame, confidence = _one_day(returns, rng.normal(0, 1, n))
    result = K.legs(frame, confidence, fraction=0.1)
    total = result["comprada"].iloc[0] + result["vendida"].iloc[0]
    assert total == pytest.approx(long_short_return(frame, confidence).iloc[0])


def test_a_model_that_shorts_crashes_scores_below_half():
    """O teste 2 depende deste sentido: posto medio abaixo de 0,5 quer dizer
    que o modelo poe as quedas do lado vendido."""
    n = 60
    returns = np.linspace(-0.3, 0.3, n)
    frame, _ = _one_day(list(returns), np.zeros(n))
    result = K.crash_response(frame, returns, quantile=0.1)   # confianca = retorno
    assert result["crash_mean_rank"] < 0.2
    assert result["spike_mean_rank"] > 0.8


# --------------------------------------------------------------------------
# 3. Estratificacao
# --------------------------------------------------------------------------

def test_strata_split_the_universe_in_three():
    rng = np.random.default_rng(2)
    days = pd.date_range("2025-01-02", periods=60, freq="B", tz="UTC")
    rows = []
    for i in range(30):
        drift = -0.01 if i < 10 else (0.0 if i < 20 else 0.01)
        for day in days:
            rows.append({"ticker": f"T{i:02d}", "ts": day,
                         "ret_1": float(rng.normal(drift, 0.02)),
                         "forward_return": float(rng.normal(0, 0.02))})
    frame = pd.DataFrame(rows)
    table = K.fragility_strata(frame, rng.normal(0, 1, len(frame)))
    assert len(table) == 3
    assert table["n_tickers"].sum() == 30
    # O grupo mais fragil tem de ter o drawdown mais profundo.
    ordered = table.set_index("grupo")["drawdown_medio"]
    assert ordered["mais fragil"] < ordered["mais solido"]


# --------------------------------------------------------------------------
# 4. As listas de universo
# --------------------------------------------------------------------------

def test_universe_groups_do_not_overlap():
    """Uma accao classificada nos dois grupos apareceria duas vezes no painel e
    contaria a dobrar nos percentis do dia."""
    assert not set(K.INDEX_REMOVED) & set(K.NEVER_LARGE_CAP)


def test_no_fallen_ticker_is_already_in_the_universe():
    """Repor uma accao que ja' la' esta' nao testa nada e duplica-a."""
    import config
    from src.research.fallen_universe import FALLEN

    assert not set(FALLEN) & set(config.TREE_TICKERS)


def test_delisted_list_is_documented_as_unrecoverable():
    """O buraco que fica por tapar tem de continuar registado: e' a ressalva
    principal da experiencia e nao pode desaparecer numa limpeza de codigo."""
    from src.research.fallen_universe import DELISTED_NO_DATA

    assert len(DELISTED_NO_DATA) >= 15
    assert "SIVB" in DELISTED_NO_DATA      # falencia bancaria de 2023
    assert "ATVI" in DELISTED_NO_DATA      # aquisicao pela Microsoft
