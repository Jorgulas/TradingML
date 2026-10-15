"""Testes da simulacao de quatro horas (experiencia V).

O TESTE QUE CARREGA A SIMULACAO e' `test_only_one_position_at_a_time`.

A pergunta era "negociar ao minuto durante quatro horas", e a resposta depende
inteiramente de quantos negocios CABEM la' dentro. Uma conta real tem uma
posicao de cada vez: se um negocio demora quatro minutos a resolver, nao cabem
240 negocios em 240 minutos, cabem ~70. Uma simulacao que deixasse posicoes
sobrepor-se contaria custos a mais e lucros a mais ao mesmo tempo, e o numero
que sai no fim nao significaria nada.

O segundo e' `test_costs_scale_with_count_and_size`: a conclusao desta
experiencia e' que o custo e' quase tudo, e se o custo nao escalar exactamente
com o numero de negocios e com o tamanho, a conclusao nao se sustenta.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research import live_simulation as V
from src.research import pivot_bracket as T


def _walk_frame(bars=600, sub=60, step=0.9, seed=5, sessions=2):
    """Barras de 1 minuto a partir de uma caminhada por dentro da barra, com a
    volatilidade na ordem de grandeza do MNQ (amplitude mediana ~8 pontos)."""
    rng = np.random.default_rng(seed)
    rows = []
    for session in range(sessions):
        ticks = 30000.0 + np.cumsum(rng.normal(0, step, bars * sub))
        grid = ticks.reshape(bars, sub)
        rows.append(pd.DataFrame({
            "ts": pd.date_range(f"2026-03-{session + 2:02d} 14:30",
                                periods=bars, freq="1min", tz="UTC"),
            "open": grid[:, 0], "high": grid.max(1), "low": grid.min(1),
            "close": grid[:, -1], "volume": 1,
        }))
    return T.add_session(pd.concat(rows).reset_index(drop=True))


# --------------------------------------------------------------------------
# 1. Uma posicao de cada vez
# --------------------------------------------------------------------------

def test_only_one_position_at_a_time():
    """Num intervalo de 240 minutos nao podem caber 240 negocios se cada um
    demora mais de um minuto a resolver."""
    frame = _walk_frame()
    result = V.simulate(frame, 10, 250, mode="continuo",
                        rng=np.random.default_rng(1))
    assert result["n"] > 0
    assert result["n"] < 240, result["n"]


def test_a_slow_market_fits_fewer_trades_than_a_fast_one():
    """Controlo do anterior: se o aparelho nao notasse a duracao, os dois
    mercados dariam o mesmo numero de negocios."""
    quiet = V.simulate(_walk_frame(step=0.3, seed=2), 10, 250, "continuo",
                       rng=np.random.default_rng(1))
    busy = V.simulate(_walk_frame(step=2.5, seed=2), 10, 250, "continuo",
                      rng=np.random.default_rng(1))
    assert busy["n"] > quiet["n"]


def test_windows_do_not_overlap_or_cross_sessions():
    """Janelas que partilham barras partilham negocios, e o intervalo
    estreitaria por uma razao falsa -- o erro da ronda 7."""
    frame = _walk_frame(bars=600, sessions=3)
    spans = V.windows(frame, minutes=240)
    assert len(spans) == 6                       # 2 por sessao de 600 barras
    for (lo_a, hi_a), (lo_b, _) in zip(spans, spans[1:]):
        assert hi_a < lo_b
    sessions = frame["session"].to_numpy()
    for lo, hi in spans:
        assert sessions[lo] == sessions[hi]


# --------------------------------------------------------------------------
# 2. Os custos
# --------------------------------------------------------------------------

def test_costs_scale_with_count_and_size():
    frame = _walk_frame()
    one = V.simulate(frame, 10, 250, "continuo", contracts=1,
                     rng=np.random.default_rng(3))
    two = V.simulate(frame, 10, 250, "continuo", contracts=2,
                     rng=np.random.default_rng(3))
    assert one["n"] == two["n"]
    assert two["custos_usd"] == pytest.approx(2 * one["custos_usd"])
    assert two["liquido_usd"] == pytest.approx(2 * one["liquido_usd"])
    assert one["custos_usd"] == pytest.approx(
        one["n"] * V.COST_POINTS * V.POINT_VALUE)


def test_net_is_gross_minus_costs():
    frame = _walk_frame(seed=7)
    result = V.simulate(frame, 10, 250, "continuo", rng=np.random.default_rng(4))
    assert result["liquido_usd"] == pytest.approx(
        result["bruto_usd"] - result["custos_usd"])


def test_zero_cost_removes_the_whole_drag():
    frame = _walk_frame(seed=9)
    free = V.simulate(frame, 10, 250, "continuo", cost=0.0,
                      rng=np.random.default_rng(5))
    assert free["custos_usd"] == 0.0
    assert free["liquido_usd"] == pytest.approx(free["bruto_usd"])


# --------------------------------------------------------------------------
# 3. A aritmetica da alegacao
# --------------------------------------------------------------------------

def test_the_claim_arithmetic_matches_by_hand():
    """0.89*5 - 0.11*25 - 1.10 = +0.60 pontos = +1.20 USD por negocio."""
    claim = V.if_the_claim_were_true(0.89, n_trades=240, contracts=1)
    assert claim["por_negocio_pts"] == pytest.approx(0.60, abs=1e-9)
    assert claim["por_negocio_usd"] == pytest.approx(1.20, abs=1e-9)
    assert claim["dia_usd"] == pytest.approx(288.0, abs=1e-6)


def test_the_random_entry_rate_loses_exactly_the_cost():
    """A 83.33% a esperanca BRUTA e' zero por construcao, portanto a perda tem
    de ser exactamente o custo. E' a verificacao de que a conta fecha."""
    claim = V.if_the_claim_were_true(25 / 30, n_trades=240)
    assert claim["por_negocio_pts"] == pytest.approx(-V.COST_POINTS, abs=1e-9)
    assert claim["dia_usd"] == pytest.approx(
        -240 * V.COST_POINTS * V.POINT_VALUE, abs=1e-6)


# --------------------------------------------------------------------------
# 4. Ruina
# --------------------------------------------------------------------------

def test_trading_stops_when_the_margin_is_gone():
    frame = _walk_frame(seed=11)
    broke = V.simulate(frame, 10, 250, "continuo", balance=V.DAY_MARGIN * 0.5,
                       rng=np.random.default_rng(6))
    assert broke["n"] == 0


def test_compounding_a_losing_day_ruins_the_account():
    losing = pd.DataFrame({"retorno": [-0.05] * 20})
    year = V.compound(losing, sessions=252, draws=200)
    assert year["prob_ruina"] == pytest.approx(1.0)
    assert year["prob_lucro"] == 0.0


def test_compounding_a_winning_day_does_not():
    """Sem este, o teste anterior nao prova nada: um simulador que arruinasse
    sempre passaria os dois."""
    winning = pd.DataFrame({"retorno": [0.01] * 20})
    year = V.compound(winning, sessions=60, draws=200)
    assert year["prob_ruina"] == 0.0
    assert year["prob_lucro"] == pytest.approx(1.0)


def test_summary_counts_positive_windows():
    frame = pd.DataFrame({"n": [5, 5, 5, 5], "acerto": [0.8] * 4,
                          "bruto_usd": [1.0] * 4, "custos_usd": [1.0] * 4,
                          "liquido_usd": [10.0, -5.0, -5.0, -5.0],
                          "mediana_usd": [0.0] * 4})
    result = V.summarise(frame, "teste")
    assert result["janelas_positivas"] == pytest.approx(0.25)
    assert result["pior_usd"] == -5.0
    assert result["melhor_usd"] == 10.0
