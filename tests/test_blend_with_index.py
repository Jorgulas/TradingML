"""Testes da mistura sinal + indice (experiencia P).

Protegem a peca de que a conclusao depende: a CONDICAO DE SHARPE. Se ela
estiver mal escrita, a experiencia responde ao contrario -- e a condicao tem uma
subtileza que e' facil perder, que e' o sinal da correlacao decidir se um activo
de retorno negativo ajuda ou nao.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research import blend_with_index as B


def _series(values, start="2025-04-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values),
                                                 freq="5D", tz="UTC"))


def test_blending_with_cash_keeps_the_arithmetic_sharpe_exactly_constant():
    """E' a fasquia contra a qual o diversificador e' julgado. Se esta
    propriedade nao se verificar, a comparacao nao significa nada."""
    rng = np.random.default_rng(1)
    returns = rng.normal(0.005, 0.03, 300)
    baseline = None
    for weight in (0.0, 0.2, 0.5, 0.8):
        values = (1 - weight) * returns
        sharpe = values.mean() / values.std(ddof=1)
        if baseline is None:
            baseline = sharpe
        assert sharpe == pytest.approx(baseline, rel=1e-12)


def test_a_zero_correlation_asset_needs_a_positive_return_to_help():
    """Com correlacao nula a condicao reduz-se a 'o retorno tem de ser > 0'."""
    rng = np.random.default_rng(4)
    index = _series(rng.normal(0.005, 0.03, 400))
    strategy = _series(rng.normal(0.0, 0.015, 400))
    result = B.break_even_return(index, strategy)
    assert abs(result["correlation"]) < 0.15
    assert abs(result["required_cagr"]) < 0.02      # a fasquia fica perto de zero


def test_a_negatively_correlated_asset_may_help_while_losing_money():
    """A subtileza que da' sentido a' experiencia: com correlacao negativa, a
    fasquia desce ABAIXO de zero, e um activo que perde pouco ainda melhora a
    carteira. Se a condicao estivesse escrita sem o sinal da correlacao, esta
    hipotese nunca seria considerada."""
    rng = np.random.default_rng(9)
    base = rng.normal(0.005, 0.03, 400)
    index = _series(base)
    strategy = _series(-0.5 * base + rng.normal(0.0, 0.004, 400))
    result = B.break_even_return(index, strategy)
    assert result["correlation"] < -0.5
    assert result["required_cagr"] < 0            # a fasquia e' negativa


def test_the_bar_rises_with_correlation():
    """Quanto mais o activo acompanha o indice, mais tem de render para valer a
    pena. E' a monotonia que da' sentido a' formula."""
    rng = np.random.default_rng(12)
    base = rng.normal(0.005, 0.03, 500)
    index = _series(base)

    bars = []
    for loading in (0.0, 0.3, 0.8):
        strategy = _series(loading * base + rng.normal(0.0, 0.01, 500))
        bars.append(B.break_even_return(index, strategy)["required_cagr"])
    assert bars[0] < bars[1] < bars[2]


def test_blend_weights_move_the_portfolio_toward_the_strategy():
    index = _series([0.02] * 10)
    strategy = _series([-0.01] * 10)
    assert B.blend(index, strategy, 0.0)[0] == pytest.approx(0.02)
    assert B.blend(index, strategy, 1.0)[0] == pytest.approx(-0.01)
    assert B.blend(index, strategy, 0.5)[0] == pytest.approx(0.005)


def test_crisis_test_finds_a_hedge_when_there_is_one():
    """Um activo que sobe quando o indice cai tem de aparecer com retorno
    positivo nos piores periodos -- senao o teste nao deteta um hedge real."""
    rng = np.random.default_rng(15)
    base = rng.normal(0.0, 0.03, 300)
    index = _series(base)
    strategy = _series(-base)
    result = B.crisis_behaviour(index, strategy)
    assert result["strategy_in_bad"] > 0
    assert result["strategy_in_good"] < 0


def test_crisis_test_exposes_an_asset_that_falls_with_the_index():
    rng = np.random.default_rng(16)
    base = rng.normal(0.0, 0.03, 300)
    index = _series(base)
    strategy = _series(0.5 * base)
    result = B.crisis_behaviour(index, strategy)
    assert result["strategy_in_bad"] < 0


def test_annualise_reports_drawdown_from_the_peak():
    result = B.annualise(np.array([0.5, -0.5]))
    assert result["max_drawdown"] == pytest.approx(-0.5)


def test_weights_grid_starts_at_zero():
    """O peso 0 e' a referencia contra a qual tudo se compara; sem ele a tabela
    nao tem linha de base."""
    assert B.WEIGHTS[0] == 0.0
    assert all(0.0 <= w <= 1.0 for w in B.WEIGHTS)
