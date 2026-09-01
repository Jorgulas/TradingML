"""Testes do confronto contra comprar e ficar quieto (experiencia O).

O que protegem:

  1. EXCESSO NAO E' TOTAL. Foi o erro que este modulo cometeu e que teria dado a
     resposta errada a' pergunta mais importante do projecto: a carteira de peso
     igual apareceu a -4,2% quando o indice fazia +35,9%, e a diferenca era
     inteiramente uma subtraccao esquecida. O painel preve retorno em EXCESSO
     sobre o indice porque e' isso que os modelos fazem; uma comparacao de
     "quanto rendeu o meu dinheiro" precisa do total.

  2. AS METRICAS DE RISCO. Um drawdown mal calculado inverte a conclusao sobre
     o que e' "seguro", que e' metade da pergunta.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research import benchmark_comparison as O


def _stamps(n):
    return pd.date_range("2025-04-01", periods=n, freq="5D", tz="UTC")


# --------------------------------------------------------------------------
# 1. Excesso contra total
# --------------------------------------------------------------------------

def test_total_return_is_excess_plus_the_index():
    """A identidade que corrige o erro: excesso = accao - indice, logo
    accao = excesso + indice. Se isto se partir, a comparacao volta a medir
    'o universo menos o SPY' e a chamar-lhe 'comprar o universo'."""
    stamps = _stamps(4)
    index_returns = pd.Series([0.02, -0.01, 0.03, 0.00], index=stamps)
    frame = pd.DataFrame({
        "ts": list(stamps) * 2,
        # Duas accoes: uma bate o indice em 1pp, a outra perde 1pp.
        "forward_return": [0.01, 0.01, 0.01, 0.01] + [-0.01] * 4,
    })

    total = O.buy_and_hold(frame, stamps, index_returns)
    # Excesso medio = 0, portanto o total tem de ser exactamente o indice.
    assert np.allclose(total.values, index_returns.values)


def test_a_universe_that_beats_the_index_shows_up_above_it():
    stamps = _stamps(3)
    index_returns = pd.Series([0.01, 0.01, 0.01], index=stamps)
    frame = pd.DataFrame({"ts": list(stamps),
                          "forward_return": [0.005, 0.005, 0.005]})
    total = O.buy_and_hold(frame, stamps, index_returns)
    assert np.allclose(total.values, [0.015, 0.015, 0.015])


def test_missing_index_dates_are_dropped_not_treated_as_zero():
    """Uma data sem indice nao pode virar 'o indice rendeu 0%' -- isso
    inventaria um periodo de desempenho relativo perfeito."""
    stamps = _stamps(3)
    index_returns = pd.Series([0.01, np.nan, 0.01], index=stamps)
    frame = pd.DataFrame({"ts": list(stamps),
                          "forward_return": [0.005, 0.005, 0.005]})
    total = O.buy_and_hold(frame, stamps, index_returns)
    assert len(total) == 2


# --------------------------------------------------------------------------
# 2. As metricas
# --------------------------------------------------------------------------

def test_drawdown_is_measured_from_the_peak():
    """+50% seguido de -50% nao volta a zero: volta a -25%. O drawdown mede-se
    do maximo atingido, nao do inicio."""
    returns = pd.Series([0.5, -0.5])
    result = O.describe(returns, "teste")
    assert result["max_drawdown"] == pytest.approx(-0.5)
    assert result["total"] == pytest.approx(-0.25)


def test_a_series_that_only_rises_has_no_drawdown():
    returns = pd.Series([0.01] * 50)
    assert O.describe(returns, "teste")["max_drawdown"] == pytest.approx(0.0)


def test_total_return_compounds_and_does_not_add():
    returns = pd.Series([0.1] * 3)
    result = O.describe(returns, "teste")
    assert result["total"] == pytest.approx(1.1 ** 3 - 1)
    assert result["total"] != pytest.approx(0.3)


def test_annualisation_uses_the_trading_calendar():
    """252 retornos diarios de 0,1% sao um ano, nao 252 anos."""
    returns = pd.Series([0.001] * O.TRADING_DAYS)
    result = O.describe(returns, "teste")
    assert result["years"] == pytest.approx(1.0)
    assert result["cagr"] == pytest.approx(1.001 ** O.TRADING_DAYS - 1, rel=1e-6)


def test_volatility_is_annualised_by_the_square_root_of_time():
    rng = np.random.default_rng(5)
    daily = rng.normal(0, 0.01, 5000)
    result = O.describe(pd.Series(daily), "teste")
    assert result["vol"] == pytest.approx(0.01 * np.sqrt(O.TRADING_DAYS), rel=0.05)


def test_a_short_series_returns_no_statistics():
    """Uma serie de um ponto nao tem volatilidade nem drawdown; devolver
    numeros seria inventa-los."""
    assert "total" not in O.describe(pd.Series([0.01]), "teste")


# --------------------------------------------------------------------------
# 3. O ano a ano
# --------------------------------------------------------------------------

def test_yearly_aggregation_compounds_within_the_year():
    index = pd.date_range("2024-01-01", periods=4, freq="90D", tz="UTC")
    returns = pd.Series([0.1, 0.1, 0.1, 0.1], index=index)
    yearly = O.decade_by_decade(returns)
    assert yearly.loc[2024] == pytest.approx(1.1 ** 4 - 1)


def test_yearly_separates_the_years():
    index = pd.DatetimeIndex(["2024-06-01", "2025-06-01"], tz="UTC")
    returns = pd.Series([0.2, -0.1], index=index)
    yearly = O.decade_by_decade(returns)
    assert yearly.loc[2024] == pytest.approx(0.2)
    assert yearly.loc[2025] == pytest.approx(-0.1)


# --------------------------------------------------------------------------
# 4. Coerencia
# --------------------------------------------------------------------------

def test_rebalance_horizon_matches_the_signal_horizon():
    """A simulacao rebalanceia ao ritmo que o sinal preve. Se alguem mudar um
    sem o outro, a carteira passa a negociar um horizonte que o modelo nao
    preve, e o resultado deixa de testar o sinal."""
    import config

    assert O.REBALANCE_DAYS == config.DECISION["horizon_days"]
