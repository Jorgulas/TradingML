"""Testes do condicionamento por regime (experiencia Q).

Protegem tres coisas, por ordem de gravidade:

  1. OS REGIMES EX-ANTE NAO PODEM OLHAR PARA A FRENTE. E' a diferenca entre uma
     regra que se pode seguir e uma que so' se sabe depois. Se a bandeira de
     regime usar uma barra futura, a experiencia deixa de responder a "ligo-o
     quando?" e passa a responder a nada.

  2. O PLACEBO POR ROTACAO tem de preservar a proporcao do regime. Se a rotacao
     mudasse quantos periodos ficam dentro, comparar-se-iam amostras de tamanhos
     diferentes e o placebo mediria isso em vez da ligacao ao sinal.

  3. O CALCULO DE ANOS NECESSARIOS e' o numero que separa "o efeito nao existe"
     de "a amostra e' cega". Este projecto ja' confundiu as duas coisas.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research import regime_conditional as Q


def _prices(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values),
                                                 freq="B", tz="UTC"))


# --------------------------------------------------------------------------
# 1. Os regimes so' olham para tras
# --------------------------------------------------------------------------

def test_regime_flags_ignore_the_future():
    """O teste decisivo: mudar radicalmente os precos DEPOIS de um instante nao
    pode alterar a bandeira de regime desse instante."""
    rng = np.random.default_rng(2)
    base = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 400))
    prices = _prices(base)
    stamps = prices.index[::5]

    first = Q.ex_ante_regimes(prices, stamps)

    altered = prices.copy()
    altered.iloc[300:] = altered.iloc[300:] * 5      # colapso/explosao no fim
    second = Q.ex_ante_regimes(altered, stamps)

    early = stamps[stamps < prices.index[290]]
    assert (first.loc[early] == second.loc[early]).all().all()


def test_below_the_moving_average_is_detected():
    prices = _prices(list(np.linspace(200, 100, 300)))   # a cair sempre
    stamps = prices.index[250:]
    flags = Q.ex_ante_regimes(prices, stamps)
    assert flags["abaixo da MA200"].all()
    assert flags["abaixo da MA50"].all()


def test_a_rising_market_is_not_flagged():
    prices = _prices(list(np.linspace(100, 200, 300)))
    stamps = prices.index[250:]
    flags = Q.ex_ante_regimes(prices, stamps)
    assert not flags["abaixo da MA200"].any()
    assert not flags["em drawdown > 5%"].any()


def test_drawdown_is_measured_against_the_running_peak():
    """Um mercado que sobe para 120 e volta a 110 esta' em drawdown de 8%, mesmo
    estando acima de onde comecou."""
    prices = _prices([100.0] * 10 + [120.0] * 10 + [110.0] * 10)
    stamps = prices.index[[5, 15, 25]]
    flags = Q.ex_ante_regimes(prices, stamps)
    assert not flags["em drawdown > 5%"].iloc[0]
    assert not flags["em drawdown > 5%"].iloc[1]
    assert flags["em drawdown > 5%"].iloc[2]


# --------------------------------------------------------------------------
# 2. O placebo
# --------------------------------------------------------------------------

def test_rotation_preserves_how_many_periods_are_inside():
    """np.roll nao muda a contagem. Se mudasse, o placebo comparava amostras de
    tamanhos diferentes e media isso em vez da ligacao ao sinal."""
    labels = np.array([True] * 30 + [False] * 70)
    for shift in (1, 17, 55, 99):
        assert np.roll(labels, shift).sum() == labels.sum()


def test_placebo_finds_a_real_split():
    """Um sinal que rende mesmo dentro do regime tem de dar p pequeno."""
    n = 200
    stamps = pd.date_range("2024-01-01", periods=n, freq="5D", tz="UTC")
    flag = pd.Series([True] * 60 + [False] * (n - 60), index=stamps)
    rng = np.random.default_rng(5)
    values = np.where(flag.values, 0.03, -0.01) + rng.normal(0, 0.002, n)
    result = Q.circular_placebo(pd.Series(values, index=stamps), flag, n=500)
    assert result["p_value"] < 0.05


def test_placebo_is_not_fooled_by_noise():
    """E um sinal sem ligacao nenhuma ao regime nao pode dar p pequeno."""
    n = 200
    stamps = pd.date_range("2024-01-01", periods=n, freq="5D", tz="UTC")
    flag = pd.Series([True] * 60 + [False] * (n - 60), index=stamps)
    rng = np.random.default_rng(6)
    values = pd.Series(rng.normal(0, 0.02, n), index=stamps)
    assert Q.circular_placebo(values, flag, n=500)["p_value"] > 0.10


# --------------------------------------------------------------------------
# 3. Poder estatistico
# --------------------------------------------------------------------------

def test_years_needed_grows_with_the_gap_to_the_detectable_effect():
    """Um efeito dez vezes menor do que o minimo detectavel precisa de cem vezes
    mais observacoes -- o erro-padrao cai com a raiz."""
    near = Q.years_needed(mean=0.01, mde=0.02, n_inside=50, frequency=0.25)
    far = Q.years_needed(mean=0.001, mde=0.02, n_inside=50, frequency=0.25)
    assert far == pytest.approx(near * 100, rel=1e-6)


def test_a_rarer_regime_needs_more_history():
    common = Q.years_needed(mean=0.01, mde=0.02, n_inside=50, frequency=0.50)
    rare = Q.years_needed(mean=0.01, mde=0.02, n_inside=50, frequency=0.05)
    assert rare == pytest.approx(common * 10, rel=1e-6)


def test_years_needed_is_undefined_without_an_effect():
    assert np.isnan(Q.years_needed(mean=0.0, mde=0.02, n_inside=50, frequency=0.25))


# --------------------------------------------------------------------------
# 4. A comparacao e o beta
# --------------------------------------------------------------------------

def test_compare_refuses_a_thin_regime():
    """Tres periodos dentro do regime nao dao uma media com significado."""
    stamps = pd.date_range("2024-01-01", periods=50, freq="5D", tz="UTC")
    flag = pd.Series([True] * 3 + [False] * 47, index=stamps)
    values = pd.Series(np.zeros(50), index=stamps)
    assert not Q.compare(values, flag)["usable"]


def test_beta_detects_a_book_that_is_not_neutral():
    """Se o livro carregasse beta, o diagnostico tinha de o dizer -- foi a
    hipotese que esta experiencia levantou e teve de testar."""
    rng = np.random.default_rng(7)
    stamps = pd.date_range("2024-01-01", periods=300, freq="5D", tz="UTC")
    index = pd.Series(rng.normal(0.004, 0.02, 300), index=stamps)
    loaded = pd.Series(0.6 * index.values + rng.normal(0, 0.002, 300), index=stamps)
    result = Q.beta_diagnostic(loaded, index)
    assert result["beta"] == pytest.approx(0.6, abs=0.1)
    assert result["beta_p"] < 0.01


def test_beta_confirms_a_neutral_book():
    rng = np.random.default_rng(8)
    stamps = pd.date_range("2024-01-01", periods=300, freq="5D", tz="UTC")
    index = pd.Series(rng.normal(0.004, 0.02, 300), index=stamps)
    neutral = pd.Series(rng.normal(0, 0.01, 300), index=stamps)
    result = Q.beta_diagnostic(neutral, index)
    assert abs(result["beta"]) < 0.15
    assert result["r_squared"] < 0.05
