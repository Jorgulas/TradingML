"""Testes da decomposicao por noticias agendadas (experiencia S).

O teste que importa mais e' o do SLOPE_DIFFERENCE, porque corrige um erro
estatistico que esta experiencia chegou a cometer: comparar os p-values de duas
regressoes e concluir que sao diferentes. Um declive pode cair 30% e perder
significancia apenas por o intervalo ser largo. A diferenca entre "significativo"
e "nao significativo" nao e' ela propria significativa, e uma experiencia que
conclua assim responde ao contrario.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research import scheduled_news_share as S


def _days(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


# --------------------------------------------------------------------------
# 1. A janela de resultados
# --------------------------------------------------------------------------

def test_the_window_covers_the_day_before_and_after():
    """A empresa pode anunciar depois do fecho, e nesse caso a reaccao cai no dia
    seguinte. Uma janela de tres dias apanha as duas convencoes sem ter de
    adivinhar qual delas a fonte usou."""
    stamps = _days(10)
    dates = np.array(["2020-01-05"], dtype="datetime64[D]")
    mask = S.earnings_mask(stamps, dates)
    assert mask.sum() == 3
    assert list(np.where(mask)[0]) == [3, 4, 5]      # dias 4, 5 e 6 de Janeiro


def test_no_earnings_means_no_flags():
    assert not S.earnings_mask(_days(10), np.array([], dtype="datetime64[D]")).any()


def test_windows_that_overlap_are_not_double_counted():
    """Duas datas a dois dias de distancia partilham um dia. A mascara e' de
    booleanos, portanto a uniao nao pode contar esse dia duas vezes."""
    stamps = _days(10)
    dates = np.array(["2020-01-05", "2020-01-07"], dtype="datetime64[D]")
    mask = S.earnings_mask(stamps, dates)
    assert mask.sum() == 5                            # 4,5,6 uniao 6,7,8


def test_dates_outside_the_series_are_ignored():
    stamps = _days(10)
    dates = np.array(["2019-06-01", "2021-06-01"], dtype="datetime64[D]")
    assert not S.earnings_mask(stamps, dates).any()


# --------------------------------------------------------------------------
# 2. O controlo de composicao
# --------------------------------------------------------------------------

def test_only_tickers_present_in_every_year_survive():
    """Sem isto, uma subida podia ser apenas quem entrou. E aqui e' ainda mais
    critico do que na ronda 15, porque a cobertura do calendario de resultados
    cresce com o tempo -- as accoes que so' aparecem tarde entrariam todas na
    parte recente da serie."""
    rows = []
    for year in range(2005, 2010):
        for ticker in ("SEMPRE_A", "SEMPRE_B"):
            rows.append({"ticker": ticker, "period": year, "concentration": 0.2})
        if year >= 2008:
            rows.append({"ticker": "TARDIO", "period": year, "concentration": 0.9})
    stable = S.constant_panel(pd.DataFrame(rows))
    assert set(stable["ticker"]) == {"SEMPRE_A", "SEMPRE_B"}


# --------------------------------------------------------------------------
# 3. A diferenca de declives -- o teste que corrige o erro
# --------------------------------------------------------------------------

def _two_series(slope_a, slope_b, noise_a=0.0, noise_b=0.0, seed=3, years=21,
                n_tickers=30):
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(years):
        year = 2005 + index
        for ticker in range(n_tickers):
            rows.append({
                "ticker": f"T{ticker}", "period": year,
                "a": 0.2 + slope_a * index + rng.normal(0, noise_a),
                "b": 0.2 + slope_b * index + rng.normal(0, noise_b),
            })
    return pd.DataFrame(rows)


def test_identical_slopes_are_not_distinguishable():
    """Duas series que sobem ao mesmo ritmo nao podem ser declaradas diferentes,
    por muito ruido que tenham."""
    frame = _two_series(0.002, 0.002, noise_a=0.05, noise_b=0.05)
    result = S.slope_difference(frame, "a", "b")
    assert not result["distinguishable"]
    assert result["ci"][0] < 0 < result["ci"][1]


def test_a_real_slope_gap_is_detected():
    frame = _two_series(0.004, 0.000, noise_a=0.01, noise_b=0.01)
    result = S.slope_difference(frame, "a", "b")
    assert result["distinguishable"]
    assert result["difference"] == pytest.approx(0.004, abs=0.001)


def test_the_p_value_fallacy_is_actually_avoided():
    """O caso que motivou a correccao, construido de proposito.

    A serie `a` sobe com pouco ruido e da' p pequeno; a `b` sobe ao MESMO ritmo
    mas com muito ruido e da' p grande. Comparar os dois p-values diria
    "diferentes"; o teste sobre a diferenca tem de dizer que nao."""
    from scipy import stats as scipy_stats

    # O ruido tem de ser grande DEPOIS de fazer a media dos tickers: com 30
    # accoes por ano, um desvio de 0.20 vira 0.037, e a tendencia continua a
    # sair significativa. Poucos tickers e muito ruido e' o que constroi mesmo
    # o caso em que a falacia estaria disponivel.
    frame = _two_series(0.003, 0.003, noise_a=0.005, noise_b=1.2, seed=11,
                        n_tickers=6)
    yearly_a = frame.groupby("period")["a"].mean()
    yearly_b = frame.groupby("period")["b"].mean()
    p_a = scipy_stats.linregress(yearly_a.index, yearly_a.values).pvalue
    p_b = scipy_stats.linregress(yearly_b.index, yearly_b.values).pvalue

    assert p_a < 0.01 and p_b > 0.05         # a falacia estaria disponivel
    assert not S.slope_difference(frame, "a", "b")["distinguishable"]


def test_slope_difference_needs_enough_years():
    frame = _two_series(0.002, 0.001, years=5)
    assert S.slope_difference(frame, "a", "b") == {}


# --------------------------------------------------------------------------
# 4. A tendencia
# --------------------------------------------------------------------------

def test_trend_recovers_a_known_slope():
    frame = _two_series(0.0025, 0.0)
    result = S.trend(frame, "a")
    assert result["slope"] == pytest.approx(0.0025, rel=1e-6)
    assert result["n_years"] == 21


def test_trend_needs_a_minimum_of_years():
    frame = _two_series(0.002, 0.001, years=6)
    assert S.trend(frame, "a") == {}


def test_intensity_is_one_when_earnings_days_are_ordinary():
    """A intensidade e' a quota de variancia a dividir pela quota de dias. Se um
    dia de resultados for igual aos outros, tem de dar exactamente 1 -- e' a
    calibracao que torna o numero legivel."""
    # A amostra tem de ser grande. O quadrado de uma normal e' qui-quadrado com
    # 1 grau de liberdade, cuja variancia e' o dobro da media, e com 252 dias
    # marcados o erro-padrao do racio e' 0.089 -- uma tolerancia de 0.15 seria
    # 1.7 erros-padrao e falharia por acaso com frequencia. Com 25200 dias o
    # erro-padrao cai para 0.028.
    rng = np.random.default_rng(7)
    values = rng.normal(0, 0.02, 25_200)
    flags = np.zeros(len(values), dtype=bool)
    flags[::10] = True                                # 10% dos dias
    squares = values ** 2
    intensity = (squares[flags].sum() / squares.sum()) / flags.mean()
    assert intensity == pytest.approx(1.0, abs=0.10)
