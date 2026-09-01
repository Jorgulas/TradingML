"""Testes da forma da cauda (experiencia R).

Dois testes carregam a experiencia inteira, e os outros sao acessorios:

  1. AS MEDIDAS TEM DE SER SEM ESCALA. Multiplicar todos os retornos por dez nao
     pode mexer nelas. Se mexer, sobem com a volatilidade e o estudo colapsa em
     persistencia de volatilidade -- que se conhece desde os anos 80 e nao seria
     novidade nenhuma.

  2. TEM DE DISTINGUIR CAUDA GORDA DE CAUDA FINA com a MESMA volatilidade. E' a
     propriedade toda: se duas series com igual desvio-padrao e caudas
     diferentes derem a mesma leitura, a medida nao mede nada.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research import tail_shape as R


def _panel(rows):
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. Sem escala
# --------------------------------------------------------------------------

def test_measures_are_invariant_to_scale():
    """O teste que sustenta a experiencia. Dez vezes maiores, mesma forma."""
    rng = np.random.default_rng(3)
    returns = rng.standard_t(4, 3000) * 0.01

    small = R.tail_measures(returns)
    large = R.tail_measures(returns * 10)

    for measure in ("tail_ratio", "concentration", "hill"):
        assert small[measure] == pytest.approx(large[measure], rel=1e-9), measure
    # E a volatilidade TEM de mudar -- senao o teste acima seria vacuo.
    assert large["volatility"] == pytest.approx(10 * small["volatility"], rel=1e-9)


def test_a_gaussian_sample_lands_near_the_gaussian_reference():
    """q99/q75 de |N(0,1)| e' Phi^-1(0.995)/Phi^-1(0.875) = 2.576/1.150 = 2.24.

    Este teste apanhou um erro a serio: a constante estava em 2.02, calculada
    com o percentil 99 UNILATERAL. A medida e' sobre o valor absoluto, onde o
    percentil 99 e' 2.576 e nao 2.326. Com a referencia errada, uma cauda
    observada de 2.97 parecia 47% acima da normal em vez de 33%."""
    rng = np.random.default_rng(11)
    result = R.tail_measures(rng.normal(0, 0.02, 200_000))
    assert result["tail_ratio"] == pytest.approx(R.GAUSSIAN_TAIL_RATIO, rel=0.03)


# --------------------------------------------------------------------------
# 2. Distingue forma com volatilidade igual
# --------------------------------------------------------------------------

def test_fat_tails_are_distinguished_from_thin_at_equal_volatility():
    """A propriedade toda da experiencia. Duas series com o MESMO desvio-padrao
    e caudas diferentes tem de dar leituras diferentes."""
    rng = np.random.default_rng(5)
    thin = rng.normal(0, 1, 50_000)
    fat = rng.standard_t(3, 50_000)
    # Iguala-se a volatilidade, para a unica diferenca ser a forma.
    fat = fat * (thin.std(ddof=1) / fat.std(ddof=1))
    assert fat.std(ddof=1) == pytest.approx(thin.std(ddof=1), rel=1e-9)

    thin_measures = R.tail_measures(thin)
    fat_measures = R.tail_measures(fat)

    assert fat_measures["tail_ratio"] > thin_measures["tail_ratio"] * 1.3
    assert fat_measures["concentration"] > thin_measures["concentration"] * 1.3
    assert fat_measures["hill"] < thin_measures["hill"]     # menor = mais gorda


def test_concentration_is_a_share_between_zero_and_one():
    rng = np.random.default_rng(9)
    result = R.tail_measures(rng.standard_t(5, 5000))
    assert 0.0 < result["concentration"] < 1.0


def test_zero_return_days_are_kept():
    """Retirar dias parados deslocaria os quantis para cima e inflacionaria a
    cauda das accoes menos liquidas -- exactamente o vies que a experiencia tem
    de evitar."""
    rng = np.random.default_rng(13)
    active = rng.normal(0, 0.02, 1000)
    with_zeros = np.concatenate([active, np.zeros(1000)])
    assert R.tail_measures(with_zeros)["n"] == 2000


def test_a_short_series_returns_nothing():
    assert R.tail_measures(np.array([0.01, -0.01, 0.02])) == {}


# --------------------------------------------------------------------------
# 3. Residualizacao
# --------------------------------------------------------------------------

def test_residualising_removes_what_the_control_explains():
    """Se a medida for inteiramente uma funcao linear do controlo, o residuo tem
    de ser zero -- e nesse caso a persistencia dele nao pode significar nada."""
    rng = np.random.default_rng(21)
    rows = []
    for period in range(2000, 2010):
        for ticker in range(40):
            control = rng.normal()
            rows.append({"ticker": f"T{ticker}", "period": period,
                         "tail_ratio": 3.0 + 2.0 * control,
                         "zeros": control, "log_price": 0.0,
                         "log_dollar_volume": 0.0, "log_volatility": 0.0,
                         "volatility": 0.02})
    residual = R.residualise(_panel(rows), "tail_ratio")
    assert np.allclose(residual.dropna().values, 0.0, atol=1e-8)


# --------------------------------------------------------------------------
# 4. Persistencia e placebo
# --------------------------------------------------------------------------

def _persistent_panel(strength, seed=31, n_tickers=250, periods=range(2000, 2020)):
    rng = np.random.default_rng(seed)
    character = {f"T{i}": rng.normal() for i in range(n_tickers)}
    rows = []
    for period in periods:
        for ticker, level in character.items():
            rows.append({"ticker": ticker, "period": period,
                         "tail_ratio": 3.0 + strength * level + rng.normal(0, 1.0),
                         "volatility": 0.02})
    return _panel(rows)


def test_persistence_finds_a_real_character():
    table = R.persistence(_persistent_panel(strength=3.0), "tail_ratio")
    assert table["spearman"].mean() > 0.5


def test_persistence_is_absent_when_there_is_no_character():
    """O painel de ruido tem de ser GRANDE para este teste significar alguma
    coisa. Com 60 tickers e 12 periodos, o desvio-padrao da persistencia media
    e' ~0.04 e um painel sem estrutura nenhuma chega facilmente a 0.10 por
    acaso -- foi o que aconteceu na primeira versao deste teste. Com 250
    tickers e 20 periodos o acaso fica em ~0.015, e o teste passa a testar."""
    table = R.persistence(_persistent_panel(strength=0.0), "tail_ratio")
    assert abs(table["spearman"].mean()) < 0.05


def test_identity_placebo_confirms_a_real_effect():
    """O placebo tem de dar p pequeno quando ha' efeito -- senao nao serve de
    prova nenhuma quando o der."""
    result = R.identity_placebo(_persistent_panel(strength=3.0), "tail_ratio", n=100)
    assert result["observed"] > 0.5
    assert result["p_value"] < 0.05


def test_identity_placebo_does_not_fire_on_noise():
    frame = _persistent_panel(strength=0.0)
    assert R.identity_placebo(frame, "tail_ratio", n=100)["p_value"] > 0.10


# --------------------------------------------------------------------------
# 5. A tendencia secular e o controlo de composicao
# --------------------------------------------------------------------------

def test_secular_trend_uses_only_tickers_present_in_every_decade():
    """O erro obvio desta medicao: ha' 19 accoes nos anos 60 e 219 nos 2020, e
    uma subida podia ser apenas quem entrou. So' contam as que estao em todas."""
    rows = []
    for period in range(1980, 2026):
        tickers = [f"OLD{i}" for i in range(40)]
        if period >= 2015:
            tickers += [f"NEW{i}" for i in range(40)]
        for ticker in tickers:
            # As novas tem concentracao muito mais alta: se entrarem na conta,
            # fabricam uma tendencia que nao existe.
            rows.append({"ticker": ticker, "period": period,
                         "concentration": 0.9 if ticker.startswith("NEW") else 0.20,
                         "volatility": 0.02})
    result = R.secular_trend(_panel(rows), exclude=())
    assert result["n_constant"] == 40
    assert result["slope"] == pytest.approx(0.0, abs=1e-9)


def test_secular_trend_detects_a_real_rise():
    rows = []
    for period in range(1980, 2026):
        for ticker in range(40):
            rows.append({"ticker": f"T{ticker}", "period": period,
                         "concentration": 0.15 + 0.002 * (period - 1980),
                         "volatility": 0.02})
    result = R.secular_trend(_panel(rows), exclude=())
    assert result["slope"] == pytest.approx(0.002, rel=1e-6)
    assert result["p_value"] < 1e-6


def test_secular_trend_needs_enough_constant_tickers():
    rows = [{"ticker": "A", "period": p, "concentration": 0.2, "volatility": 0.02}
            for p in range(1980, 2026)]
    assert R.secular_trend(_panel(rows)) == {}


# --------------------------------------------------------------------------
# 6. Decadencia
# --------------------------------------------------------------------------

def test_decay_is_monotone_for_a_fading_character():
    """Uma propriedade que se dilui com o tempo tem de dar persistencia a cair
    com a distancia. Se nao der, a medicao de decadencia nao mede nada."""
    rng = np.random.default_rng(41)
    n_tickers, periods = 60, list(range(2000, 2026))
    level = {f"T{i}": rng.normal() for i in range(n_tickers)}
    rows = []
    for index, period in enumerate(periods):
        for ticker in level:
            # O caracter e' substituido lentamente por ruido novo.
            level[ticker] = 0.85 * level[ticker] + 0.15 * rng.normal()
            rows.append({"ticker": ticker, "period": period,
                         "tail_ratio": 3.0 + 3.0 * level[ticker] + rng.normal(0, 0.5),
                         "volatility": 0.02})
    table = R.persistence_decay(_panel(rows), "tail_ratio",
                                lags=(1, 5, 15)).set_index("lag_anos")
    assert table.loc[1, "spearman"] > table.loc[5, "spearman"]
    assert table.loc[5, "spearman"] > table.loc[15, "spearman"]
