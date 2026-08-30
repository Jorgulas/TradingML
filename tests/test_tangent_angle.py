"""Testes do angulo da tangente a' media movel (experiencia L).

O que protegem, por ordem de gravidade:

  1. A NORMALIZACAO. E' a correccao que torna a ideia mensuravel: sem ela o
     angulo depende da escala do preco e do zoom do grafico, e um limiar de 25
     graus significa coisas diferentes na AAPL e na KO. Se alguem trocar a
     escala por uma constante, os testes tem de gritar.

  2. O SINAL DO JOELHO. A regra inteira assenta em "media plana que passa a
     subir da' joelho positivo, e quando volta a ficar plana da' negativo". Se
     este sinal se inverter, a regra passa a fazer o contrario do que diz.

  3. A ZONA MORTA. Dentro do limiar a posicao tem de ser exactamente zero, e
     nao um numero pequeno -- nao agir e' uma decisao, nao um arredondamento.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research import tangent_angle as L


def _bars(closes):
    closes = np.asarray(closes, dtype=float)
    index = pd.date_range("2025-01-02", periods=len(closes), freq="B", tz="UTC")
    return pd.DataFrame({"open": closes, "high": closes * 1.001, "low": closes * 0.999,
                         "close": closes, "volume": 10_000 + np.arange(len(closes))},
                        index=index)


def _flat_then_rising(n_flat=150, n_rise=40, step=1.0, noise=0.8, seed=1):
    """O cenario exacto do enunciado: media plana e, no fim, a subir.

    O RUIDO tem de ser realista, e nao decorativo. Uma rampa perfeitamente lisa
    tem volatilidade quase nula, e como o angulo e' medido em unidades de
    volatilidade por barra, o denominador colapsa e o angulo satura nos 90 graus
    -- com um joelho residual permanente que nao existe em dados reais. Com
    ruido de 0.8% por barra sobre um preco de 100 (a ordem de grandeza de uma
    accao verdadeira) o problema desaparece."""
    rng = np.random.default_rng(seed)
    flat = 100 + rng.normal(0, noise, n_flat)
    rising = 100 + step * np.arange(1, n_rise + 1) + rng.normal(0, noise, n_rise)
    return np.concatenate([flat, rising])


# --------------------------------------------------------------------------
# 1. Normalizacao
# --------------------------------------------------------------------------

def test_angle_is_invariant_to_the_price_level():
    """Uma accao de 500 dolares e uma de 5 com exactamente o mesmo movimento
    relativo tem de dar o MESMO angulo. E' a razao de ser da normalizacao: sem
    ela, o limiar de 25 graus dependeria do preco da accao."""
    prices = _flat_then_rising()
    cheap = L.build(_bars(prices), 20, 10)["angle"]
    expensive = L.build(_bars(prices * 100), 20, 10)["angle"]
    both = pd.concat([cheap, expensive], axis=1).dropna()
    assert len(both) > 50
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-6)


def test_angle_is_bounded():
    """`arctan` limita o angulo a (-90, 90). E' isso que impede um movimento
    extremo de dominar tudo o resto -- a compressao e' a parte da ideia que
    funciona como regularizacao."""
    violent = np.concatenate([np.full(120, 100.0) + np.arange(120) * 0.001,
                              100 * np.exp(np.arange(30) * 0.4)])
    angle = L.build(_bars(violent), 20, 10)["angle"].dropna()
    assert angle.abs().max() < 90.0
    assert angle.max() > 60.0          # o movimento violento chega perto do tecto


def test_a_flat_series_has_an_angle_near_zero():
    rng = np.random.default_rng(3)
    flat = 100 + rng.normal(0, 0.8, 300)
    angle = L.build(_bars(flat), 20, 10)["angle"].dropna()
    assert angle.abs().median() < 15.0


# --------------------------------------------------------------------------
# 2. O sinal do joelho
# --------------------------------------------------------------------------

def test_flat_then_rising_gives_a_positive_kink():
    """O cenario do enunciado, verificado literalmente."""
    frame = L.build(_bars(_flat_then_rising()), 20, 10).dropna()
    assert frame["kink"].max() > 15.0


def test_rising_then_flat_gives_a_negative_kink():
    """"Se a linha voltar a ficar reta entao o angulo sera' de -30" -- e' a
    metade da regra que manda VENDER, e depende deste sinal."""
    rng = np.random.default_rng(2)
    rising = 100 + 1.0 * np.arange(1, 161) + rng.normal(0, 0.8, 160)
    flat = np.full(40, rising[-1]) + rng.normal(0, 0.8, 40)
    frame = L.build(_bars(np.concatenate([rising, flat])), 20, 10).dropna()
    assert frame["kink"].min() < -15.0


def test_the_kink_only_fires_at_the_transition():
    """Propriedade central do mecanismo, e a que explica a exposicao baixissima
    que ele produz na pratica.

    O joelho e' a diferenca entre duas rectas separadas por `span` barras.
    Assim que a subida dura mais do que `span`, as duas rectas voltam a ser a
    MESMA e o joelho colapsa para zero -- mesmo com a media a subir a pique. O
    sinal e' um impulso curto no instante da viragem, nao um estado.

    Consequencia pratica: uma regra construida sobre ele fica fora do mercado
    quase sempre. No varrimento a 1h, o limiar de 20 graus dava 2.2% de
    exposicao."""
    n_flat, n_rise = 150, 80
    frame = L.build(_bars(_flat_then_rising(n_flat, n_rise)), 20, 10)
    kink = frame["kink"].dropna()

    peak_position = frame.index.get_loc(kink.abs().idxmax())
    # O pico chega perto da viragem, nao no fim de uma subida ja' longa.
    assert n_flat - 5 <= peak_position <= n_flat + 35
    # E no fim, com a subida ja' bem instalada, o joelho e' uma fraccao do pico
    # -- embora o ANGULO continue grande, porque a media continua a subir.
    assert abs(kink.iloc[-1]) < 0.4 * kink.abs().max()
    assert frame["angle"].iloc[-1] > 20.0


def test_the_proposed_threshold_almost_never_fires():
    """Numero pratico que a experiencia produziu, fixado aqui para nao se
    perder: com ruido realista, uma tendencia FORTE (1% de deriva por barra
    contra 0.8% de ruido) da' um joelho de pico a rondar os 20 graus.

    O limiar de 25 graus do enunciado fica portanto acima do que series reais
    produzem, e a regra quase nunca age -- no varrimento a 1h dava 2.2% de
    exposicao. Nao invalida a ideia; diz que a escala de graus precisa de ser
    calibrada aos dados em vez de escolhida por intuicao visual, e foi por isso
    que o varrimento escolheu sempre 5 graus."""
    frame = L.build(_bars(_flat_then_rising(150, 40, step=1.0)), 20, 10).dropna()
    assert frame["kink"].max() < 25.0


def test_a_steady_trend_has_a_kink_near_zero():
    """Uma tendencia constante nao tem joelho nenhum: as duas rectas sao a
    mesma. E' o que distingue a proposta de simples momentum."""
    rng = np.random.default_rng(7)
    steady = 100 + 0.3 * np.arange(300) + rng.normal(0, 0.8, 300)
    frame = L.build(_bars(steady), 20, 10).dropna()
    assert frame["kink"].abs().median() < 3.0
    assert frame["angle"].median() > 10.0     # mas o ANGULO e' claramente positivo


def test_the_two_segments_do_not_overlap():
    """O troco anterior tem de acabar onde o actual comeca. Sobreposicao
    partilharia barras entre as duas rectas e amorteceria o joelho
    artificialmente."""
    frame = L.build(_bars(_flat_then_rising()), 20, 10)
    assert frame["angle_previous"].equals(frame["angle"].shift(10))


# --------------------------------------------------------------------------
# 3. Zona morta e moeda
# --------------------------------------------------------------------------

def test_inside_the_deadband_the_position_is_exactly_zero():
    """Nao agir e' uma decisao, nao um arredondamento."""
    frame = pd.DataFrame({"angle": [0.0, 10.0, -10.0, 24.9],
                          "kink": [0.0, 10.0, -10.0, 24.9]})
    assert (L.positions(frame, "joelho", 25.0) == 0.0).all()


def test_the_coin_saturates_and_keeps_the_sign():
    frame = pd.DataFrame({"angle": [30.0, -30.0, 60.0, -60.0],
                          "kink": [30.0, -30.0, 60.0, -60.0]})
    weights = L.positions(frame, "joelho", 25.0)
    assert weights.iloc[0] == pytest.approx(0.25)     # 5 graus acima de 25
    assert weights.iloc[1] == pytest.approx(-0.25)
    assert weights.iloc[2] == pytest.approx(1.0)      # saturada
    assert weights.iloc[3] == pytest.approx(-1.0)


def test_the_combined_rule_needs_both_to_agree():
    """"ambos" so' age quando a media ja' sobe E esta' a acelerar. Sinais
    opostos tem de dar zero, senao a regra e' apenas o joelho com outro nome."""
    frame = pd.DataFrame({"angle": [40.0, -40.0, 40.0],
                          "kink": [40.0, 40.0, -40.0]})
    weights = L.positions(frame, "ambos", 10.0)
    assert weights.iloc[0] > 0
    assert weights.iloc[1] == 0.0
    assert weights.iloc[2] == 0.0


def test_the_weaker_of_the_two_sets_the_size():
    """Concordar em sinal nao chega: um angulo enorme com um joelho minusculo
    nao e' uma aceleracao forte, e o tamanho tem de reflectir isso."""
    frame = pd.DataFrame({"angle": [80.0], "kink": [30.0]})
    combined = L.positions(frame, "ambos", 10.0).iloc[0]
    only_kink = L.positions(frame, "joelho", 10.0).iloc[0]
    assert combined == pytest.approx(only_kink)


# --------------------------------------------------------------------------
# 4. Contabilidade da carteira
# --------------------------------------------------------------------------

def test_portfolio_divides_by_available_names_not_by_open_positions():
    """Um dia com uma unica aposta nao pode valer tanto como um dia com cem.
    Normalizar pelas posicoes abertas media "quanto rendeu quem apostou"; o que
    interessa e' quanto rendeu o CAPITAL."""
    frame = pd.DataFrame({
        "ts": [pd.Timestamp("2025-01-02", tz="UTC")] * 10,
        "forward_1": [0.10] + [0.0] * 9,
    })
    weight = pd.Series([1.0] + [0.0] * 9)
    assert L.bar_returns(frame, weight, "forward_1").iloc[0] == pytest.approx(0.01)


def test_daily_aggregation_sums_within_the_day():
    index = pd.to_datetime(["2025-01-02T14:00Z", "2025-01-02T15:00Z",
                            "2025-01-03T14:00Z"])
    series = pd.Series([0.01, 0.02, 0.05], index=index)
    aggregated = L.daily(series)
    assert len(aggregated) == 2
    assert aggregated.iloc[0] == pytest.approx(0.03)


# --------------------------------------------------------------------------
# 5. A verificacao que apanhou o falso positivo
# --------------------------------------------------------------------------

def test_consistency_reports_both_partitions():
    """A funcao existe para impedir que um resultado so' do teste passe por
    descoberta. Tem de devolver sempre as duas particoes e dizer se concordam."""
    rng = np.random.default_rng(11)
    days = pd.date_range("2025-01-02", periods=400, freq="B", tz="UTC")
    frame = pd.DataFrame({
        "ts": np.repeat(days, 20),
        "angle": rng.normal(0, 20, 8000),
        "kink": rng.normal(0, 20, 8000),
        "forward_5": rng.normal(0, 0.02, 8000),
    })
    train, test = L.split_frame(frame, 5)
    result = L.consistency(train, test, "angle", 5)
    assert set(result) == {"treino", "teste", "agree"}
    # Ruido puro: nada deve concordar entre particoes.
    assert result["agree"] is False


def test_split_leaves_an_embargo():
    days = pd.date_range("2025-01-02", periods=300, freq="B", tz="UTC")
    frame = pd.DataFrame({"ts": days, "angle": 0.0, "kink": 0.0, "forward_5": 0.0})
    train, test = L.split_frame(frame, 5)
    assert train["ts"].max() < test["ts"].min()
    assert (test["ts"].min() - train["ts"].max()).days >= 5
