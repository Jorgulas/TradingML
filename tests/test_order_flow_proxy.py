"""Testes do proxy de fluxo assinado (experiencia N).

Tres propriedades que, se se partirem, invalidam a experiencia inteira:

  1. A NORMALIZACAO DO VOLUME. A metrica e' a correlacao transversal dentro de
     cada barra. Sem dividir pelo volume medio do proprio ticker, a AAPL fica
     sempre no topo por negociar mais, e o que se mede e' capitalizacao
     bolsista disfarcada de fluxo.

  2. O CLV tem de ser exactamente o que diz ser, incluindo no caso degenerado
     de uma barra sem intervalo -- que ao minuto acontece muitas vezes e sao
     precisamente as barras sem fluxo nenhum.

  3. O SENTIDO DA CARTEIRA e' estimado no treino. Foi um erro real desta
     experiencia: a carteira ia sempre comprada no topo do ranking, o IC era
     negativo, e portanto ela negociava ao contrario -- reportava prejuizo
     naquilo que era o sinal a funcionar. A correccao nao pode ser "olhar para
     a resposta", ou o lucro passa a ser construido em vez de medido.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research import order_flow_proxy as N


def _bar(open_, high, low, close, volume=1000):
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _frame(rows, start="2026-08-17 13:30", freq="1min"):
    return pd.DataFrame(rows, index=pd.date_range(start, periods=len(rows),
                                                  freq=freq, tz="UTC"))


# --------------------------------------------------------------------------
# 1. O close location value
# --------------------------------------------------------------------------

def test_clv_is_plus_one_when_the_bar_closes_at_its_high():
    bars = _frame([_bar(100, 110, 90, 110)])
    assert N.close_location_value(bars).iloc[0] == pytest.approx(1.0)


def test_clv_is_minus_one_when_the_bar_closes_at_its_low():
    bars = _frame([_bar(100, 110, 90, 90)])
    assert N.close_location_value(bars).iloc[0] == pytest.approx(-1.0)


def test_clv_is_zero_at_the_middle_of_the_range():
    bars = _frame([_bar(100, 110, 90, 100)])
    assert N.close_location_value(bars).iloc[0] == pytest.approx(0.0)


def test_a_bar_with_no_range_is_neutral_not_missing():
    """Ao minuto ha' muitas barras em que o preco nao se mexe. Sao informacao
    neutra -- nao houve fluxo -- e deixa-las NaN eliminaria em silencio
    precisamente os minutos mais calmos, enviesando a amostra para os agitados."""
    bars = _frame([_bar(100, 100, 100, 100)])
    value = N.close_location_value(bars).iloc[0]
    assert value == 0.0
    assert not np.isnan(value)


def test_clv_stays_inside_minus_one_and_one():
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(200):
        low = 100 + rng.normal(0, 2)
        high = low + abs(rng.normal(1, 0.5))
        close = rng.uniform(low, high)
        rows.append(_bar(low, high, low, close))
    values = N.close_location_value(_frame(rows))
    assert values.between(-1.0, 1.0).all()


# --------------------------------------------------------------------------
# 2. A normalizacao do volume (a armadilha que invalidaria tudo)
# --------------------------------------------------------------------------

def _identical_shape(volume_scale, n=120):
    """Duas accoes com EXACTAMENTE o mesmo padrao de fechos, e volumes que
    diferem por um factor constante."""
    rng = np.random.default_rng(11)
    rows = []
    for _ in range(n):
        low = 100.0
        high = 101.0
        close = rng.uniform(low, high)
        rows.append(_bar(low, high, low, close, volume=1000 * volume_scale))
    return _frame(rows)


def test_volume_normalisation_makes_two_sizes_identical():
    """A propriedade que faz a comparacao transversal significar alguma coisa:
    uma accao que negoceia cem vezes mais nao pode ficar cem vezes mais alta no
    ranking so' por isso."""
    small = N.build_signals(_identical_shape(1), session_aware=False)
    large = N.build_signals(_identical_shape(100), session_aware=False)
    valid = small["ofi"].notna() & large["ofi"].notna()
    assert valid.sum() > 50
    assert np.allclose(small.loc[valid, "ofi"], large.loc[valid, "ofi"])


def test_without_normalisation_the_bigger_ticker_would_dominate():
    """O contrafactual, para o teste anterior nao ser vacuo: em bruto, o sinal
    da accao grande e' cem vezes maior."""
    small = _identical_shape(1)
    large = _identical_shape(100)
    raw_small = (N.close_location_value(small) * small["volume"]).abs().mean()
    raw_large = (N.close_location_value(large) * large["volume"]).abs().mean()
    assert raw_large == pytest.approx(100 * raw_small)


def test_relative_volume_reacts_to_a_volume_spike():
    """Normalizar nao pode achatar o sinal: uma barra com dez vezes o volume
    normal tem de pesar mais do que uma barra normal."""
    rows = [_bar(100, 101, 100, 101, volume=1000) for _ in range(80)]
    rows.append(_bar(100, 101, 100, 101, volume=10_000))
    signals = N.build_signals(_frame(rows), session_aware=False)
    assert signals["ofi"].iloc[-1] > 5 * signals["ofi"].iloc[-2]


# --------------------------------------------------------------------------
# 3. A sessao
# --------------------------------------------------------------------------

def test_accumulation_restarts_each_session():
    """A soma movel de fluxo nao pode arrastar ontem para hoje: a acumulacao
    intradiaria e' sobre o que aconteceu DESDE A ABERTURA."""
    day_one = [_bar(100, 101, 100, 101) for _ in range(80)]
    day_two = [_bar(100, 101, 100, 100) for _ in range(80)]
    index = (list(pd.date_range("2026-08-17 13:30", periods=80, freq="1min", tz="UTC"))
             + list(pd.date_range("2026-08-18 13:30", periods=80, freq="1min", tz="UTC")))
    bars = pd.DataFrame(day_one + day_two, index=pd.DatetimeIndex(index))

    signals = N.build_signals(bars, session_aware=True)
    first_bars_of_day_two = signals["ofi_20"].iloc[80:99]
    assert first_bars_of_day_two.isna().all()


def test_naive_accumulation_would_carry_yesterday_over():
    """O contrafactual do teste anterior."""
    day_one = [_bar(100, 101, 100, 101) for _ in range(80)]
    day_two = [_bar(100, 101, 100, 100) for _ in range(80)]
    index = (list(pd.date_range("2026-08-17 13:30", periods=80, freq="1min", tz="UTC"))
             + list(pd.date_range("2026-08-18 13:30", periods=80, freq="1min", tz="UTC")))
    bars = pd.DataFrame(day_one + day_two, index=pd.DatetimeIndex(index))

    naive = N.build_signals(bars, session_aware=False)
    assert naive["ofi_20"].iloc[80:99].notna().any()


# --------------------------------------------------------------------------
# 4. O sinal acumulado e' a soma do sinal por barra
# --------------------------------------------------------------------------

def test_the_sum_is_the_sum():
    rng = np.random.default_rng(7)
    rows = [_bar(100, 101, 100, rng.uniform(100, 101),
                 volume=int(rng.integers(500, 2000))) for _ in range(150)]
    signals = N.build_signals(_frame(rows), session_aware=False)
    manual = signals["ofi"].rolling(20, min_periods=20).sum()
    valid = signals["ofi_20"].notna()
    assert np.allclose(signals.loc[valid, "ofi_20"], manual[valid])


def test_every_declared_signal_is_produced():
    """A grelha itera sobre SIGNALS; um nome declarado e nao produzido daria uma
    celula em falta em silencio."""
    rows = [_bar(100, 101, 100, 100.5, volume=1000) for _ in range(200)]
    signals = N.build_signals(_frame(rows), session_aware=False)
    for name in N.SIGNALS:
        assert name in signals.columns


# --------------------------------------------------------------------------
# 5. O sentido da carteira
# --------------------------------------------------------------------------

def test_direction_comes_from_the_training_slice_only():
    """O teste que protege contra o erro que esta experiencia cometeu ao
    contrario: escolher o sentido pela amostra inteira construiria lucro em vez
    de o medir.

    Constroi-se uma serie em que o sinal INVERTE a meio. Se o sentido fosse
    estimado com tudo, sairia a media das duas metades; estimado so' no treino,
    tem de sair o da primeira."""
    rng = np.random.default_rng(2)
    stamps = pd.date_range("2026-08-17 13:30", periods=200, freq="1min", tz="UTC")
    rows = []
    for index, stamp in enumerate(stamps):
        # Primeira metade: sinal e retorno alinhados. Segunda: opostos.
        sign = 1.0 if index < 100 else -1.0
        for ticker in range(20):
            value = rng.normal()
            rows.append({"ts": stamp, "ticker": f"T{ticker}",
                         "ofi_20": value, "forward_3": sign * value})
    frame = pd.DataFrame(rows)

    from src.research.tangent_intraday import bar_ic
    stamps_sorted = np.sort(frame["ts"].unique())
    boundary = stamps_sorted[int(len(stamps_sorted) * 0.7)]
    train = frame[frame["ts"] < boundary]

    assert np.sign(bar_ic(train, "ofi_20", 3).mean()) == 1.0     # so' o treino
    assert np.sign(bar_ic(frame, "ofi_20", 3).mean()) != 1.0     # tudo junto mente
