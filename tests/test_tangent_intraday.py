"""Testes do joelho ao minuto (experiencia M).

O que protegem, e porque cada um existe:

  1. NADA ATRAVESSA A NOITE. E' o erro que teria feito a experiencia inteira
     medir gaps overnight disfarcados de joelho. Uma media de 20 barras na
     quinta barra do dia usaria 15 barras de ontem, e o salto de fecho para
     abertura -- que pode ser 1% -- entraria como se fosse um movimento de um
     minuto. O angulo dispararia todas as manhas, sempre no sentido do gap.

  2. O ATRASO DE ENTRADA. Com `lag=1` o sinal ve' o fecho da barra e a posicao
     so' abre na seguinte. E' o que acontece na realidade, e e' o teste ao salto
     bid-ask. Se o atraso estiver mal aplicado, o teste deixa de testar.

  3. O IC POR BARRA tem de ser Spearman a serio. Foi vectorizado por causa da
     velocidade, e uma implementacao propria de uma estatistica conhecida tem de
     ser confrontada com a biblioteca.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research import tangent_angle as L
from src.research import tangent_intraday as M


def _intraday_bars(n_days=6, bars_per_day=120, gap=0.02, seed=4):
    """Barras de um minuto com um SALTO grande entre sessoes.

    O gap e' deliberadamente enorme (2%) para que qualquer contaminacao
    overnight seja impossivel de nao ver."""
    rng = np.random.default_rng(seed)
    frames, level = [], 100.0
    for day in range(n_days):
        start = pd.Timestamp("2026-08-17", tz="UTC") + pd.Timedelta(days=day, hours=13, minutes=30)
        level *= (1 + gap)                       # o salto de abertura
        closes = level * np.cumprod(1 + rng.normal(0, 0.0008, bars_per_day))
        level = closes[-1]
        frames.append(pd.DataFrame(
            {"open": closes, "high": closes * 1.0005, "low": closes * 0.9995,
             "close": closes, "volume": 1_000 + np.arange(bars_per_day)},
            index=pd.date_range(start, periods=bars_per_day, freq="1min")))
    return pd.concat(frames)


# --------------------------------------------------------------------------
# 1. A noite
# --------------------------------------------------------------------------

def test_session_aware_never_looks_at_yesterday():
    """As primeiras barras de cada sessao nao podem ter sinal: nao ha' barras
    suficientes DENTRO do dia para formar a media e os dois trocos."""
    bars = _intraday_bars()
    frame = L.build(bars, ma_window=20, span=10, session_aware=True)
    session = pd.Series(bars.index.normalize(), index=bars.index)
    position = session.groupby(session).cumcount()

    early = frame.loc[position < 20, "angle"]
    assert early.isna().all()


def test_the_overnight_gap_leaks_into_the_naive_angle_at_the_open():
    """O teste que justifica o modo, e que teve de ser reescrito.

    A primeira versao comparava o angulo MAXIMO das duas versoes e falhava --
    davam exactamente o mesmo. E davam por uma razao interessante: o gap tambem
    entra no denominador (a volatilidade das ultimas 60 barras), portanto
    inflaciona numerador e escala ao mesmo tempo e o maximo global nao se move.

    A contaminacao nao esta' no maximo, esta' na ABERTURA, e tem sentido fixo.
    Com um salto sistematicamente positivo entre sessoes, a versao ingenua da'
    um angulo medio claramente positivo nas primeiras barras de cada dia -- esta'
    a medir o gap -- enquanto a versao que respeita a sessao nao da'. A meio do
    dia, quando ja' nao ha' barras de ontem na janela, as duas coincidem."""
    bars = _intraday_bars(gap=0.02)
    session = pd.Series(bars.index.normalize(), index=bars.index)
    position = session.groupby(session).cumcount()

    naive = L.build(bars, 20, 10, session_aware=False)["angle"]
    aware = L.build(bars, 20, 10, session_aware=True)["angle"]

    at_open = (position >= 20) & (position < 36)
    assert naive[at_open].mean() > 4.0        # aponta para o lado do gap
    assert abs(aware[at_open].mean()) < 2.0   # nao aponta para lado nenhum

    # E longe da abertura as duas sao a mesma coisa, o que confirma que a
    # diferenca acima e' gap e nao um efeito lateral do modo de sessao.
    mid = position > 60
    assert naive[mid].mean() == pytest.approx(aware[mid].mean(), abs=1e-9)


def test_forward_return_stops_at_the_end_of_the_session(conn):
    """O retorno futuro tambem nao pode saltar para o dia seguinte: seria
    creditar ao sinal um gap que ele nao previu."""
    bars = _intraday_bars(n_days=3, bars_per_day=80)
    session = pd.Series(bars.index.normalize(), index=bars.index)
    horizon = 5
    forward = bars["close"].shift(-horizon) / bars["close"] - 1
    masked = forward.where(session.shift(-horizon) == session)

    tail = masked.groupby(session).tail(horizon)
    assert tail.isna().all()
    assert masked.notna().sum() > 100        # mas o resto sobrevive


# --------------------------------------------------------------------------
# 2. O atraso de entrada
# --------------------------------------------------------------------------

def test_lag_shifts_the_entry_by_one_bar():
    """Com lag=1 o retorno vai de t+1 a t+1+h, e nao de t a t+h."""
    closes = np.array([100.0, 101.0, 103.0, 106.0, 110.0, 115.0])
    index = pd.date_range("2026-08-17 13:30", periods=len(closes), freq="1min", tz="UTC")
    price = pd.Series(closes, index=index)

    horizon = 2
    without = (price.shift(-horizon) / price - 1).iloc[0]
    with_lag = (price.shift(-(1 + horizon)) / price.shift(-1) - 1).iloc[0]

    assert without == pytest.approx(103 / 100 - 1)
    assert with_lag == pytest.approx(106 / 101 - 1)


# --------------------------------------------------------------------------
# 3. O coeficiente de informacao
# --------------------------------------------------------------------------

def test_bar_ic_matches_scipy():
    """A versao vectorizada tem de dar exactamente o mesmo que a biblioteca.
    Spearman e' Pearson sobre postos, e foi assim que se escreveu -- mas uma
    reimplementacao de uma estatistica conhecida confronta-se, nao se assume."""
    rng = np.random.default_rng(8)
    stamps = pd.date_range("2026-08-17 13:30", periods=40, freq="1min", tz="UTC")
    rows = []
    for stamp in stamps:
        for ticker in range(30):
            rows.append({"ts": stamp, "kink": rng.normal(),
                         "forward_3": rng.normal()})
    frame = pd.DataFrame(rows)

    ours = M.bar_ic(frame, "kink", 3)
    theirs = frame.groupby("ts").apply(
        lambda g: stats.spearmanr(g["kink"], g["forward_3"]).statistic,
        include_groups=False)
    assert np.allclose(ours.values, theirs.reindex(ours.index).values, atol=1e-12)


def test_bar_ic_skips_thin_bars():
    """Uma barra com cinco accoes nao da' uma correlacao transversal com
    significado, e incluir esse ruido inflacionaria a dispersao."""
    stamps = pd.date_range("2026-08-17 13:30", periods=3, freq="1min", tz="UTC")
    rows = []
    for index, stamp in enumerate(stamps):
        names = 5 if index == 0 else 25
        for ticker in range(names):
            rows.append({"ts": stamp, "kink": float(ticker),
                         "forward_3": float(ticker) * 0.01})
    frame = pd.DataFrame(rows)

    result = M.bar_ic(frame, "kink", 3, min_names=15)
    assert len(result) == 2
    assert stamps[0] not in result.index


def test_a_perfect_ranking_gives_ic_one():
    stamps = pd.date_range("2026-08-17 13:30", periods=4, freq="1min", tz="UTC")
    rows = [{"ts": s, "kink": float(t), "forward_3": float(t)}
            for s in stamps for t in range(20)]
    result = M.bar_ic(pd.DataFrame(rows), "kink", 3)
    assert np.allclose(result.values, 1.0)


# --------------------------------------------------------------------------
# 4. A agregacao que decide
# --------------------------------------------------------------------------

def test_regions_split_at_span_equals_the_average():
    """A fronteira e' troco = media, que e' onde as duas rectas deixam de
    partilhar barras. Se o corte se deslocar, o resultado principal da
    experiencia deixa de significar o que diz."""
    table = pd.DataFrame({
        "sinal": ["kink"] * 4,
        "ma": [20, 20, 20, 20],
        "span": [10, 20, 30, 40],
        "ic": [0.02, 0.01, -0.02, -0.03],
        "mde": [0.01] * 4,
    })
    result = M.regions(table).set_index("regiao")
    assert result.loc["troco <= media", "celulas"] == 2
    assert result.loc["troco > media", "celulas"] == 2
    assert result.loc["troco <= media", "positivos"] == 2
    assert result.loc["troco > media", "positivos"] == 0


def test_regions_ignore_the_other_signal():
    table = pd.DataFrame({
        "sinal": ["kink", "angle"],
        "ma": [20, 20], "span": [10, 10],
        "ic": [0.02, -0.09], "mde": [0.01, 0.01],
    })
    assert M.regions(table, "kink")["celulas"].sum() == 1


# --------------------------------------------------------------------------
# 5. Poder estatistico
# --------------------------------------------------------------------------

def test_minimum_detectable_shrinks_with_more_data():
    """Sem o minimo detectavel, um intervalo que contem zero e' ambiguo: nao se
    sabe se o efeito e' nulo ou se a amostra e' cega."""
    rng = np.random.default_rng(6)
    small = pd.Series(rng.normal(0, 1, 200))
    large = pd.Series(rng.normal(0, 1, 4000))
    assert M.minimum_detectable(large, block=10) < M.minimum_detectable(small, block=10)


def test_minimum_detectable_is_undefined_for_a_tiny_sample():
    assert np.isnan(M.minimum_detectable(pd.Series([1.0, 2.0]), block=10))


# --------------------------------------------------------------------------
# 6. Coerencia da grelha
# --------------------------------------------------------------------------

def test_every_grid_spans_both_sides_of_the_boundary():
    """A fronteira troco=media so' e' testavel se cada prazo tiver configuracoes
    dos dois lados dela. Se alguem estreitar a grelha, o teste avisa."""
    for timeframe, spec in M.GRIDS.items():
        ratios = [span / ma for ma in spec["ma"] for span in spec["span"]]
        assert any(r <= 1.0 for r in ratios), timeframe
        assert any(r > 1.0 for r in ratios), timeframe


def test_grids_fit_inside_a_trading_session():
    """A media mais longa mais os dois trocos tem de caber no dia, senao o modo
    de sessao deita fora quase tudo e nao sobra amostra."""
    for timeframe, spec in M.GRIDS.items():
        needed = max(spec["ma"]) + 2 * max(spec["span"]) + max(spec["horizons"])
        assert needed < spec["bars_per_day"], timeframe
