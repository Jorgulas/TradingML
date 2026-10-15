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


# --------------------------------------------------------------------------
# 5. O ano inteiro, e a indivisibilidade do contrato
# --------------------------------------------------------------------------

def _sessions(pnl, balance=V.ACCOUNT_USD, days=None):
    pnl = np.asarray(pnl, dtype=float)
    return pd.DataFrame({"liquido_usd": pnl, "retorno": pnl / balance,
                         "session": days if days is not None else range(len(pnl))})


def test_fixed_size_falls_in_a_straight_line_and_proportional_does_not():
    """A CORRECCAO QUE ESTA FUNCAO EXISTE PARA FAZER.

    Com um contrato minimo indivisivel a perda em dolares nao encolhe quando a
    conta encolhe, portanto a conta desce em linha recta e chega ao fim. Compor
    percentagens pressupoe tamanho fraccionario e nunca chega a zero -- foi o
    que a ronda 19 fez, e subestimou o estrago."""
    losing = _sessions([-20.0] * 30)
    fixed = V.year(losing, sessions=252, sizing="fixo", draws=300)
    scaled = V.year(losing, sessions=252, sizing="proporcional", draws=300)

    assert fixed["prob_ruina"] == pytest.approx(1.0)
    assert scaled["prob_ruina"] == 0.0
    assert scaled["mediana_fim"] > V.DAY_MARGIN


def test_sessions_to_ruin_matches_the_arithmetic():
    """Perda fixa de 20 USD sobre 3366: a margem de 200 e' atingida ao fim de
    (3366-200)/20 = 159 sessoes. Se nao bater, a contabilidade esta' errada."""
    result = V.year(_sessions([-20.0] * 10), sessions=252, sizing="fixo",
                    draws=200)
    assert result["sessoes_ate_ruina"] == pytest.approx(159, abs=1)


def test_a_winning_distribution_does_not_ruin():
    """Sem este, o teste anterior nao prova nada: um simulador que arruinasse
    sempre passaria os dois."""
    result = V.year(_sessions([+20.0] * 10), sessions=252, sizing="fixo",
                    draws=200)
    assert result["prob_ruina"] == 0.0
    assert result["prob_lucro"] == pytest.approx(1.0)
    assert result["mediana_fim"] == pytest.approx(V.ACCOUNT_USD + 252 * 20)


def test_a_fair_coin_ends_where_it_started():
    """O contrafactual sem custos: se o bruto for mesmo zero em media, o ano
    acaba onde comecou e a probabilidade de lucro e' metade."""
    result = V.year(_sessions([+60.0, -60.0] * 20), sessions=252,
                    sizing="fixo", draws=3000, by_day=False)
    assert result["mediana_fim"] == pytest.approx(V.ACCOUNT_USD, rel=0.08)
    assert result["prob_lucro"] == pytest.approx(0.5, abs=0.08)


def test_day_resampling_weights_days_equally_not_windows():
    """Sao 99 janelas mas 20 dias distintos, e os dias nao tem todos o mesmo
    numero de janelas. O dia e' a unidade independente: um dia com o triplo das
    janelas nao vale o triplo.

    Construcao que separa as duas: 30 janelas de +100 num dia, 10 de -100
    noutro. Por JANELA a media e' +50; por DIA e' zero. Se a funcao devolvesse
    o mesmo nos dois casos, nao estaria a reamostrar dias de todo."""
    pnl = [100.0] * 30 + [-100.0] * 10
    days = ["bom"] * 30 + ["mau"] * 10
    frame = _sessions(pnl, days=days)

    by_day = V.year(frame, sessions=200, sizing="fixo", draws=1500, by_day=True)
    by_window = V.year(frame, sessions=200, sizing="fixo", draws=1500,
                       by_day=False)

    # Por janela ganha-se +50 por sessao em media: 200 x 50 = +10000.
    assert by_window["mediana_fim"] > V.ACCOUNT_USD + 7000
    # Por dia a esperanca e' zero, portanto a conta ou aguenta perto do inicio
    # ou arruina-se pelo caminho -- nunca acumula dez mil.
    assert by_day["mediana_fim"] < V.ACCOUNT_USD + 2000
    assert by_day["prob_lucro"] < by_window["prob_lucro"]


def test_year_needs_a_non_empty_result():
    assert V.year(pd.DataFrame(), sessions=252) == {}


# --------------------------------------------------------------------------
# 6. O RSI(2) ao minuto
# --------------------------------------------------------------------------

def _rsi_frame(close, session="2026-03-02"):
    close = np.asarray(close, dtype=float)
    frame = pd.DataFrame({
        "ts": pd.date_range(f"{session} 14:30", periods=len(close), freq="1min",
                            tz="UTC"),
        "open": close, "high": close, "low": close, "close": close, "volume": 1,
    })
    frame = T.add_session(frame)
    frame["ma_fast"] = frame["close"].rolling(3).mean()
    frame["ma_slow"] = frame["close"].rolling(5).mean()
    frame["rsi"] = 0.0
    return frame


def test_connors_exit_fires_on_the_first_recovery():
    """Sai na PRIMEIRA barra em que o fecho volta acima da media rapida. Se
    saisse mais tarde, estaria a escolher o melhor momento com informacao que
    nao tinha."""
    frame = _rsi_frame([100, 99, 98, 97, 96, 95, 110, 130, 150])
    gross, held = V.resolve_connors(frame, 5, 1, hi=8, max_bars=10)
    assert held == 1                       # recupera logo na barra seguinte
    assert gross == pytest.approx(110 - 95)


def test_connors_exit_never_crosses_the_session():
    """Uma posicao sem stop num futuro alavancado nao atravessa a noite."""
    first = _rsi_frame([100] * 6 + [90, 90], session="2026-03-02")
    second = _rsi_frame([500] * 8, session="2026-03-09")
    frame = pd.concat([first, second]).reset_index(drop=True)
    frame = T.add_session(frame)
    frame["ma_fast"] = frame["close"].rolling(3).mean()
    gross, _ = V.resolve_connors(frame, 6, 1, hi=15, max_bars=10)
    assert gross == pytest.approx(0.0, abs=1e-9)   # fecha na sua propria sessao


def test_connors_exit_stops_at_the_window_edge():
    frame = _rsi_frame([100, 99, 98, 97, 96, 95, 94, 93, 92, 91])
    _, held = V.resolve_connors(frame, 4, 1, hi=7, max_bars=50)
    assert 4 + held <= 7


def test_the_short_side_mirrors_the_long():
    """Uma venda sai quando o preco cai ABAIXO da media rapida. Sem isto o
    resultado dependeria de o indice ter subido na amostra."""
    frame = _rsi_frame([100, 101, 102, 103, 104, 105, 90, 80, 70])
    gross, held = V.resolve_connors(frame, 5, -1, hi=8, max_bars=10)
    assert held == 1
    assert gross == pytest.approx(105 - 90)


def test_rsi_signals_need_all_three_conditions():
    frame = _rsi_frame(np.linspace(100, 120, 40))
    frame["rsi"] = 50.0                                # RSI nunca abaixo de 10
    assert V.rsi_signals(frame) == {}
    frame["rsi"] = 5.0                                 # agora dispara onde couber
    frame.loc[:, "ma_fast"] = frame["close"] + 1.0     # fecho abaixo da rapida
    frame.loc[:, "ma_slow"] = frame["close"] - 1.0     # fecho acima da lenta
    assert len(V.rsi_signals(frame)) > 0


def test_short_signals_only_appear_when_asked():
    frame = _rsi_frame(np.linspace(120, 100, 40))
    frame["rsi"] = 95.0
    frame.loc[:, "ma_fast"] = frame["close"] - 1.0
    frame.loc[:, "ma_slow"] = frame["close"] + 1.0
    assert V.rsi_signals(frame, short_side=False) == {}
    assert all(v == -1 for v in V.rsi_signals(frame, short_side=True).values())
