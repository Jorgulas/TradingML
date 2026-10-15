"""Testes do RSI(2) de Connors (experiencia U).

O TESTE QUE CARREGA A RONDA e' `test_the_win_rate_is_manufactured_by_the_exit_rule`.

A estrategia sai quando o preco recupera acima da media de 5 dias. Isso
SELECCIONA vencedores por construcao: todo o negocio que fecha por regra fecha
num preco localmente alto, e as perdas sao so' as que nunca recuperaram dentro
do limite de dias. Logo uma taxa de acerto alta nao e' prova de nada -- tem de
aparecer mesmo numa serie sem sinal nenhum. O teste corre a regra completa
sobre uma caminhada aleatoria SEM DERIVA e exige que o acerto fique bem acima
de 50% enquanto a media fica em zero. Se nao aparecesse, a leitura dos dados
reais seria a oposta.

O segundo que importa e' `test_vectorised_wilder_matches_the_loop`: a
suavizacao foi reescrita como `ewm` porque o ciclo em Python custava 69 milhoes
de iteracoes a varrer a grelha, e uma optimizacao que mude a aritmetica
silenciosamente seria pior do que a lentidao.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research import connors_rsi2 as U


def _bars(close, open_=None):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "ts": pd.date_range("1990-01-01", periods=len(close), freq="B", tz="UTC"),
        "open": close if open_ is None else np.asarray(open_, dtype=float),
        "high": close, "low": close, "close": close, "volume": 1,
    })


def _walk(n=2500, drift=0.0, sigma=0.015, seed=7):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(drift, sigma, n)))


# --------------------------------------------------------------------------
# 1. O teste que carrega a ronda
# --------------------------------------------------------------------------

def test_the_win_rate_is_manufactured_by_the_exit_rule():
    """Sem deriva nenhuma, a regra tem de dar um acerto alto e media zero."""
    frames = []
    for seed in range(25):
        bars = U.indicators(_bars(_walk(seed=seed)))
        trades = U.build_trades(bars, U.entry_mask(bars), U.MAX_HOLD, "fecho")
        if not trades.empty:
            frames.append(trades)
    result = U.summary(pd.concat(frames), cost_bps=0.0)

    assert result["n"] > 1500
    assert result["acerto"] > 0.55, result          # acerto alto...
    assert abs(result["media"]) < 0.006, result     # ...com esperanca zero
    # E o pagamento e' assimetrico no sentido que explica as duas coisas.
    assert result["perda_media"] > result["ganho_medio"]


def test_drift_is_what_moves_the_mean_not_the_rule():
    """Controlo do teste anterior: com deriva positiva a media sobe. Sem isto
    nao se saberia se a media zero vinha da regra ou de o aparelho nao medir."""
    frames = []
    for seed in range(25):
        bars = U.indicators(_bars(_walk(drift=0.0006, seed=seed)))
        trades = U.build_trades(bars, U.entry_mask(bars), U.MAX_HOLD, "fecho")
        if not trades.empty:
            frames.append(trades)
    assert U.summary(pd.concat(frames), cost_bps=0.0)["media"] > 0.002


# --------------------------------------------------------------------------
# 2. O indicador
# --------------------------------------------------------------------------

def test_vectorised_wilder_matches_the_loop():
    close = _walk(n=4000, seed=3)
    for period in (2, 3, 14):
        fast = U.wilder_rsi(close, period)
        slow = U._wilder_loop(close, period)
        assert np.allclose(fast, slow, equal_nan=True, atol=1e-10), period


def test_rsi_saturates_on_monotone_series():
    rising = np.arange(1, 60, dtype=float)
    falling = rising[::-1].copy()
    assert U.wilder_rsi(rising, 2)[-1] == pytest.approx(100.0)
    assert U.wilder_rsi(falling, 2)[-1] == pytest.approx(0.0)


def test_rsi_of_a_flat_series_is_neutral():
    """Sem ganhos nem perdas o RSI e' 50 e nao NaN -- um dia parado e'
    informacao, nao ausencia de dados."""
    assert U.wilder_rsi(np.full(50, 7.0), 2)[-1] == pytest.approx(50.0)


def test_rsi_needs_more_bars_than_its_period():
    assert np.isnan(U.wilder_rsi(np.array([1.0, 2.0]), 2)).all()


# --------------------------------------------------------------------------
# 3. A saida
# --------------------------------------------------------------------------

def test_next_exit_is_strictly_in_the_future():
    flags = np.array([False, True, False, False, True, False])
    assert list(U.next_exit_index(flags)) == [1, 4, 4, 4, 6, 6]


def test_next_exit_returns_the_sentinel_when_there_is_none():
    flags = np.zeros(5, dtype=bool)
    assert list(U.next_exit_index(flags)) == [5, 5, 5, 5, 5]


def test_max_hold_caps_the_holding_period():
    bars = U.indicators(_bars(_walk(seed=11)))
    for cap in (3, 10):
        trades = U.build_trades(bars, U.entry_mask(bars), cap, "fecho")
        assert trades["hold"].max() <= cap


def test_no_trade_has_zero_or_negative_duration():
    bars = U.indicators(_bars(_walk(seed=13)))
    trades = U.build_trades(bars, U.entry_mask(bars), U.MAX_HOLD, "fecho")
    assert not trades.empty
    assert (trades["hold"] > 0).all()


# --------------------------------------------------------------------------
# 4. A execucao
# --------------------------------------------------------------------------

def test_open_execution_uses_the_next_open():
    """O RSI(2) e a media de 5 dias so' se conhecem DEPOIS do fecho. Comprar a
    esse fecho e' uma antevisao de uma barra; a abertura seguinte e' a unica
    versao executavel como a regra esta' escrita."""
    close = _walk(n=1200, seed=17)
    opens = close * 1.05                       # abertura sempre 5% acima
    bars = U.indicators(_bars(close, open_=opens))
    mask = U.entry_mask(bars)
    at_close = U.build_trades(bars, mask, U.MAX_HOLD, "fecho")
    at_open = U.build_trades(bars, mask, U.MAX_HOLD, "abertura")
    assert not at_open.empty
    # Com a abertura a ser sempre o fecho vezes 1.05, o retorno medido na
    # abertura e' exactamente o do fecho deslocado um dia -- nunca identico.
    common = min(len(at_close), len(at_open))
    assert not np.allclose(at_close["ret"].to_numpy()[:common],
                           at_open["ret"].to_numpy()[:common])


def test_entry_conditions_compose():
    bars = U.indicators(_bars(_walk(seed=19)))
    full = U.entry_mask(bars).sum()
    dip = U.entry_mask(bars, use_rsi=False).sum()
    trend = U.entry_mask(bars, use_rsi=False, use_fast=False).sum()
    assert full <= dip <= trend
    assert full > 0


def test_a_mask_never_fires_before_the_slow_average_exists():
    bars = U.indicators(_bars(_walk(seed=23)), slow=200)
    assert np.flatnonzero(U.entry_mask(bars)).min() >= 199


# --------------------------------------------------------------------------
# 5. Leitura
# --------------------------------------------------------------------------

def test_breakeven_is_loss_over_win_plus_loss():
    """A aritmetica que a alegacao publica omite: 71% de acerto com ganho de
    1.22% contra perda de 2.10% deixa 8pp de margem, nao 71%."""
    trades = pd.DataFrame({"ret": [0.0122] * 71 + [-0.0210] * 29,
                           "hold": [4] * 100})
    result = U.summary(trades, cost_bps=0.0)
    assert result["acerto"] == pytest.approx(0.71)
    assert result["ponto_morto"] == pytest.approx(0.0210 / (0.0210 + 0.0122))
    assert result["media"] > 0


def test_costs_lower_the_mean_by_exactly_the_cost():
    trades = pd.DataFrame({"ret": [0.01, -0.02, 0.03], "hold": [2, 3, 4]})
    free = U.summary(trades, cost_bps=0.0)["media"]
    charged = U.summary(trades, cost_bps=10.0)["media"]
    assert free - charged == pytest.approx(10.0 / 1e4)


def test_matched_hold_baseline_uses_the_same_lengths():
    """A linha de base mais apertada: a MESMA accao, janelas do MESMO
    comprimento. Se os comprimentos nao coincidirem, compara-se outra coisa."""
    bars = _bars(_walk(n=800, seed=29))
    holds = np.array([1, 5, 10, 3])
    rng = np.random.default_rng(2)
    values = U.buy_and_hold(bars, holds, rng)
    assert values.size == holds.size
    assert np.isfinite(values).all()


# --------------------------------------------------------------------------
# 6. O bootstrap por data
# --------------------------------------------------------------------------

def test_day_resampling_is_exact_as_sums_over_counts():
    """A optimizacao: a media de uma uniao de grupos e'
    soma(somas)/soma(contagens). Se nao fosse exacto, o intervalo publicado
    estaria errado por uma razao invisivel."""
    groups = [np.array([1.0, 2.0, 3.0]), np.array([10.0]), np.array([4.0, 6.0])]
    picked = [0, 2, 2, 1]
    direct = np.concatenate([groups[i] for i in picked]).mean()
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups], dtype=float)
    assert sums[picked].sum() / counts[picked].sum() == pytest.approx(direct)


def test_date_bootstrap_is_wider_than_resampling_trades():
    """Duzentas accoes entram no mesmo recuo do mercado no mesmo dia. Tratar
    esses negocios como independentes estreita o intervalo -- o erro das
    janelas sobrepostas da ronda 7, por outro caminho."""
    rng = np.random.default_rng(31)
    rows = []
    days = pd.date_range("2000-01-03", periods=40, freq="B", tz="UTC")
    for day in days:
        shared = rng.normal(0, 0.03)           # o dia inteiro anda junto
        for _ in range(50):
            rows.append({"ts": day, "hold": 4,
                         "ret": shared + rng.normal(0, 0.002)})
    trades = pd.DataFrame(rows)

    _, low, high = U.date_bootstrap(trades, cost_bps=0.0, draws=2000)
    values = trades["ret"].to_numpy()
    naive = np.array([rng.choice(values, len(values)).mean() for _ in range(2000)])
    naive_width = np.quantile(naive, 0.975) - np.quantile(naive, 0.025)
    assert (high - low) > 3 * naive_width


def test_by_decade_splits_on_the_decade():
    trades = pd.DataFrame({
        "ts": pd.to_datetime(["1975-01-02", "1979-06-01", "1985-01-02"], utc=True),
        "ret": [0.01, 0.02, 0.03], "hold": [3, 3, 3],
        "ticker": ["A", "B", "A"]})
    table = U.by_decade(trades, cost_bps=0.0).set_index("decada")
    assert list(table.index) == [1970, 1980]
    assert table.loc[1970, "n"] == 2
    assert table.loc[1980, "accoes"] == 1
