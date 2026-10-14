"""Testes do bracket de pivots (experiencia T).

UM TESTE CARREGA A EXPERIENCIA TODA: `test_random_entries_reproduce_the_theoretical_rate`.

Numa caminhada aleatoria sem deriva, a probabilidade de tocar +A antes de -S e'
S/(S+A) pelo teorema da paragem opcional -- 83.33% com o bracket 25/5. Se o
simulador nao reproduzir esse numero sobre barras sinteticas construidas a
partir de uma caminhada verdadeira, entao NADA do que ele disser sobre dados
reais significa alguma coisa: nao se saberia distinguir um sinal de um erro de
contabilidade. Foi esse teste, aplicado aos dados reais, que mostrou que a taxa
de acerto alegada de 89-91% e' o que uma entrada ao acaso produz quando se
deixa a barra de entrada arrumar o negocio.

O segundo que importa e' `test_resolving_inside_the_entry_bar_inflates`: a
diferenca entre resolver a partir da barra de entrada e a partir da seguinte
tem de ser GRANDE e tem de ir sempre no mesmo sentido. Se o teste nao a
apanhasse, o resultado principal desta ronda passaria por sinal.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research import pivot_bracket as T


# --------------------------------------------------------------------------
# Barras sinteticas a partir de uma caminhada aleatoria verdadeira
# --------------------------------------------------------------------------

def _walk_bars(sessions=20, bars=600, sub=60, step=0.516, seed=5, drift=0.0):
    """OHLC agregado de uma caminhada por dentro da barra.

    O percurso INTRABARRA tem de ser real e nao inventado a partir do OHLC,
    senao o teste de calibracao estaria a verificar o simulador contra as suas
    proprias hipoteses em vez de contra a teoria."""
    rng = np.random.default_rng(seed)
    rows = []
    for session in range(sessions):
        ticks = 100.0 + np.cumsum(rng.normal(drift, step, bars * sub)) + 1000.0
        grid = ticks.reshape(bars, sub)
        stamps = pd.date_range(f"2026-01-{session + 1:02d} 14:30", periods=bars,
                               freq="1min", tz="UTC")
        rows.append(pd.DataFrame({
            "ts": stamps, "open": grid[:, 0], "high": grid.max(1),
            "low": grid.min(1), "close": grid[:, -1], "volume": 1,
        }))
    return T.add_session(pd.concat(rows).reset_index(drop=True))


# --------------------------------------------------------------------------
# 1. Calibracao -- o teste que sustenta a experiencia
# --------------------------------------------------------------------------

def test_random_entries_reproduce_the_theoretical_rate():
    """S/(S+A) = 83.33%, e o simulador tem de la' chegar sozinho."""
    frame = _walk_bars()
    rng = np.random.default_rng(3)
    picks = T.random_entries(frame, 4000, rng, max_bars=400)
    out = T.resolve(frame, picks["entry_index"].to_numpy(),
                    picks["entry_price"].to_numpy(float),
                    picks["side"].to_numpy(int), stop=25, target=5,
                    max_bars=400, model="honesta", start_offset=1)
    resolved = out[out["outcome"] != "tempo"]
    rate = (resolved["outcome"] == "alvo").mean()
    assert rate == pytest.approx(25 / 30, abs=0.02), rate


def test_the_breakeven_equals_the_random_walk_rate():
    """O ponto morto sem custos e' O MESMO numero que a moeda ao ar. E' a
    razao pela qual "83% de ponto morto" nao e' fasquia nenhuma."""
    assert T.breakeven(25, 5, cost=0.0) == pytest.approx(25 / 30)
    assert T.expectancy(25 / 30, 25, 5, cost=0.0) == pytest.approx(0.0, abs=1e-12)


def test_cost_moves_the_breakeven_by_three_points_per_point():
    """Com um alvo de 5 pontos, cada ponto de custo vale 3.33pp de taxa de
    acerto. E' o que poe o ponto morto de um CFD acima da taxa alegada."""
    assert T.breakeven(25, 5, 1.0) - T.breakeven(25, 5, 0.0) == pytest.approx(1 / 30)
    assert T.breakeven(25, 5, 2.5) == pytest.approx(0.91666, abs=1e-4)


# --------------------------------------------------------------------------
# 2. A barra de entrada
# --------------------------------------------------------------------------

def test_resolving_inside_the_entry_bar_inflates():
    """Deixar o maximo e o minimo da barra de ENTRADA arrumarem um negocio que
    entrou ao FECHO dessa barra e' ler o passado, e tem de inflacionar sempre
    no mesmo sentido."""
    frame = _walk_bars(seed=9)
    rng = np.random.default_rng(4)
    picks = T.random_entries(frame, 3000, rng, max_bars=400)
    rates = {}
    for offset in (0, 1):
        out = T.resolve(frame, picks["entry_index"].to_numpy(),
                        picks["entry_price"].to_numpy(float),
                        picks["side"].to_numpy(int), 25, 5, 400,
                        "honesta", start_offset=offset)
        rates[offset] = float((out["outcome"] == "alvo").mean())
    assert rates[0] > rates[1] + 0.02, rates
    assert rates[1] == pytest.approx(25 / 30, abs=0.03)


# --------------------------------------------------------------------------
# 3. O motor de pivots
# --------------------------------------------------------------------------

def test_pivot_is_the_strict_extreme_of_its_window():
    values = np.array([1, 2, 3, 9, 3, 2, 1, 2, 3, 4], dtype=float)
    flags = T.pivot_flags(values, left=3, right=3, high=True)
    assert list(np.flatnonzero(flags)) == [3]


def test_a_flat_plateau_produces_no_pivots():
    """Com igualdade admitida, dez barras iguais dariam dez "pivots" no mesmo
    preco e a contagem de niveis passaria a medir barras paradas."""
    values = np.full(20, 5.0)
    assert not T.pivot_flags(values, 3, 3, high=True).any()
    assert not T.pivot_flags(values, 3, 3, high=False).any()


def test_low_pivots_are_strict_minima():
    values = np.array([9, 8, 7, 1, 7, 8, 9, 8, 7, 6], dtype=float)
    assert list(np.flatnonzero(T.pivot_flags(values, 3, 3, high=False))) == [3]


def test_a_level_is_not_tradeable_before_it_is_confirmed():
    """Um pivot com 10 barras a direita so' EXISTE 10 barras depois. Libertar o
    nivel na barra do pivot e' ver dez minutos do futuro."""
    frame = _walk_bars(sessions=2, bars=300, seed=7)
    honest = T.levels(frame, 10, 10, lookahead=False)
    cheating = T.levels(frame, 10, 10, lookahead=True)
    assert (honest["ready_index"] - honest["pivot_index"] == 10).all()
    assert (cheating["ready_index"] == cheating["pivot_index"]).all()


def test_entries_happen_strictly_after_confirmation():
    frame = _walk_bars(sessions=3, bars=400, seed=11)
    table = T.levels(frame, 10, 10)
    found = T.entries(frame, table, rule="fade", lag=0)
    assert not found.empty
    merged = found.join(table.set_index("level"), on="entry_price", rsuffix="_lvl")
    assert (found["entry_index"].to_numpy() > 0).all()
    # Nenhuma entrada pode cair antes da barra em que o nivel ficou pronto.
    assert (found["wait"].to_numpy() >= 0).all()


def test_each_level_trades_at_most_once():
    frame = _walk_bars(sessions=3, bars=400, seed=13)
    table = T.levels(frame, 10, 10)
    found = T.entries(frame, table, rule="fade", lag=0)
    assert len(found) <= len(table)


# --------------------------------------------------------------------------
# 4. Os modelos de preenchimento
# --------------------------------------------------------------------------

def _one_trade_frame(bar):
    """Uma barra de entrada neutra seguida da barra que decide."""
    rows = [{"ts": pd.Timestamp("2026-01-02 14:30", tz="UTC"), "open": 100.0,
             "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1}, bar]
    return T.add_session(pd.DataFrame(rows))


def test_the_three_models_are_ordered_on_an_ambiguous_bar():
    """Uma barra que contem o alvo E o stop: optimista arruma o alvo, honesta
    arruma o stop. Sao 30 pontos de diferenca no mesmo negocio."""
    wide = {"ts": pd.Timestamp("2026-01-02 14:31", tz="UTC"), "open": 100.0,
            "high": 106.0, "low": 74.0, "close": 105.0, "volume": 1}
    frame = _one_trade_frame(wide)
    args = dict(entry_index=np.array([0]), entry_price=np.array([100.0]),
                side=np.array([1]), stop=25, target=5, max_bars=5)
    optimistic = T.resolve(frame, model="optimista", **args)
    honest = T.resolve(frame, model="honesta", **args)
    assert bool(optimistic["ambiguous"].iloc[0])
    assert optimistic["gross"].iloc[0] == 5.0
    assert honest["gross"].iloc[0] == -25.0
    assert optimistic["gross"].iloc[0] - honest["gross"].iloc[0] == 30.0


def test_the_direction_model_uses_the_bar_sign():
    """Barra que fecha acima da abertura: percurso abertura->minimo->maximo,
    portanto o minimo veio primeiro e uma COMPRA leva com o stop."""
    up = {"ts": pd.Timestamp("2026-01-02 14:31", tz="UTC"), "open": 100.0,
          "high": 106.0, "low": 74.0, "close": 105.0, "volume": 1}
    down = dict(up, open=105.0, close=100.0)
    args = dict(entry_index=np.array([0]), entry_price=np.array([100.0]),
                stop=25, target=5, max_bars=5, model="direccao")
    long_up = T.resolve(_one_trade_frame(up), side=np.array([1]), **args)
    long_down = T.resolve(_one_trade_frame(down), side=np.array([1]), **args)
    short_up = T.resolve(_one_trade_frame(up), side=np.array([-1]), **args)
    assert long_up["gross"].iloc[0] == -25.0        # minimo primeiro
    assert long_down["gross"].iloc[0] == 5.0        # maximo primeiro
    assert short_up["gross"].iloc[0] == 5.0         # minimo primeiro favorece a venda


def test_an_unambiguous_bar_is_identical_under_every_model():
    clean = {"ts": pd.Timestamp("2026-01-02 14:31", tz="UTC"), "open": 100.0,
             "high": 106.0, "low": 99.0, "close": 105.0, "volume": 1}
    frame = _one_trade_frame(clean)
    args = dict(entry_index=np.array([0]), entry_price=np.array([100.0]),
                side=np.array([1]), stop=25, target=5, max_bars=5)
    values = {m: T.resolve(frame, model=m, **args)["gross"].iloc[0]
              for m in ("optimista", "honesta", "direccao")}
    assert set(values.values()) == {5.0}


# --------------------------------------------------------------------------
# 5. Sessoes
# --------------------------------------------------------------------------

def test_a_trade_never_resolves_on_the_next_session():
    """Um stop nao e' tocado durante uma pausa, e o salto entre sessoes nao e'
    um movimento de um minuto. E' a mesma disciplina da ronda 10."""
    rows = [
        {"ts": pd.Timestamp("2026-01-02 20:00", tz="UTC"), "open": 100.0,
         "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1},
        {"ts": pd.Timestamp("2026-01-05 14:30", tz="UTC"), "open": 200.0,
         "high": 300.0, "low": 50.0, "close": 250.0, "volume": 1},
    ]
    frame = T.add_session(pd.DataFrame(rows))
    assert frame["session"].nunique() == 2
    out = T.resolve(frame, np.array([0]), np.array([100.0]), np.array([1]),
                    stop=25, target=5, max_bars=5, model="honesta",
                    start_offset=1)
    # Este teste apanhou um bug a serio: a versao anterior ia buscar o preco de
    # saida a' PRIMEIRA barra da janela quando nao havia barras validas, e essa
    # barra era ja' da sessao seguinte. O negocio arrecadava +150 pontos de
    # salto overnight. Nos dados reais eram 5 negocios e valiam +0.06 pontos
    # por negocio -- a favor da estrategia, que e' sempre a direccao perigosa.
    assert out["outcome"].iloc[0] == "sem_barras"
    assert out["gross"].iloc[0] == pytest.approx(0.0)


def test_entry_at_close_uses_the_close_and_not_the_level():
    frame = _walk_bars(sessions=2, bars=400, seed=17)
    at_level = T.trades(frame, entry_at="nivel", start_offset=1)
    at_close = T.trades(frame, entry_at="fecho", start_offset=1)
    assert len(at_level) == len(at_close)
    assert not np.allclose(at_level["gross"], at_close["gross"])


# --------------------------------------------------------------------------
# 6. Consistencia interna da alegacao
# --------------------------------------------------------------------------

def test_max_loss_streak_counts_the_longest_run():
    assert T.max_loss_streak(np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)) == 3
    assert T.max_loss_streak(np.ones(10, dtype=bool)) == 0
    assert T.max_loss_streak(np.zeros(4, dtype=bool)) == 4


def test_five_losses_in_a_row_contradict_a_ninety_percent_rate():
    """O teste que mostra que a alegacao se contradiz a si propria sem precisar
    de dados nenhuns."""
    result = T.streak_consistency(300, 0.90, observed=5, draws=4000)
    assert result["expected_streak"] <= 3
    assert result["p_at_least"] < 0.01
    assert result["implied_win_rate"] < 0.75


def test_a_consistent_claim_is_not_flagged():
    """O teste anterior nao serve de prova nenhuma se a funcao acusar tudo. Uma
    sequencia de 2 e' precisamente o que 90% produz."""
    result = T.streak_consistency(300, 0.90, observed=2, draws=4000)
    assert result["p_at_least"] > 0.20
    assert result["implied_win_rate"] == pytest.approx(0.90, abs=0.07)


# --------------------------------------------------------------------------
# 7. Bootstrap por sessao
# --------------------------------------------------------------------------

def test_session_bootstrap_is_wider_than_resampling_trades():
    """Negocios do mesmo dia partilham regime. Reamostra-los como se fossem
    independentes estreitaria o intervalo -- o erro das janelas sobrepostas da
    ronda 7, outra vez."""
    rng = np.random.default_rng(23)
    rows = []
    for day in range(20):
        level = rng.normal(0, 2.0)                 # o dia inteiro anda junto
        for _ in range(60):
            rows.append({"session": day, "gross": level + rng.normal(0, 0.3)})
    result = pd.DataFrame(rows)

    _, low, high = T.session_bootstrap(result, "gross", draws=2000)
    values = result["gross"].to_numpy()
    naive = np.array([rng.choice(values, len(values)).mean() for _ in range(2000)])
    naive_width = np.quantile(naive, 0.975) - np.quantile(naive, 0.025)
    assert (high - low) > 3 * naive_width


def test_paired_random_entries_keep_the_time_of_day():
    """O placebo exigente: os pivots concentram-se nas horas volateis, e uma
    linha de base uniforme dar-lhes-ia uma vantagem de horario disfarcada de
    vantagem de sinal."""
    frame = _walk_bars(sessions=10, bars=600, seed=29)
    like = pd.DataFrame({"minute_of_session": [10, 10, 50, 120, 200]})
    picks = T.random_entries(frame, 5, np.random.default_rng(2), max_bars=100,
                             like=like)
    got = frame["minute_of_session"].to_numpy()[picks["entry_index"].to_numpy()]
    assert sorted(got) == sorted(like["minute_of_session"])
