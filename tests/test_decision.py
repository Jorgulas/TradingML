"""Testes da camada de decisao (painel, sinais e tabela de verdade).

O que estes testes protegem, por ordem de gravidade se falharem:

  1. NENHUM VALOR FUTURO ENTRA NUMA DECISAO. E' a unica classe de erro que
     produz resultados espectaculares e completamente falsos, e a mais dificil
     de ver a olho. Aqui verifica-se explicitamente que os cortes do regime
     dependem so' do treino, que os padroes usados sao anteriores ao dia, e que
     o corte cronologico deixa embargo entre as particoes.

  2. O ENCOLHIMENTO FAZ O QUE DIZ. Uma celula com muito suporte tem de falar
     por si; uma celula rara tem de dizer o mesmo que a votacao simples diria.
     Se esta mecanica se partir, a tabela passa a seguir celulas de tres
     observacoes e ninguem repara ate' ao resultado sair bom de mais.

  3. O BOOTSTRAP DE BLOCOS ALARGA O INTERVALO em dados dependentes. Foi o erro
     que apanhei a meio deste trabalho -- com reamostragem simples, um
     resultado que nao era significativo parecia se-lo.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.decision import panel as P
from src.decision import signals as S
from src.decision import truth_table as T


# --------------------------------------------------------------------------
# Utilitarios
# --------------------------------------------------------------------------

def _bars(n=200, seed=3, start="2025-01-02", drift=0.0):
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(drift, 0.8, n))
    index = pd.date_range(start, periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.01, "low": closes * 0.99,
         "close": closes, "volume": rng.integers(1_000, 50_000, n)},
        index=index,
    )


def _panel(n_tickers=12, n_days=120, seed=5):
    """Painel sintetico com as colunas que a tabela precisa."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-02", periods=n_days, freq="B", tz="UTC")
    rows = []
    for i in range(n_tickers):
        for day in days:
            rows.append({
                "ticker": f"T{i:02d}", "ts": day,
                "forward_return": float(rng.normal(0, 0.02)),
                "label": float(rng.integers(0, 3)),
                "mkt_vol_20": float(abs(rng.normal(0.01, 0.003))),
                "abs_return": float(abs(rng.normal(0, 0.02))),
            })
    frame = pd.DataFrame(rows)
    for name, values in S.SIGNAL_VALUES.items():
        frame[name] = rng.choice(values, len(frame))
    return frame


# --------------------------------------------------------------------------
# 1. O alvo
# --------------------------------------------------------------------------

def test_market_neutral_target_removes_a_common_move():
    """Se a accao e o indice fizerem exactamente o mesmo, o retorno em excesso
    e' zero. E' a propriedade toda do alvo: sem ela mede-se a deriva do mercado
    e nao competencia nenhuma."""
    bars = _bars(120)
    plain = P.label_horizon(bars, bars, 5, market_neutral=False)["forward_return"]
    neutral = P.label_horizon(bars, bars, 5, market_neutral=True)["forward_return"]
    assert plain.abs().max() > 0.001            # ha' mesmo movimento para remover
    assert neutral.dropna().abs().max() < 1e-12


def test_band_grows_with_the_horizon():
    """A banda de indiferenca escala com raiz do horizonte. Se fosse fixa, o
    horizonte de 5 dias daria quase so' SOBE/DESCE e ESPERAR desapareceria."""
    bars = _bars(200)
    short = P.label_horizon(bars, None, 1, market_neutral=False)["band"].dropna()
    long = P.label_horizon(bars, None, 5, market_neutral=False)["band"].dropna()
    ratio = (long.mean() / short.mean())
    assert 2.0 < ratio < 2.6                    # sqrt(5) = 2.236


def test_label_has_no_value_at_the_tail():
    """As ultimas `horizon` barras nao tem futuro conhecido; deixa-las com
    rotulo seria inventar dados."""
    bars = _bars(100)
    labels = P.label_horizon(bars, None, 5, market_neutral=False)["label"]
    assert labels.iloc[-5:].isna().all()


# --------------------------------------------------------------------------
# 2. Features transversais
# --------------------------------------------------------------------------

def test_cross_sectional_rank_ignores_a_market_wide_move():
    """Somar a mesma quantidade a todas as accoes de um dia nao muda a ordem --
    e e' por isso que estas features sao neutras ao mercado por construcao."""
    frame = pd.DataFrame({
        "ticker": ["A", "B", "C"] * 2,
        "ts": [pd.Timestamp("2025-01-02", tz="UTC")] * 3
              + [pd.Timestamp("2025-01-03", tz="UTC")] * 3,
        "ret_5": [0.01, 0.02, 0.03, 0.51, 0.52, 0.53],
    })
    ranked = P.add_cross_sectional(frame)
    first = ranked[ranked["ts"] == frame["ts"].iloc[0]]["xs_ret_5"].values
    second = ranked[ranked["ts"] == frame["ts"].iloc[-1]]["xs_ret_5"].values
    assert np.allclose(first, second)


def test_cross_sectional_rank_never_crosses_days():
    """O percentil de hoje nao pode depender do que acontece amanha."""
    frame = pd.DataFrame({
        "ticker": ["A", "B"] * 2,
        "ts": [pd.Timestamp("2025-01-02", tz="UTC")] * 2
              + [pd.Timestamp("2025-01-03", tz="UTC")] * 2,
        "ret_5": [0.01, 0.02, 99.0, -99.0],
    })
    ranked = P.add_cross_sectional(frame)
    assert ranked["xs_ret_5"].iloc[0] == pytest.approx(0.5)
    assert ranked["xs_ret_5"].iloc[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 3. Corte cronologico
# --------------------------------------------------------------------------

def test_split_is_chronological_and_leaves_an_embargo():
    """Cortar por ticker deixaria o treino de um a sobrepor-se ao teste de
    outro no mesmo dia. Como se movem juntos, isso e' fuga disfarcada."""
    frame = _panel(n_tickers=4, n_days=100)
    train, validation, test = P.split(frame, horizon=5)
    assert train["ts"].max() < validation["ts"].min()
    assert validation["ts"].max() < test["ts"].min()
    gap = (test["ts"].min() - validation["ts"].max()).days
    assert gap >= 5


def test_split_counts_days_not_rows():
    """A amostra efectiva sao DIAS: 12 tickers no mesmo dia sao praticamente
    uma observacao, nao doze."""
    frame = _panel(n_tickers=12, n_days=60)
    assert len(frame) == 720
    assert P.effective_days(frame) == 60


# --------------------------------------------------------------------------
# 4. Sinais
# --------------------------------------------------------------------------

def test_regime_thresholds_come_only_from_the_fit_rows():
    """O teste decisivo de fuga neste modulo: mudar radicalmente os valores das
    linhas FORA do treino nao pode alterar o veredicto das linhas de treino."""
    frame = pd.DataFrame({
        "mkt_vol_20": np.concatenate([np.linspace(0.005, 0.02, 200),
                                      np.full(200, 0.001)]),
    })
    mask = pd.Series([True] * 200 + [False] * 200)
    first = S.regime_signal(frame, mask)

    altered = frame.copy()
    altered.loc[200:, "mkt_vol_20"] = 99.0
    second = S.regime_signal(altered, mask)

    assert (first[:200] == second[:200]).all()


def test_regime_produces_all_three_values():
    frame = pd.DataFrame({"mkt_vol_20": np.linspace(0.005, 0.05, 400)})
    mask = pd.Series([True] * 400)
    assert set(S.regime_signal(frame, mask).unique()) == {"CALMO", "NORMAL", "AGITADO"}


def test_tercile_leaves_thin_days_neutral():
    """Um dia com cinco accoes nao se corta em tercis com significado. Melhor
    dizer NEUTRO do que inventar um veredicto."""
    frame = pd.DataFrame({
        "ts": [pd.Timestamp("2025-01-02", tz="UTC")] * 5,
        "ticker": list("ABCDE"),
    })
    verdict = S._cross_sectional_tercile(frame, np.arange(5.0))
    assert (verdict == "NEUTRO").all()


def test_tercile_splits_a_full_day_in_three():
    n = 60
    frame = pd.DataFrame({
        "ts": [pd.Timestamp("2025-01-02", tz="UTC")] * n,
        "ticker": [f"T{i}" for i in range(n)],
    })
    verdict = S._cross_sectional_tercile(frame, np.arange(float(n)))
    counts = verdict.value_counts()
    assert counts["DESCE"] == counts["SOBE"] == 20
    assert verdict.iloc[0] == "DESCE" and verdict.iloc[-1] == "SOBE"


def test_pattern_signal_never_uses_a_future_pattern(conn):
    """Datar pela confirmacao e nao pelo fim da formacao, e nunca ir buscar um
    padrao posterior ao dia da decisao."""
    conn.execute(
        "INSERT INTO detected_patterns (ticker, timeframe, pattern_type, start_ts, "
        "end_ts, confirmed_ts, start_idx, end_idx, quality, detected_at) "
        "VALUES ('AAPL','1h','DOUBLE_TOP','2025-03-10T13:30:00+00:00',"
        "'2025-03-12T13:30:00+00:00','2025-03-20T13:30:00+00:00',0,10,0.9,'2025-03-20')"
    )
    conn.commit()
    frame = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "ts": [pd.Timestamp("2025-03-15", tz="UTC"), pd.Timestamp("2025-03-22", tz="UTC")],
    })
    verdict = S.pattern_signal(conn, frame)
    assert verdict.iloc[0] == "NENHUM"    # o padrao ainda nao estava confirmado
    assert verdict.iloc[1] == "BAIXA"     # DOUBLE_TOP e' bearish


def test_pattern_signal_forgets_old_patterns(conn):
    """Um triangulo de ha' dois meses nao informa a decisao de hoje."""
    conn.execute(
        "INSERT INTO detected_patterns (ticker, timeframe, pattern_type, start_ts, "
        "end_ts, confirmed_ts, start_idx, end_idx, quality, detected_at) "
        "VALUES ('AAPL','1h','DOUBLE_BOTTOM','2025-01-02T13:30:00+00:00',"
        "'2025-01-05T13:30:00+00:00','2025-01-06T13:30:00+00:00',0,10,0.9,'2025-01-06')"
    )
    conn.commit()
    frame = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "ts": [pd.Timestamp("2025-01-08", tz="UTC"), pd.Timestamp("2025-03-01", tz="UTC")],
    })
    verdict = S.pattern_signal(conn, frame)
    assert verdict.iloc[0] == "ALTA"
    assert verdict.iloc[1] == "NENHUM"


# --------------------------------------------------------------------------
# 5. A tabela
# --------------------------------------------------------------------------

def _table_frame(cell_days: int, cell_mean: float):
    """Painel onde uma celula especifica tem `cell_days` dias e uma media
    conhecida, e o resto do universo e' plano."""
    days = pd.date_range("2025-01-02", periods=300, freq="B", tz="UTC")
    rows = []
    for i, day in enumerate(days):
        special = i < cell_days
        rows.append({
            "ticker": "X", "ts": day, "label": 2.0,
            "forward_return": cell_mean if special else 0.0,
            "curto": "SOBE" if special else "NEUTRO",
            "rapido": "SOBE" if special else "NEUTRO",
            "padrao": "ALTA" if special else "NENHUM",
            "regime": "CALMO" if special else "NORMAL",
            "alarme": "SIM" if special else "NAO",
        })
    return pd.DataFrame(rows)


def test_shrinkage_trusts_a_well_supported_cell():
    frame = _table_frame(cell_days=250, cell_mean=0.05)
    row = T.build_table(frame).iloc[0]
    assert row["weight"] > 0.85
    assert abs(row["estimate"] - row["cell_mean"]) < abs(row["cell_mean"]) * 0.2


def test_shrinkage_pulls_a_rare_cell_towards_the_vote():
    """Tres dias nao chegam para acreditar em nada. A estimativa usada tem de
    ficar muito mais perto do aditivo do que da media crua."""
    frame = _table_frame(cell_days=3, cell_mean=0.5)
    table = T.build_table(frame)
    row = table[table["curto"] == "SOBE"].iloc[0]
    assert row["weight"] < 0.10
    assert abs(row["estimate"] - row["additive"]) < abs(row["estimate"] - row["cell_mean"])


def test_shrinkage_weight_is_monotone_in_support():
    weights = [T.build_table(_table_frame(d, 0.05))
               .query("curto == 'SOBE'")["weight"].iloc[0]
               for d in (5, 40, 200)]
    assert weights[0] < weights[1] < weights[2]


def test_unseen_combination_becomes_wait():
    """"Nunca vi isto acontecer" tem uma resposta certa, e e' nao fazer nada."""
    frame = _table_frame(cell_days=50, cell_mean=0.05)
    table = T.verdicts(T.build_table(frame), 0.001)
    unseen = pd.DataFrame([{"curto": "DESCE", "rapido": "DESCE", "padrao": "BAIXA",
                            "regime": "AGITADO", "alarme": "SIM"}])
    assert T.apply_table(table, unseen).iloc[0] == "ESPERAR"


def test_threshold_semantics():
    table = pd.DataFrame({"estimate": [0.02, 0.0, -0.02]})
    actions = T.verdicts(table, 0.01)["accao"].tolist()
    assert actions == ["COMPRAR", "ESPERAR", "VENDER"]


def test_a_day_with_no_action_returns_zero():
    """Nao agir e' um resultado e conta como dia. Omitir esses dias
    inflacionaria a media do esquema."""
    frame = _panel(n_tickers=5, n_days=10)
    action = pd.Series("ESPERAR", index=frame.index)
    daily = T.daily_result(frame, action)
    assert len(daily) == 10
    assert (daily == 0.0).all()


def test_daily_result_is_long_minus_short():
    frame = pd.DataFrame({
        "ts": [pd.Timestamp("2025-01-02", tz="UTC")] * 4,
        "ticker": list("ABCD"),
        "forward_return": [0.10, 0.06, -0.04, -0.02],
    })
    action = pd.Series(["COMPRAR", "COMPRAR", "VENDER", "VENDER"])
    assert T.daily_result(frame, action).iloc[0] == pytest.approx(0.08 + 0.03)


def test_turnover_is_zero_when_the_selection_never_changes():
    days = pd.date_range("2025-01-02", periods=40, freq="B", tz="UTC")
    frame = pd.DataFrame([{"ts": d, "ticker": t} for d in days for t in "AB"])
    action = pd.Series(["COMPRAR", "VENDER"] * len(days))
    turnover, cost = T.turnover_and_cost(frame, action)
    assert turnover == pytest.approx(0.0)
    assert cost == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 6. Estatistica
# --------------------------------------------------------------------------

def test_block_bootstrap_is_wider_than_naive_resampling():
    """O erro que este mecanismo corrige: com janelas sobrepostas, reamostrar
    pontos soltos assume independencia que nao existe e estreita o intervalo
    por ~sqrt(horizonte). Um resultado nulo passaria a significativo."""
    from src.decision.short_term import block_bootstrap_mean

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 500)
    # Media movel de 5: e' exactamente a estrutura de uma janela de 5 dias
    # medida todos os dias.
    smoothed = pd.Series(noise).rolling(5).mean().dropna().values

    naive_low, naive_high = block_bootstrap_mean(smoothed, block=1)
    block_low, block_high = block_bootstrap_mean(smoothed, block=5)
    assert (block_high - block_low) > (naive_high - naive_low) * 1.4


def test_block_bootstrap_needs_enough_points():
    from src.decision.short_term import block_bootstrap_mean
    low, high = block_bootstrap_mean(np.array([1.0, 2.0, 3.0]), block=5)
    assert np.isnan(low) and np.isnan(high)


def test_non_overlapping_takes_one_window_per_horizon():
    from src.decision.short_term import non_overlapping
    series = pd.Series(np.arange(20.0))
    assert list(non_overlapping(series, step=5)) == [0.0, 5.0, 10.0, 15.0]


# --------------------------------------------------------------------------
# 7. Coerencia da configuracao
# --------------------------------------------------------------------------

def test_short_term_horizon_sits_between_the_two_extremes():
    """O pedido era um prazo curto que NAO fosse o de 1 minuto. Se alguem
    igualar este horizonte ao das arvores ou ao da carteira LONG, o degrau
    intermedio desaparece e este teste avisa."""
    assert config.TREES["horizon_bars"] == 1
    assert 1 < config.DECISION["horizon_days"] < config.HORIZON_PARAMS["LONG"]["predict_ahead_days"]


def test_every_signal_value_is_declared():
    """A tabela indexa por nomes de veredicto; um valor produzido mas nao
    declarado viraria uma celula fantasma."""
    for name, values in S.SIGNAL_VALUES.items():
        assert len(values) == len(set(values))
        assert all(isinstance(v, str) for v in values)
    assert set(S.SIGNAL_NAMES) == set(S.SIGNAL_VALUES)
