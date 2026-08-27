"""Testes das arvores de decisao (XGBoost, tres classes).

Protege sobretudo tres coisas:
  - o rotulo ESPERAR e' uma decisao a serio e nao um caixote de restos;
  - nenhuma feature olha para o futuro;
  - o split corta pelo TEMPO GLOBAL, senao o teste de um ticker sobrepoe-se
    ao treino de outro no mesmo minuto -- fuga entre accoes correlacionadas.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.trees import features as F
from src.trees import train


def _bars(n=300, seed=11, drift=0.0):
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(drift, 0.15, n))
    index = pd.date_range("2026-08-10 13:30", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.001, "low": closes * 0.999,
         "close": closes, "volume": rng.integers(1_000, 20_000, n)},
        index=index,
    )


# --------------------------------------------------------------------------
# O rotulo de tres classes
# --------------------------------------------------------------------------

def test_three_classes_are_all_produced():
    labels = F.label_three_class(_bars(600))["label"].dropna()
    assert set(labels.unique()) == {F.DOWN, F.WAIT, F.UP}


def test_wait_is_a_substantial_class_not_a_leftover_bin():
    """Se ESPERAR fosse residual, nao seria uma decisao -- seria ruido."""
    labels = F.label_three_class(_bars(800))["label"].dropna()
    share = (labels == F.WAIT).mean()
    assert 0.15 < share < 0.75, f"ESPERAR ficou em {share:.1%}"


def test_wider_band_produces_more_wait():
    bars = _bars(600)
    narrow = F.label_three_class(bars, band_k=0.1)["label"].dropna()
    wide = F.label_three_class(bars, band_k=2.0)["label"].dropna()
    assert (wide == F.WAIT).mean() > (narrow == F.WAIT).mean()


def test_label_matches_the_sign_of_the_forward_move_outside_the_band():
    result = F.label_three_class(_bars(400)).dropna()
    acted = result[result["label"] != F.WAIT]
    up = acted[acted["label"] == F.UP]
    down = acted[acted["label"] == F.DOWN]
    assert (up["forward_return"] > 0).all()
    assert (down["forward_return"] < 0).all()


def test_moves_inside_the_band_are_labelled_wait():
    result = F.label_three_class(_bars(400)).dropna()
    waiting = result[result["label"] == F.WAIT]
    assert (waiting["forward_return"].abs() <= waiting["band"]).all()


# --------------------------------------------------------------------------
# Anti-leakage nas features
# --------------------------------------------------------------------------

def test_indicator_features_do_not_use_future_bars():
    """Mudar uma barra FUTURA nao pode alterar a linha de features de hoje."""
    bars = _bars(300)
    original = F.indicator_features(bars)

    mutated = bars.copy()
    mutated.iloc[250:, mutated.columns.get_loc("close")] *= 1.5
    after = F.indicator_features(mutated)

    pd.testing.assert_frame_equal(original.iloc[:250], after.iloc[:250])


def test_microstructure_features_do_not_use_future_bars():
    bars = _bars(300)
    original = F.microstructure_features(bars)

    mutated = bars.copy()
    mutated.iloc[200:, mutated.columns.get_loc("close")] *= 0.7
    after = F.microstructure_features(mutated)

    pd.testing.assert_frame_equal(original.iloc[:200], after.iloc[:200])


def test_market_context_features_do_not_use_future_bars():
    """As features de mercado foram acrescentadas depois dos testes originais e
    ficaram sem cobertura -- e sao elas que sustentam o melhor resultado do
    projecto. Se o indice do futuro vazasse por aqui, esse resultado nao
    valeria nada."""
    bars = _bars(300, seed=3)
    benchmark = _bars(300, seed=9)
    original = F.market_context_features(bars, benchmark)

    mutated = benchmark.copy()
    mutated.iloc[200:, mutated.columns.get_loc("close")] *= 2.0
    after = F.market_context_features(bars, mutated)

    pd.testing.assert_frame_equal(original.iloc[:200], after.iloc[:200])


def test_market_features_react_to_the_benchmark_at_all():
    """Contraprova: se nao reagissem a nada, o teste acima passava trivialmente."""
    bars = _bars(300, seed=3)
    benchmark = _bars(300, seed=9)
    original = F.market_context_features(bars, benchmark)
    mutated = benchmark.copy()
    mutated.iloc[100:, mutated.columns.get_loc("close")] *= 2.0
    after = F.market_context_features(bars, mutated)
    assert not original.iloc[150:200].equals(after.iloc[150:200])


def test_market_features_survive_a_missing_benchmark():
    result = F.market_context_features(_bars(120), pd.DataFrame())
    assert list(result.columns) == F.MARKET_FEATURES
    assert result.notna().all().all()


def test_label_does_use_the_future_which_is_the_point():
    """Contraprova do teste anterior: o ROTULO tem mesmo de mudar."""
    bars = _bars(300)
    original = F.label_three_class(bars)["label"]
    mutated = bars.copy()
    mutated.iloc[100:, mutated.columns.get_loc("close")] *= 1.4
    after = F.label_three_class(mutated)["label"]
    assert not original.iloc[95:105].equals(after.iloc[95:105])


# --------------------------------------------------------------------------
# Features do modelo B -- a forma da janela
# --------------------------------------------------------------------------

def test_sign_lags_encode_the_recent_movement_sequence():
    names = F.microstructure_feature_names(10)
    assert [f"sign_lag{i}" for i in range(1, 11)] == names[-10:]


def test_counts_of_up_and_down_sum_to_the_window():
    result = F.microstructure_features(_bars(200), window=10).dropna()
    flat = (result["n_up"] + result["n_down"]) > 10
    assert not flat.any()


def test_inertia_is_positive_while_rising_and_negative_while_falling():
    rising = pd.DataFrame({
        "open": np.arange(60.0) + 100, "high": np.arange(60.0) + 100.2,
        "low": np.arange(60.0) + 99.8, "close": np.arange(60.0) + 100,
        "volume": np.full(60, 5000),
    }, index=pd.date_range("2026-08-10 13:30", periods=60, freq="1min", tz="UTC"))
    result = F.microstructure_features(rising, window=10).dropna()
    assert (result["inertia"] > 0).all()

    falling = rising.copy()
    for column in ("open", "high", "low", "close"):
        falling[column] = falling[column].values[::-1]
    result = F.microstructure_features(falling, window=10).dropna()
    assert (result["inertia"] < 0).all()


def test_momentum_shift_is_the_difference_between_the_two_halves():
    result = F.microstructure_features(_bars(200), window=10).dropna()
    expected = result["second_half_mean"] - result["first_half_mean"]
    pd.testing.assert_series_equal(result["momentum_shift"], expected, check_names=False)


def test_position_in_range_stays_between_zero_and_one():
    result = F.microstructure_features(_bars(300), window=10).dropna()
    valid = result["position_in_range"].dropna()
    assert ((valid >= 0) & (valid <= 1)).all()


def test_the_two_feature_sets_are_genuinely_different():
    """Se partilhassem features, comparar os dois modelos nao diria nada."""
    assert not set(F.INDICATOR_FEATURES) & set(F.microstructure_feature_names())


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

def _frame(n_ts=100, tickers=("AAPL", "MSFT")):
    stamps = pd.date_range("2026-08-10 13:30", periods=n_ts, freq="1min", tz="UTC")
    rows = [{"ts": ts, "ticker": t, "label": float(i % 3)}
            for i, ts in enumerate(stamps) for t in tickers]
    return pd.DataFrame(rows)


def test_split_is_chronological_and_partitions_do_not_overlap():
    train_df, validation_df, test_df = train.split_by_time(_frame(), horizon=1)
    assert train_df["ts"].max() < validation_df["ts"].min()
    assert validation_df["ts"].max() < test_df["ts"].min()


def test_split_keeps_correlated_tickers_on_the_same_side_of_the_cut():
    """Cortar por indice dentro de cada ticker deixaria o mesmo minuto em
    treino para um ticker e em teste para outro."""
    train_df, _, test_df = train.split_by_time(_frame(), horizon=1)
    assert not set(train_df["ts"]) & set(test_df["ts"])


def test_embargo_removes_rows_at_the_boundary():
    frame = _frame(120)
    without = train.split_by_time(frame, horizon=0)
    with_embargo = train.split_by_time(frame, horizon=10)
    assert len(with_embargo[0]) < len(without[0])


# --------------------------------------------------------------------------
# Metrica accionavel
# --------------------------------------------------------------------------

def test_always_waiting_scores_no_actionable_precision():
    """O modelo inutil (responde sempre ESPERAR) tem de aparecer como inutil."""
    y_true = np.array([F.UP, F.DOWN, F.WAIT, F.UP])
    y_pred = np.full(4, F.WAIT)
    metrics = train.actionable_metrics(y_true, y_pred)
    assert metrics["action_rate"] == 0.0
    assert metrics["action_precision"] is None


def test_actionable_precision_counts_only_the_bars_where_it_decided():
    y_true = np.array([F.UP, F.DOWN, F.WAIT, F.UP])
    y_pred = np.array([F.UP, F.UP, F.WAIT, F.WAIT])   # decidiu 2, acertou 1
    metrics = train.actionable_metrics(y_true, y_pred)
    assert metrics["action_rate"] == 0.5
    assert metrics["action_precision"] == pytest.approx(0.5)


def test_reversal_rate_counts_only_sign_inversions():
    """Dizer SOBE quando desceu custa dinheiro; confundir com ESPERAR custa
    apenas uma oportunidade. Sao erros diferentes e contam-se em separado."""
    y_true = np.array([F.DOWN, F.WAIT, F.UP])
    y_pred = np.array([F.UP, F.UP, F.UP])   # 1 inversao, 1 falso positivo, 1 certo
    metrics = train.actionable_metrics(y_true, y_pred)
    assert metrics["reversal_rate"] == pytest.approx(1 / 3)


def test_seconds_data_is_not_claimed_anywhere():
    """O yfinance rejeita interval=1s. Nenhum timeframe pode prometer segundos."""
    assert all(tf in config.PATTERN_TIMEFRAMES for tf in config.TREES["timeframes"])
    assert not any(tf.endswith("s") for tf in config.TREES["timeframes"])


# --------------------------------------------------------------------------
# Alvo de MAGNITUDE
# --------------------------------------------------------------------------

def test_magnitude_is_identical_for_equal_moves_in_opposite_directions():
    """O ponto central do alvo: um salto de +2% e um de -2%, com o mesmo
    historico de volatilidade, tem de dar exactamente a mesma magnitude.

    (Nao se testa isto espelhando precos com 2*base - preco: retornos
    percentuais nao sao simetricos sob reflexao -- 105->106 e' +0.95% e o
    espelho 95->94 e' -1.05%. Constroi-se o salto explicitamente.)"""
    rng = np.random.default_rng(5)
    history = [100.0]
    for _ in range(60):
        history.append(history[-1] * (1 + rng.choice([0.003, -0.003])))

    def series(final_move):
        closes = history + [history[-1] * (1 + final_move)]
        index = pd.date_range("2026-08-10 13:30", periods=len(closes), freq="1min", tz="UTC")
        return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                             "close": closes, "volume": np.full(len(closes), 5000)}, index=index)

    up = F.label_magnitude(series(0.02))["magnitude_ratio"].iloc[-2]
    down = F.label_magnitude(series(-0.02))["magnitude_ratio"].iloc[-2]
    assert up == pytest.approx(down, rel=1e-9)
    assert up > 0


def test_magnitude_ratio_is_a_pure_function_of_the_absolute_move():
    """Contraprova algebrica: magnitude = |retorno| / (banda / band_k).
    Se a direccao entrasse por algum lado, esta igualdade partia."""
    result = F.label_magnitude(_bars(400), band_k=0.5).dropna()
    implied = result["abs_return"] / (result["band"] / 0.5)
    pd.testing.assert_series_equal(result["magnitude_ratio"], implied,
                                   check_names=False, rtol=1e-9)


def test_big_move_is_magnitude_above_the_band():
    result = F.label_magnitude(_bars(400)).dropna()
    big = result[result["big_move"] == 1]
    small = result[result["big_move"] == 0]
    assert (big["abs_return"] > big["band"]).all()
    assert (small["abs_return"] <= small["band"]).all()


def test_magnitude_ratio_is_the_move_in_recent_standard_deviations():
    result = F.label_magnitude(_bars(400), band_k=0.5).dropna()
    assert ((result["magnitude_ratio"] > 0.5) == (result["big_move"] == 1)).all()


def test_big_move_base_rate_is_not_degenerate():
    """Se quase tudo fosse grande (ou quase nada), o alvo nao teria conteudo."""
    share = F.label_magnitude(_bars(800))["big_move"].dropna().mean()
    assert 0.2 < share < 0.8, f"taxa-base degenerada: {share:.1%}"


def test_magnitude_label_does_not_leak_the_future_into_features():
    """O rotulo usa o futuro (e' suposto); as features nao. Contraprova de que
    os dois blocos estao mesmo separados."""
    bars = _bars(300)
    features_before = F.indicator_features(bars).iloc[:200]
    mutated = bars.copy()
    mutated.iloc[250:, mutated.columns.get_loc("close")] *= 3.0
    pd.testing.assert_frame_equal(features_before, F.indicator_features(mutated).iloc[:200])
    assert not F.label_magnitude(bars)["magnitude_ratio"].iloc[240:260].equals(
        F.label_magnitude(mutated)["magnitude_ratio"].iloc[240:260])


def test_decile_lift_ranks_by_predicted_magnitude():
    from src.trees import magnitude
    predictions = np.linspace(0, 1, 1000)
    realised = predictions * 2          # relacao perfeita
    deciles = magnitude.decile_lift(predictions, realised)
    assert len(deciles) == 10
    means = [d["mean_realised"] for d in deciles]
    assert means == sorted(means), "os decis tem de ficar por ordem crescente"
    assert means[-1] > means[0] * 5


def test_decile_lift_is_flat_when_predictions_are_noise():
    from src.trees import magnitude
    rng = np.random.default_rng(4)
    deciles = magnitude.decile_lift(rng.random(2000), rng.random(2000))
    means = [d["mean_realised"] for d in deciles]
    assert max(means) - min(means) < 0.15, "ruido nao pode produzir lift entre decis"
