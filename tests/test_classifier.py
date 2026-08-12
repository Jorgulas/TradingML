"""Testes do classificador contextual e da sua integracao na cadeia.

O que se protege aqui, sobretudo: que o contexto medido so' possa influenciar
o PASSO 1 da cadeia. Os passos 2-4 partem de padroes previstos, que ainda nao
existem e nao tem contexto nenhum -- se um dia alguem os alimentar com
features "tipicas", os numeros ficam com ar mais informado sem informacao
nova por tras, e estes testes falham.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.patterns import classifier, context, sequence


def _bars(n=300, seed=3):
    rng = np.random.default_rng(seed)
    closes = 100 + rng.normal(0, 0.4, n).cumsum()
    index = pd.date_range("2026-01-05 14:30", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.002, "low": closes * 0.998,
         "close": closes, "volume": rng.integers(5_000, 50_000, n)},
        index=index,
    )


def _transitions(pairs: dict) -> dict:
    return {key: {"count": value, "median_bars": 12.0} for key, value in pairs.items()}


# --------------------------------------------------------------------------
# Extraccao de features
# --------------------------------------------------------------------------

def test_feature_vector_has_every_declared_feature():
    features = context.pattern_context_features(_bars(), 100, 140, "BULL_FLAG", 0.8, 4)
    assert set(features.keys()) == set(context.FEATURE_NAMES)


def test_pattern_type_is_one_hot_encoded():
    features = context.pattern_context_features(_bars(), 100, 140, "BULL_FLAG", 0.8, 4)
    flags = {k: v for k, v in features.items() if k.startswith("is_")}
    assert flags["is_BULL_FLAG"] == 1.0
    assert sum(flags.values()) == 1.0


def test_features_are_all_finite():
    """Um NaN silencioso aqui envenena o treino inteiro."""
    for start, end in [(0, 5), (100, 140), (250, 299)]:
        features = context.pattern_context_features(_bars(), start, end, "DOUBLE_TOP", 0.7, 3)
        for name, value in features.items():
            assert np.isfinite(value), f"{name} nao e' finito ({value})"


def test_prior_trend_ignores_bars_inside_the_pattern():
    """As features de tendencia previa tem de olhar so' para ANTES do padrao;
    se olhassem para dentro dele, estariam a usar o desfecho para o explicar."""
    bars = _bars()
    baseline = context.pattern_context_features(bars, 120, 160, "BULL_FLAG", 0.8, 4)

    mutated = bars.copy()
    mutated.iloc[130:160, mutated.columns.get_loc("close")] *= 3  # so' dentro do padrao
    after = context.pattern_context_features(mutated, 120, 160, "BULL_FLAG", 0.8, 4)

    for name in ("prior_trend_20", "prior_trend_60", "prior_volatility"):
        assert baseline[name] == pytest.approx(after[name])


def test_session_position_is_between_zero_and_one():
    bars = _bars()
    for end in (10, 100, 250):
        features = context.pattern_context_features(bars, max(0, end - 30), end, "PENNANT", 0.7, 3)
        assert 0.0 <= features["session_position"] <= 1.0


# --------------------------------------------------------------------------
# Integracao na cadeia -- o contexto so' pode tocar no passo 1
# --------------------------------------------------------------------------

def test_step1_override_changes_only_the_first_step():
    transitions = _transitions({
        ("BULL_FLAG", "DOUBLE_TOP"): 40,
        ("PENNANT", "BEAR_FLAG"): 40,
    })
    without, _ = sequence.forecast_chain(transitions, "BULL_FLAG", steps=3)

    # o classificador insiste em PENNANT no passo 1
    override = [("PENNANT", 0.9)] + [(p, 0.1 / 15) for p in config.PATTERN_TYPES if p != "PENNANT"]
    with_override, _ = sequence.forecast_chain(
        transitions, "BULL_FLAG", steps=3, step1_distribution=override
    )

    assert with_override[0].pattern_type == "PENNANT"
    assert with_override[0].step_confidence == pytest.approx(0.9)
    # passo 2 tem de seguir a cadeia de Markov a partir de PENNANT, nao do
    # padrao actual nem de qualquer contexto medido
    assert with_override[1].pattern_type == "BEAR_FLAG"
    assert without[0].pattern_type == "DOUBLE_TOP"


def test_steps_after_the_first_never_use_the_override_distribution():
    transitions = _transitions({("BULL_FLAG", "DOUBLE_TOP"): 50, ("DIAMOND_TOP", "PENNANT"): 50})
    override = [("DIAMOND_TOP", 0.95)] + [(p, 0.05 / 15) for p in config.PATTERN_TYPES if p != "DIAMOND_TOP"]
    chain, _ = sequence.forecast_chain(
        transitions, "BULL_FLAG", steps=4, step1_distribution=override
    )
    # se o override vazasse para os passos seguintes, DIAMOND_TOP repetia-se
    # com ~95% em todos eles
    assert chain[0].pattern_type == "DIAMOND_TOP"
    assert [s.pattern_type for s in chain[1:]] != ["DIAMOND_TOP"] * 3
    assert all(s.step_confidence < 0.95 for s in chain[1:])


def test_cumulative_confidence_still_compounds_with_an_override():
    transitions = _transitions({("BULL_FLAG", "DOUBLE_TOP"): 40, ("PENNANT", "BEAR_FLAG"): 40})
    override = [("PENNANT", 0.5)] + [(p, 0.5 / 15) for p in config.PATTERN_TYPES if p != "PENNANT"]
    chain, _ = sequence.forecast_chain(
        transitions, "BULL_FLAG", steps=3, step1_distribution=override
    )
    running = 1.0
    for step in chain:
        running *= step.step_confidence
        assert step.cumulative_confidence == pytest.approx(running)


def test_classifier_is_off_by_default():
    """Desligado com base em medicao (ver a nota no config). Se alguem ligar
    isto sem voltar a medir, e' bom que um teste o diga."""
    assert config.PATTERN_SEQUENCE["use_classifier"] is False


# --------------------------------------------------------------------------
# Protocolo de avaliacao
# --------------------------------------------------------------------------

def test_three_way_split_is_chronological_and_disjoint():
    frame = pd.DataFrame({
        "ticker": ["AAPL"] * 50 + ["MSFT"] * 50,
        "confirmed_ts": [f"2026-01-{d:02d}T10:00:00+00:00" for d in range(1, 51)] * 2,
        "from_pattern": ["BULL_FLAG"] * 100,
        "next_pattern": ["DOUBLE_TOP"] * 100,
    })
    train, validation, test = classifier._split_three_ways(frame)

    assert len(train) + len(validation) + len(test) == len(frame)
    for ticker in ("AAPL", "MSFT"):
        latest_train = train[train.ticker == ticker]["confirmed_ts"].max()
        earliest_test = test[test.ticker == ticker]["confirmed_ts"].min()
        assert latest_train < earliest_test, "treino tem de ser todo anterior ao teste"


def test_split_keeps_each_ticker_in_all_three_partitions():
    frame = pd.DataFrame({
        "ticker": ["AAPL"] * 30 + ["MSFT"] * 30,
        "confirmed_ts": [f"2026-02-{d:02d}T10:00:00+00:00" for d in range(1, 31)] * 2,
        "from_pattern": ["PENNANT"] * 60,
        "next_pattern": ["BEAR_FLAG"] * 60,
    })
    train, validation, test = classifier._split_three_ways(frame)
    for part in (train, validation, test):
        assert set(part["ticker"]) == {"AAPL", "MSFT"}
