"""Testes da cadeia de Markov e da previsao encadeada de N passos."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.patterns import sequence


def _transitions(pairs: dict) -> dict:
    return {key: {"count": value, "median_bars": 10.0} for key, value in pairs.items()}


def test_distribution_sums_to_one():
    transitions = _transitions({("BULL_FLAG", "DOUBLE_TOP"): 5, ("BULL_FLAG", "BEAR_FLAG"): 3})
    total = sum(p for _, p, _, _ in sequence.transition_distribution(transitions, "BULL_FLAG"))
    assert total == pytest.approx(1.0)


def test_unseen_from_state_falls_back_to_uniform_when_nothing_is_known():
    distribution = sequence.transition_distribution(_transitions({}), "DIAMOND_TOP")
    probabilities = [p for _, p, _, _ in distribution]
    assert all(p == pytest.approx(1 / len(config.PATTERN_TYPES)) for p in probabilities)
    assert all(support == 0 for _, _, support, _ in distribution)


def test_unseen_from_state_falls_back_to_the_marginal_not_to_uniform():
    """De um padrao nunca observado, a melhor estimativa e' a frequencia
    global -- nao 'todos igualmente provaveis', que e' pior do que o que ja'
    se sabe sobre os dados."""
    transitions = _transitions({
        ("BULL_FLAG", "DOUBLE_TOP"): 80,
        ("BEAR_FLAG", "DOUBLE_TOP"): 60,
        ("PENNANT", "DIAMOND_TOP"): 4,
    })
    distribution = sequence.transition_distribution(transitions, "RECTANGLE_TOP")
    assert distribution[0][0] == "DOUBLE_TOP"
    assert distribution[0][2] == 0  # sem suporte proprio nenhum
    uniform = 1 / len(config.PATTERN_TYPES)
    assert distribution[0][1] > uniform * 2


def test_strong_conditional_evidence_overrides_the_marginal():
    """Com observacoes suficientes, a condicional tem de ganhar a' marginal --
    senao o recuo estaria a apagar o sinal que se quer aprender."""
    transitions = _transitions({
        ("BULL_FLAG", "PENNANT"): 60,      # condicional forte e especifica
        ("BEAR_FLAG", "DOUBLE_TOP"): 200,  # marginal dominada por DOUBLE_TOP
        ("PENNANT", "DOUBLE_TOP"): 200,
    })
    top = sequence.transition_distribution(transitions, "BULL_FLAG")[0]
    assert top[0] == "PENNANT"


def test_smoothing_keeps_unobserved_transitions_possible():
    transitions = _transitions({("BULL_FLAG", "DOUBLE_TOP"): 50})
    distribution = dict((p, prob) for p, prob, _, _ in sequence.transition_distribution(transitions, "BULL_FLAG"))
    assert distribution["DOUBLE_TOP"] > 0.5
    assert 0 < distribution["DIAMOND_TOP"] < 0.05  # improvavel, mas nunca impossivel


def test_support_is_reported_alongside_probability():
    transitions = _transitions({("BULL_FLAG", "DOUBLE_TOP"): 7})
    top = sequence.transition_distribution(transitions, "BULL_FLAG")[0]
    assert top[0] == "DOUBLE_TOP"
    assert top[2] == 7


def test_chain_has_requested_number_of_steps():
    transitions = _transitions({("BULL_FLAG", "DOUBLE_TOP"): 10, ("DOUBLE_TOP", "BEAR_FLAG"): 10})
    best, _ = sequence.forecast_chain(transitions, "BULL_FLAG", steps=4)
    assert [s.step for s in best] == [1, 2, 3, 4]


def test_cumulative_confidence_is_product_of_step_confidences():
    """O ponto central do pedido: a certeza do passo N e' condicionada na
    previsao do passo N-1, e a acumulada e' o produto ao longo da cadeia."""
    transitions = _transitions({("BULL_FLAG", "DOUBLE_TOP"): 20, ("DOUBLE_TOP", "BEAR_FLAG"): 20})
    best, _ = sequence.forecast_chain(transitions, "BULL_FLAG", steps=4)

    running = 1.0
    for step in best:
        running *= step.step_confidence
        assert step.cumulative_confidence == pytest.approx(running)


def test_cumulative_confidence_is_monotonically_decreasing():
    transitions = _transitions({("BULL_FLAG", "DOUBLE_TOP"): 20, ("DOUBLE_TOP", "BEAR_FLAG"): 15})
    best, _ = sequence.forecast_chain(transitions, "BULL_FLAG", steps=4)
    cumulative = [s.cumulative_confidence for s in best]
    assert cumulative == sorted(cumulative, reverse=True)
    assert cumulative[-1] < cumulative[0]


def test_chain_follows_its_own_previous_prediction():
    """A -> B quase certo, B -> C quase certo: a cadeia tem de dar A,B,C,...
    Se o passo 2 fosse condicionado em A em vez de B, davam-se B,B,B."""
    transitions = _transitions({
        ("PENNANT", "DOUBLE_TOP"): 100,
        ("DOUBLE_TOP", "BEAR_FLAG"): 100,
        ("BEAR_FLAG", "DIAMOND_BOTTOM"): 100,
    })
    best, _ = sequence.forecast_chain(transitions, "PENNANT", steps=3)
    assert [s.pattern_type for s in best] == ["DOUBLE_TOP", "BEAR_FLAG", "DIAMOND_BOTTOM"]


def test_expected_bars_accumulate_along_the_chain():
    transitions = {
        ("PENNANT", "DOUBLE_TOP"): {"count": 50, "median_bars": 10.0},
        ("DOUBLE_TOP", "BEAR_FLAG"): {"count": 50, "median_bars": 25.0},
    }
    best, _ = sequence.forecast_chain(transitions, "PENNANT", steps=2)
    assert best[0].expected_bars == pytest.approx(10.0)
    assert best[1].expected_bars == pytest.approx(35.0)


def test_alternatives_never_repeat_the_chosen_pattern():
    transitions = _transitions({
        ("BULL_FLAG", "DOUBLE_TOP"): 4,
        ("BULL_FLAG", "BEAR_FLAG"): 4,   # empate deliberado com o de cima
        ("BULL_FLAG", "PENNANT"): 3,
    })
    best, _ = sequence.forecast_chain(transitions, "BULL_FLAG", steps=2)
    for step in best:
        names = [a["pattern_type"] for a in step.alternatives]
        assert step.pattern_type not in names


def test_beam_returns_distinct_alternative_paths():
    transitions = _transitions({
        ("BULL_FLAG", "DOUBLE_TOP"): 10,
        ("BULL_FLAG", "BEAR_FLAG"): 8,
        ("BULL_FLAG", "PENNANT"): 6,
    })
    best, alternatives = sequence.forecast_chain(transitions, "BULL_FLAG", steps=2, beam_width=3)
    paths = [[s.pattern_type for s in path] for path in [best] + alternatives]
    assert len(paths) == 3
    assert len({tuple(p) for p in paths}) == 3


def test_best_path_has_highest_cumulative_confidence():
    transitions = _transitions({
        ("BULL_FLAG", "DOUBLE_TOP"): 10,
        ("BULL_FLAG", "BEAR_FLAG"): 8,
        ("DOUBLE_TOP", "PENNANT"): 10,
    })
    best, alternatives = sequence.forecast_chain(transitions, "BULL_FLAG", steps=2, beam_width=4)
    for alternative in alternatives:
        assert best[-1].cumulative_confidence >= alternative[-1].cumulative_confidence


def test_build_transitions_never_bridges_two_tickers(conn):
    """A ultima formacao de um ticker nao pode contar como transicao para a
    primeira do ticker seguinte -- seria uma transicao que nunca existiu."""
    rows = [
        ("AAPL", "5m", "BULL_FLAG", "2026-01-01T10:00:00+00:00", "2026-01-01T11:00:00+00:00",
         "2026-01-01T11:05:00+00:00", 0, 10, 0.9, "{}", "now"),
        ("MSFT", "5m", "DIAMOND_TOP", "2026-01-02T10:00:00+00:00", "2026-01-02T11:00:00+00:00",
         "2026-01-02T11:05:00+00:00", 0, 10, 0.9, "{}", "now"),
    ]
    conn.executemany(
        """INSERT INTO detected_patterns (ticker, timeframe, pattern_type, start_ts, end_ts,
             confirmed_ts, start_idx, end_idx, quality, meta_json, detected_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    transitions = sequence.build_transitions(conn, "5m", tickers=["AAPL", "MSFT"])
    assert ("BULL_FLAG", "DIAMOND_TOP") not in transitions
    assert transitions == {}
