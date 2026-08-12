"""Modelo de sequencia de padroes: cadeia de Markov de 1a ordem + beam search.

O que aprende: dado que acabou de se formar o padrao X, com que probabilidade
o proximo padrao a formar-se e' Y, e ao fim de quantas barras tipicamente.

Como preve 4 passos a' frente (era exactamente isto que era pedido): o passo 1
sai da distribuicao P(.|padrao_actual); o passo 2 ja' nao parte do padrao
actual mas do padrao PREVISTO no passo 1, e por ai fora. Cada passo tem por
isso duas leituras diferentes, e ambas sao reportadas:

  step_confidence       P(este padrao | o padrao previsto no passo anterior)
                        -- "assumindo que acertei ate' aqui, quao certo estou
                        deste?"
  cumulative_confidence produto de todas as step_confidence ate' aqui
                        -- "qual a probabilidade da cadeia INTEIRA acontecer?"

A cumulativa cai depressa e isso nao e' um defeito do modelo: quatro eventos
incertos em cadeia sao mesmo improvaveis de acertar todos. Uma UI que
mostrasse so' a condicional daria uma falsa sensacao de certeza no passo 4.

Suavizacao de Dirichlet (alpha) evita probabilidades 0 ou 1 em celulas com
pouquissimas observacoes -- mas a defesa real e' o campo `support`, que diz
quantas transicoes reais sustentam cada numero. 60% assente em 2 observacoes
nao vale nada e tanto a API como a pagina web mostram sempre esse suporte.
"""

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database

import numpy as np

S = config.PATTERN_SEQUENCE


@dataclass
class ForecastStep:
    step: int
    pattern_type: str
    step_confidence: float
    cumulative_confidence: float
    expected_bars: float
    support: int
    # Os outros candidatos da MESMA distribuicao condicional deste passo.
    # Sem isto ve-se so' a moda: como a distribuicao e' relativamente plana, a
    # cadeia gulosa cai muitas vezes em auto-transicoes e esconderia estrutura
    # real (ex.: ASCENDING_TRIANGLE -> DOUBLE_TOP -> BEAR_FLAG) que fica logo
    # a seguir na mesma distribuicao.
    alternatives: list = field(default_factory=list)


@dataclass
class Forecast:
    ticker: str
    timeframe: str
    from_pattern: str
    as_of_ts: str
    current_quality: float
    steps: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Treino: contar transicoes entre padroes consecutivos
# --------------------------------------------------------------------------

def build_transitions(conn, timeframe: str, tickers=None) -> dict:
    """Conta (from -> to) sobre padroes consecutivos, por ticker.

    As sequencias sao por ticker e NUNCA se emenda o fim de um ticker com o
    inicio do seguinte -- isso inventaria transicoes que nunca aconteceram.
    Os contadores sao depois somados entre tickers (assume-se que a dinamica
    padrao-a-padrao e' partilhada; com 8 tickers, treinar uma matriz por
    ticker deixaria cada celula sem observacoes nenhumas)."""
    tickers = tickers or config.PATTERN_TICKERS
    counts = defaultdict(int)
    gaps = defaultdict(list)

    for ticker in tickers:
        rows = conn.execute(
            """SELECT pattern_type, end_idx FROM detected_patterns
               WHERE ticker = ? AND timeframe = ? ORDER BY confirmed_ts, end_idx""",
            (ticker, timeframe),
        ).fetchall()
        for previous, nxt in zip(rows, rows[1:]):
            counts[(previous["pattern_type"], nxt["pattern_type"])] += 1
            gaps[(previous["pattern_type"], nxt["pattern_type"])].append(
                nxt["end_idx"] - previous["end_idx"]
            )

    return {
        key: {"count": value, "median_bars": float(np.median(gaps[key]))}
        for key, value in counts.items()
    }


def store_transitions(conn, timeframe: str, transitions: dict) -> int:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM pattern_transitions WHERE timeframe = ?", (timeframe,))
    conn.executemany(
        """INSERT INTO pattern_transitions (timeframe, from_pattern, to_pattern, count, median_bars, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [(timeframe, f, t, v["count"], v["median_bars"], now) for (f, t), v in transitions.items()],
    )
    conn.commit()
    return len(transitions)


def load_transitions(conn, timeframe: str) -> dict:
    rows = conn.execute(
        "SELECT from_pattern, to_pattern, count, median_bars FROM pattern_transitions WHERE timeframe = ?",
        (timeframe,),
    ).fetchall()
    return {
        (r["from_pattern"], r["to_pattern"]): {"count": r["count"], "median_bars": r["median_bars"]}
        for r in rows
    }


# --------------------------------------------------------------------------
# Distribuicao condicional suavizada
# --------------------------------------------------------------------------

def marginal_distribution(transitions: dict) -> dict:
    """P(padrao) ignorando de onde se vem -- a frequencia global de cada
    padrao. E' exactamente o que o baseline de frequencia usa."""
    alpha = S["smoothing_alpha"]
    states = config.PATTERN_TYPES
    counts = defaultdict(int)
    for (_, to_pattern), value in transitions.items():
        counts[to_pattern] += value["count"]
    total = sum(counts.values())
    denominator = total + alpha * len(states)
    return {t: (counts[t] + alpha) / denominator for t in states}


def transition_distribution(transitions: dict, from_pattern: str, marginal: dict = None) -> list:
    """Devolve [(to_pattern, probabilidade, suporte, median_bars)] ordenado.

    Interpolacao de Jelinek-Mercer: a estimativa condicional e' misturada com
    a distribuicao MARGINAL, com peso proporcional ao suporte observado:

        P(to|from) = lambda * P_observado(to|from) + (1 - lambda) * P(to)
        lambda     = n_from / (n_from + BACKOFF_K)

    Porque isto e' melhor do que suavizar contra a uniforme: com 300 padroes
    espalhados por 256 celulas, muitos estados de partida tem 2 ou 3
    observacoes so'. Recuar para a uniforme nesses casos e' afirmar que todos
    os padroes sao igualmente provaveis, o que e' pior do que o que ja' se
    sabe (que ha' padroes claramente mais comuns que outros). Recuar para a
    marginal faz a cadeia degradar suavemente ate' ao baseline de frequencia
    em vez de para baixo dele -- so' se afasta da marginal quando ha' mesmo
    observacoes que o justifiquem."""
    states = config.PATTERN_TYPES
    if marginal is None:
        marginal = marginal_distribution(transitions)

    row = {t: transitions.get((from_pattern, t), {"count": 0, "median_bars": None}) for t in states}
    n_from = sum(v["count"] for v in row.values())
    weight = n_from / (n_from + S["backoff_k"]) if n_from else 0.0

    result = []
    for to_pattern in states:
        entry = row[to_pattern]
        conditional = entry["count"] / n_from if n_from else 0.0
        probability = weight * conditional + (1 - weight) * marginal[to_pattern]
        result.append((to_pattern, probability, entry["count"], entry["median_bars"]))
    result.sort(key=lambda item: -item[1])
    return result


def _median_bars_fallback(transitions: dict, timeframe: str) -> float:
    values = [v["median_bars"] for v in transitions.values() if v["median_bars"]]
    return float(np.median(values)) if values else 20.0


# --------------------------------------------------------------------------
# Beam search encadeada
# --------------------------------------------------------------------------

def forecast_chain(transitions: dict, from_pattern: str, steps: int = None,
                   beam_width: int = None, timeframe: str = "5m") -> tuple:
    """Devolve (melhor_caminho, alternativas).

    Cada caminho e' uma lista de ForecastStep. O passo i+1 e' condicionado no
    padrao previsto no passo i -- e' isto que faz da previsao uma cadeia e nao
    quatro previsoes independentes."""
    steps = steps or S["horizon_steps"]
    beam_width = beam_width or S["beam_width"]
    fallback_bars = _median_bars_fallback(transitions, timeframe)

    # beam: (cumulative_probability, ultimo_padrao, caminho)
    beams = [(1.0, from_pattern, [])]

    for step in range(1, steps + 1):
        expanded = []
        for cumulative, last_pattern, path in beams:
            distribution = transition_distribution(transitions, last_pattern)
            for to_pattern, probability, support, median_bars in distribution:
                # Excluir explicitamente o proprio padrao escolhido: com
                # probabilidades empatadas, a ordem da distribuicao e a do
                # desempate da beam podem divergir e ele reapareceria como
                # "alternativa a si mesmo".
                runners_up = [
                    {"pattern_type": p, "probability": round(prob, 4), "support": sup}
                    for p, prob, sup, _ in distribution if p != to_pattern
                ][:3]
                new_cumulative = cumulative * probability
                elapsed = path[-1].expected_bars if path else 0.0
                expanded.append((
                    new_cumulative, to_pattern,
                    path + [ForecastStep(
                        step=step,
                        pattern_type=to_pattern,
                        step_confidence=probability,
                        cumulative_confidence=new_cumulative,
                        expected_bars=elapsed + (median_bars or fallback_bars),
                        support=support,
                        alternatives=runners_up,
                    )],
                ))
        expanded.sort(key=lambda item: -item[0])
        beams = expanded[:beam_width]

    best = beams[0][2]
    alternatives = [b[2] for b in beams[1:]]
    return best, alternatives


# --------------------------------------------------------------------------
# Avaliacao honesta: a cadeia bate mesmo o baseline de frequencia?
# --------------------------------------------------------------------------

def evaluate(conn, timeframe: str, train_fraction: float = 0.7) -> dict:
    """Walk-forward por ticker: treina nos primeiros 70% da sequencia de cada
    ticker, testa nos restantes 30%. Compara top-1 accuracy da cadeia de
    Markov contra o baseline de prever sempre o padrao globalmente mais
    frequente (que ignora completamente o padrao actual)."""
    train_pairs, test_pairs = [], []
    for ticker in config.PATTERN_TICKERS:
        rows = conn.execute(
            """SELECT pattern_type FROM detected_patterns
               WHERE ticker = ? AND timeframe = ? ORDER BY confirmed_ts, end_idx""",
            (ticker, timeframe),
        ).fetchall()
        sequence = [r["pattern_type"] for r in rows]
        pairs = list(zip(sequence, sequence[1:]))
        split = int(len(pairs) * train_fraction)
        train_pairs.extend(pairs[:split])
        test_pairs.extend(pairs[split:])

    if not test_pairs:
        return {"n_test": 0}

    counts = defaultdict(int)
    for from_pattern, to_pattern in train_pairs:
        counts[(from_pattern, to_pattern)] += 1
    transitions = {key: {"count": value, "median_bars": None} for key, value in counts.items()}

    marginal = defaultdict(int)
    for _, to_pattern in train_pairs:
        marginal[to_pattern] += 1
    most_common = max(marginal, key=marginal.get) if marginal else config.PATTERN_TYPES[0]

    markov_hits = sum(
        1 for from_pattern, actual in test_pairs
        if transition_distribution(transitions, from_pattern)[0][0] == actual
    )
    baseline_hits = sum(1 for _, actual in test_pairs if actual == most_common)

    return {
        "n_train": len(train_pairs),
        "n_test": len(test_pairs),
        "markov_top1_accuracy": markov_hits / len(test_pairs),
        "baseline_frequency_accuracy": baseline_hits / len(test_pairs),
        "baseline_pattern": most_common,
        "n_distinct_transitions": len(counts),
        "matrix_density": len(counts) / (len(config.PATTERN_TYPES) ** 2),
    }


if __name__ == "__main__":
    connection = database.get_connection()
    for tf in config.PATTERN_TIMEFRAMES:
        transitions = build_transitions(connection, tf)
        store_transitions(connection, tf, transitions)
        metrics = evaluate(connection, tf)
        print(f"=== {tf} ===")
        print(f"  transicoes distintas: {len(transitions)} / {len(config.PATTERN_TYPES)**2} celulas "
              f"(densidade {metrics.get('matrix_density', 0):.1%})")
        if metrics["n_test"]:
            print(f"  treino={metrics['n_train']} teste={metrics['n_test']}")
            print(f"  Markov top-1:  {metrics['markov_top1_accuracy']:.1%}")
            print(f"  baseline freq: {metrics['baseline_frequency_accuracy']:.1%} "
                  f"(prever sempre {metrics['baseline_pattern']})")
    connection.close()
