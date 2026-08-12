"""Ciclo do subsistema de padroes, para correr a seguir ao pipeline diario:
ingerir barras intradiarias -> redetectar padroes -> reconstruir a matriz de
transicoes -> gravar um snapshot da cadeia prevista para cada ticker.

Uso: py src/patterns/run_patterns.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.patterns import backfill, classifier, ingest_intraday, live, sequence


def main():
    conn = database.get_connection()
    database.seed_watchlist(conn)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    ingest_result = ingest_intraday.ingest_all(conn)
    failed = [k for k, v in ingest_result.items() if v is None]
    print(f"barras intradiarias: {len(ingest_result) - len(failed)}/{len(ingest_result)} ok")
    database.log_run(conn, today, "ingest_intraday", "OK" if not failed else "WARN",
                      f"failed={failed}")

    for timeframe in config.PATTERN_TIMEFRAMES:
        tally = backfill.backfill_timeframe(conn, timeframe)
        transitions = sequence.build_transitions(conn, timeframe)
        sequence.store_transitions(conn, timeframe, transitions)
        metrics = sequence.evaluate(conn, timeframe)
        print(f"{timeframe}: {sum(tally.values())} padroes, {len(transitions)} transicoes, "
              f"markov={metrics.get('markov_top1_accuracy', 0):.1%} "
              f"baseline={metrics.get('baseline_frequency_accuracy', 0):.1%}")
        database.log_run(
            conn, today, f"patterns_{timeframe}", "OK",
            f"patterns={sum(tally.values())} markov={metrics.get('markov_top1_accuracy')} "
            f"baseline={metrics.get('baseline_frequency_accuracy')}",
        )

    # Retreina o classificador contextual e volta a medir se ele ja' bate a
    # cadeia de Markov. Nao esta' em producao (ver config), mas manter a
    # medicao a correr todos os dias e' o que responde a pergunta "ja' ha'
    # dados suficientes para o contexto valer a pena?" sem trabalho manual.
    for timeframe in config.PATTERN_TIMEFRAMES:
        result = classifier.train(conn, timeframe)
        if result is None:
            print(f"classificador {timeframe}: adiado (dados insuficientes)")
            continue
        delta = (result["accuracy"] - result["markov_accuracy"]) * 100
        se = result["standard_error"] * 100
        verdict = "SIGNIFICATIVO" if abs(delta) > 2 * se else "dentro do ruido"
        print(f"classificador {timeframe}: {result['accuracy']:.1%} vs markov "
              f"{result['markov_accuracy']:.1%} ({delta:+.1f}pp, +-{se:.1f}pp) -> {verdict}")
        database.log_run(conn, today, f"pattern_classifier_{timeframe}", "OK",
                          f"accuracy={result['accuracy']:.4f} markov={result['markov_accuracy']:.4f} "
                          f"delta_pp={delta:.2f} se_pp={se:.2f} verdict={verdict}")

    live.get_transitions(conn, config.PATTERN_LIVE_TIMEFRAME, refresh=True)
    stored = 0
    for ticker in config.PATTERN_TICKERS:
        result = live.analyse(conn, ticker, config.PATTERN_LIVE_TIMEFRAME)
        stored += live.store_forecast(conn, result)
    print(f"snapshots de previsao gravados: {stored}")

    conn.close()


if __name__ == "__main__":
    main()
