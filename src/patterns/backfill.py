"""Backfill: corre a deteccao sobre todo o historico intradiario, guarda os
padroes, reconstroi a matriz de transicoes e reporta a avaliacao.

Uso: py src/patterns/backfill.py
"""

import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.patterns import scanner, sequence
from src.patterns.ingest_intraday import load_bars


def backfill_timeframe(conn, timeframe: str, tickers=None) -> Counter:
    tickers = tickers or config.PATTERN_TICKERS
    pivot_window = config.PATTERN_TIMEFRAMES[timeframe]["pivot_window"]
    tally = Counter()
    for ticker in tickers:
        bars = load_bars(conn, ticker, timeframe)
        if bars.empty:
            continue
        matches = scanner.scan_patterns(bars, pivot_window)
        scanner.store_patterns(conn, ticker, timeframe, bars, matches)
        tally.update(m.pattern_type for m in matches)
    return tally


def main():
    conn = database.get_connection()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for timeframe in config.PATTERN_TIMEFRAMES:
        print(f"=== {timeframe} ===")
        tally = backfill_timeframe(conn, timeframe)
        total = sum(tally.values())
        print(f"  {total} padroes detectados:")
        for pattern_type, n in tally.most_common():
            print(f"    {pattern_type:30s} {n:4d}  ({100 * n / total:.1f}%)")

        transitions = sequence.build_transitions(conn, timeframe)
        sequence.store_transitions(conn, timeframe, transitions)
        metrics = sequence.evaluate(conn, timeframe)
        cells = len(config.PATTERN_TYPES) ** 2
        print(f"  matriz: {len(transitions)}/{cells} celulas com dados "
              f"({metrics.get('matrix_density', 0):.1%} de densidade)")
        if metrics.get("n_test"):
            print(f"  avaliacao walk-forward (treino={metrics['n_train']}, teste={metrics['n_test']}):")
            print(f"    Markov top-1:      {metrics['markov_top1_accuracy']:.1%}")
            print(f"    baseline frequencia: {metrics['baseline_frequency_accuracy']:.1%} "
                  f"(prever sempre {metrics['baseline_pattern']})")
            print(f"    lift: {metrics['lift']*100:+.2f}pp +-{metrics['standard_error']*100:.2f}pp  "
                  f"kappa={metrics['kappa']:+.4f}  -> "
                  f"{'SIGNIFICATIVO' if metrics['significant'] else 'dentro do ruido'}")
        database.log_run(conn, today, f"pattern_backfill_{timeframe}", "OK",
                          f"patterns={total} transitions={len(transitions)}")

    conn.close()


if __name__ == "__main__":
    main()
