"""Bootstrap one-time do TradingML: schema, seed, historico de precos,
technical_features, outcomes dos dois horizontes, treino inicial dos dois
modelos. Idempotente -- pode ser corrido de novo sem duplicar nada (tudo por
baixo sao upserts), mas serve sobretudo para arrancar o projeto do zero.

Uso: py scripts/bootstrap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from db import database
from src import features, ingest_prices, model, outcomes


def main():
    conn = database.get_connection()
    database.seed_watchlist(conn)
    database.seed_portfolio_state(conn)

    print(f"Ingesting ~{config.BOOTSTRAP_YEARS}y of history for {len(config.TICKERS)} tickers...")
    ingest_result = ingest_prices.ingest_all(conn, period=f"{config.BOOTSTRAP_YEARS}y")
    for ticker, n in ingest_result.items():
        print(f"  prices {ticker}: {n if n is not None else 'FAILED'} rows")

    print("Computing technical_features...")
    tech_result = features.recompute_all_technical_features(conn)
    for ticker, n in tech_result.items():
        print(f"  technical_features {ticker}: {n} rows")

    print("Resolving outcomes (labels) per horizon...")
    for horizon in config.HORIZONS:
        result = outcomes.resolve_all_outcomes(conn, horizon)
        print(f"  outcomes {horizon}: {sum(result.values())} rows touched")

    print("Training initial models per horizon...")
    for horizon in config.HORIZONS:
        train_result = model.train(conn, horizon)
        if train_result is None:
            print(f"  model {horizon}: adiado (dados insuficientes)")
        else:
            print(
                f"  model {horizon}: n={train_result['n_rows']} "
                f"logistic_cv={train_result['cv_accuracy_logistic']:.4f} "
                f"rf_cv={train_result['cv_accuracy_rf']:.4f} "
                f"baseline_majority={train_result['baseline_majority']:.4f} "
                f"baseline_persistence={train_result['baseline_persistence']}"
            )

    conn.close()
    print("Bootstrap complete.")


if __name__ == "__main__":
    main()
