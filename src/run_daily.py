"""Orquestrador diario: ingest precos -> features tecnicas -> resolver
outcomes -> por horizonte (retreino condicional -> prever -> simular).

Cada peca que escreve na BD e' idempotente por si (upserts com chave
ticker+date[+horizon], simulate_day() salta se (date,horizon) ja' foi
processado) -- por isso correr este script duas vezes para a mesma data e'
seguro, e correr num dia sem sessao de mercado (fim de semana/feriado) e'
um no-op limpo, sem erro.

Uso:
    py src/run_daily.py                    # processa a data mais recente disponivel
    py src/run_daily.py --date 2026-08-12   # reprocessa uma data especifica
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from db import database
from src import features, ingest_prices, model, outcomes, simulator


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _should_retrain(conn, horizon: str) -> bool:
    last = conn.execute(
        "SELECT trained_at FROM model_versions WHERE horizon = ? ORDER BY trained_at DESC LIMIT 1",
        (horizon,),
    ).fetchone()
    if last is None:
        return True
    trained_at = datetime.fromisoformat(last["trained_at"])
    age_days = (datetime.now(timezone.utc) - trained_at).days
    threshold = 7 if config.HORIZON_PARAMS[horizon]["retrain_frequency"] == "weekly" else 30
    return age_days >= threshold


def run(date: str = None) -> dict:
    conn = database.get_connection()
    database.seed_watchlist(conn)
    database.seed_portfolio_state(conn)

    summary = {"stages": {}}

    ingest_result = ingest_prices.ingest_all(conn, period="5d")
    failed = [t for t, v in ingest_result.items() if v is None]
    summary["stages"]["ingest_prices"] = ingest_result
    database.log_run(conn, _today(), "ingest_prices", "OK" if not failed else "WARN", str(ingest_result))

    latest_date = conn.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    target_date = date or latest_date
    if target_date is None:
        database.log_run(conn, _today(), "run_daily", "ERROR", "sem nenhum preco na BD -- corre scripts/bootstrap.py primeiro")
        conn.close()
        summary["skipped"] = "no price data at all"
        return summary

    has_bar = conn.execute("SELECT COUNT(*) FROM prices WHERE date = ?", (target_date,)).fetchone()[0] > 0
    if not has_bar:
        database.log_run(conn, target_date, "run_daily", "WARN",
                          f"sem barra de precos para {target_date} (fim de semana/feriado/ainda nao disponivel) -- a sair sem fazer nada")
        conn.close()
        summary["skipped"] = f"no price bar for {target_date}"
        return summary

    tech_result = features.recompute_all_technical_features(conn)
    summary["stages"]["technical_features"] = tech_result

    for horizon in config.HORIZONS:
        outcomes.resolve_all_outcomes(conn, horizon)

    for horizon in config.HORIZONS:
        if _should_retrain(conn, horizon):
            train_result = model.train(conn, horizon)
            summary["stages"][f"train_{horizon}"] = train_result
            database.log_run(conn, target_date, f"train_{horizon}", "OK" if train_result else "WARN", str(train_result))

        predictions = model.predict_and_record_all(conn, target_date, horizon)
        summary["stages"][f"predict_{horizon}"] = {
            t: {"direction": r["predicted_direction"], "confidence": round(r["confidence"], 3)} if r else None
            for t, r in predictions.items()
        }

        sim_result = simulator.simulate_day(conn, target_date, horizon)
        summary["stages"][f"simulate_{horizon}"] = sim_result
        database.log_run(conn, target_date, f"simulate_{horizon}", "OK",
                          f"trades={len(sim_result.get('trades', []))} total_value={sim_result.get('total_value')}")

    conn.close()
    summary["date"] = target_date
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=str, default=None, help="reprocessar uma data especifica (YYYY-MM-DD)")
    args = parser.parse_args()

    outcome = run(args.date)
    if "skipped" in outcome:
        print(f"run_daily: SKIPPED -- {outcome['skipped']}")
    else:
        print(f"run_daily concluido para {outcome['date']}")
        for stage, value in outcome["stages"].items():
            print(f"  {stage}: {value}")
