"""Calcula a tabela de verdade, grava-a, e regista o veredicto de hoje.

Unico ponto de escrita da camada de decisao. A aplicacao web nunca calcula nada
-- le esta tabela e mostra-a. A separacao e' deliberada: uma pagina que treina
modelos a pedido fica lenta, nao e' reproduzivel, e o utilizador nunca sabe se
esta' a ver o resultado de agora ou o da ultima vez que carregou.

    py src/decision/run_decision.py                 # tabela + veredicto de hoje
    py src/decision/run_decision.py --walkforward   # inclui a avaliacao completa
    py src/decision/run_decision.py --rebuild       # ignora a cache do painel
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.decision import panel as P
from src.decision import signals as S
from src.decision import truth_table as T
from src.decision import walkforward as W

import pandas as pd

D = config.DECISION
PANEL_CACHE = Path(config.DATA_DIR) / "decision_panel.pkl"
SIGNALS_CACHE = Path(config.DATA_DIR) / "decision_signals.pkl"


def prepare(conn, rebuild: bool = False) -> pd.DataFrame:
    """Painel com sinais. A cache poupa alguns minutos e nao muda resultados --
    e' o mesmo calculo deterministico sobre os mesmos precos."""
    if not rebuild and SIGNALS_CACHE.exists():
        return pd.read_pickle(SIGNALS_CACHE)

    frame = (pd.read_pickle(PANEL_CACHE)
             if not rebuild and PANEL_CACHE.exists() else P.build(conn))
    frame.to_pickle(PANEL_CACHE)

    _, validation, _ = P.split(frame)
    enriched = S.attach(conn, frame, frame["ts"] < validation["ts"].max())
    enriched.to_pickle(SIGNALS_CACHE)
    return enriched


def latest_verdicts(enriched: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Onde cada accao cai na tabela, no ultimo dia disponivel."""
    last = enriched["ts"].max()
    today = enriched[enriched["ts"] == last].copy()
    lookup = table.set_index(S.SIGNAL_NAMES)[["estimate", "accao"]]
    keys = pd.MultiIndex.from_frame(today[S.SIGNAL_NAMES])
    joined = lookup.reindex(keys)
    today["estimate"] = joined["estimate"].values
    today["accao"] = pd.Series(joined["accao"].values, index=today.index).fillna("ESPERAR")
    return today


def store(conn, outcome: dict, today: pd.DataFrame, walk: dict | None) -> int:
    cursor = conn.execute(
        """INSERT INTO decision_runs
             (created_at, horizon_days, threshold, n_cells, n_cells_supported,
              mean_weight, train_days, validation_days, test_days,
              results_json, ablation_json, interaction_diff, interaction_low,
              interaction_high, walkforward_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now(timezone.utc).isoformat(), D["horizon_days"],
         outcome["threshold"], outcome["n_cells"], outcome["n_cells_supported"],
         outcome["mean_weight"], outcome["train_days"],
         outcome["validation_days"], outcome["test_days"],
         json.dumps(outcome["results"]), json.dumps(outcome["ablation"]),
         outcome["interaction_difference"], outcome["interaction_ci"][0],
         outcome["interaction_ci"][1],
         json.dumps(walk) if walk else None),
    )
    run_id = int(cursor.lastrowid)

    table = outcome["table"]
    conn.executemany(
        """INSERT OR REPLACE INTO decision_cells
             (run_id, curto, rapido, padrao, regime, alarme, n_rows, n_days,
              cell_mean, additive, weight, estimate, p_up, accao)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(run_id, r["curto"], r["rapido"], r["padrao"], r["regime"], r["alarme"],
          int(r["n_rows"]), int(r["n_days"]), float(r["cell_mean"]),
          float(r["additive"]), float(r["weight"]), float(r["estimate"]),
          float(r["p_up"]), r["accao"]) for _, r in table.iterrows()],
    )

    conn.executemany(
        """INSERT OR REPLACE INTO decision_current
             (run_id, ticker, ts, curto, rapido, padrao, regime, alarme,
              confidence, estimate, accao)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [(run_id, r["ticker"], pd.Timestamp(r["ts"]).isoformat(),
          r["curto"], r["rapido"], r["padrao"], r["regime"], r["alarme"],
          float(r.get("curto_confidence", 0.0)),
          None if pd.isna(r["estimate"]) else float(r["estimate"]), r["accao"])
         for _, r in today.iterrows()],
    )
    conn.commit()
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walkforward", action="store_true",
                        help="avaliacao trimestral fora da amostra (varios minutos)")
    parser.add_argument("--rebuild", action="store_true",
                        help="reconstruir o painel a partir dos precos")
    args = parser.parse_args()

    conn = database.get_connection()
    enriched = prepare(conn, rebuild=args.rebuild)
    outcome = T.run(conn, enriched)
    if outcome is None:
        print("dados insuficientes para construir a tabela")
        return

    walk = None
    if args.walkforward:
        print("walk-forward (re-treino trimestral)...", flush=True)
        walk = W.run(conn, enriched, verbose=True)
        if walk:
            # A tabela nao e' serializavel e nao e' precisa aqui.
            walk = {k: v for k, v in walk.items() if k != "table"}

    today = latest_verdicts(enriched, outcome["table"])
    run_id = store(conn, outcome, today, walk)

    counts = today["accao"].value_counts().to_dict()
    print(f"execucao #{run_id} gravada: {outcome['n_cells']} celulas, "
          f"{len(today)} accoes avaliadas em {today['ts'].max().date()}")
    print("  veredicto de hoje: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    database.log_run(conn, str(today["ts"].max().date()), "decision", "OK",
                     f"run {run_id}, {outcome['n_cells']} celulas")
    conn.close()


if __name__ == "__main__":
    main()
