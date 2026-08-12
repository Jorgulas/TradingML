"""Resolve os labels (outcomes) por horizonte a partir de `prices`.

Uma unica funcao serve tanto o backfill em massa do bootstrap (2 anos de
historico, quase tudo resolvivel de imediato) como o uso incremental diario
(so' fica resolvivel o que passou a ter `predict_ahead_days` sessoes de
preco a mais desde a ultima vez) -- a diferenca e' so' quanta informacao nova
existe quando e' chamada, o codigo e' o mesmo.

Garantia anti-leakage: o outcome de (ticker, date, horizon) so' e' calculado
quando a sessao alvo (`predict_ahead_days` sessoes depois da referencia)
already existe em `prices`. Nunca resolve cedo.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from db import database


def resolve_outcomes_for_ticker(conn, ticker: str, horizon: str) -> int:
    ahead = config.HORIZON_PARAMS[horizon]["predict_ahead_days"]
    rows = conn.execute(
        "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date", (ticker,)
    ).fetchall()
    dates = [r["date"] for r in rows]
    closes = [r["close"] for r in rows]
    now = datetime.now(timezone.utc).isoformat()

    n = 0
    for i in range(1, len(dates)):
        target_idx = i - 1 + ahead
        if target_idx >= len(closes):
            break  # sessao alvo ainda nao existe -- nada mais a resolver por agora
        date_str = dates[i]
        ref_close = closes[i - 1]
        target_close = closes[target_idx]
        direction = 1 if target_close > ref_close else 0
        conn.execute(
            """INSERT INTO outcomes (ticker, date, horizon, ref_close, target_close, direction, resolved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker, date, horizon) DO UPDATE SET
                 target_close = excluded.target_close, direction = excluded.direction,
                 resolved_at = excluded.resolved_at
               WHERE outcomes.direction IS NULL""",
            (ticker, date_str, horizon, ref_close, target_close, direction, now),
        )
        n += 1
    conn.commit()
    return n


def resolve_all_outcomes(conn, horizon: str, tickers=None) -> dict:
    tickers = tickers or config.TICKERS
    return {t: resolve_outcomes_for_ticker(conn, t, horizon) for t in tickers}


if __name__ == "__main__":
    connection = database.get_connection()
    for hz in config.HORIZONS:
        result = resolve_all_outcomes(connection, hz)
        resolved = connection.execute(
            "SELECT COUNT(*) FROM outcomes WHERE horizon = ? AND direction IS NOT NULL", (hz,)
        ).fetchone()[0]
        print(f"{hz}: touched {sum(result.values())} rows, {resolved} total resolved outcomes")
    connection.close()
