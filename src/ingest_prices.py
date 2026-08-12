"""Ingestao de barras diarias (OHLCV) via yfinance, upsert idempotente em `prices`.

Uso:
    py src/ingest_prices.py                 # ultimos 5 dias, watchlist completa
    py src/ingest_prices.py --period 2y      # historico maior (usado pelo bootstrap)
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from db import database

import pandas as pd
import yfinance as yf


def today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def fetch_ticker_history(ticker: str, period: str = "5d") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df.empty:
        return df
    df = df.dropna(subset=["Close"])
    return df


def upsert_prices(conn, ticker: str, df: pd.DataFrame) -> int:
    now = datetime.now(timezone.utc).isoformat()
    rows = 0
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        volume = int(row["Volume"]) if pd.notna(row["Volume"]) else 0
        conn.execute(
            """INSERT INTO prices (ticker, date, open, high, low, close, volume, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker, date) DO UPDATE SET
                 open = excluded.open, high = excluded.high, low = excluded.low,
                 close = excluded.close, volume = excluded.volume,
                 ingested_at = excluded.ingested_at""",
            (
                ticker, date_str,
                float(row["Open"]) if pd.notna(row["Open"]) else None,
                float(row["High"]) if pd.notna(row["High"]) else None,
                float(row["Low"]) if pd.notna(row["Low"]) else None,
                float(row["Close"]), volume, now,
            ),
        )
        rows += 1
    conn.commit()
    return rows


def ingest_all(conn, period: str = "5d", tickers=None) -> dict:
    """Ingere barras recentes para os tickers dados (ou toda a watchlist).
    Uma falha num ticker (rede, rate-limit, etc.) nao aborta os restantes --
    fica registada em run_log e o dict de resultados tem None nesse ticker."""
    tickers = tickers or config.TICKERS
    results = {}
    for ticker in tickers:
        try:
            df = fetch_ticker_history(ticker, period=period)
            results[ticker] = upsert_prices(conn, ticker, df) if not df.empty else 0
        except Exception as exc:
            database.log_run(conn, today_str(), "ingest_prices", "ERROR", f"{ticker}: {exc}")
            results[ticker] = None
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="5d", help="periodo yfinance, ex: 5d, 1mo, 2y")
    args = parser.parse_args()

    connection = database.get_connection()
    database.seed_watchlist(connection)
    outcome = ingest_all(connection, period=args.period)
    ok = sum(1 for v in outcome.values() if v)
    failed = [t for t, v in outcome.items() if v is None]
    for ticker, n in outcome.items():
        print(f"  {ticker}: {n if n is not None else 'FAILED'} rows")
    print(f"Ingested {ok}/{len(outcome)} tickers ok.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    database.log_run(connection, today_str(), "ingest_prices", "OK" if not failed else "WARN",
                      f"ok={ok} failed={failed}")
    connection.close()
