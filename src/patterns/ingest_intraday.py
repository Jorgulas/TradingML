"""Ingestao de barras intradiarias (1h, 5m) via yfinance -> intraday_prices.

Uso:
    py src/patterns/ingest_intraday.py                 # todos os timeframes
    py src/patterns/ingest_intraday.py --timeframe 5m
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database

import pandas as pd
import yfinance as yf


def fetch_intraday(ticker: str, timeframe: str) -> pd.DataFrame:
    spec = config.PATTERN_TIMEFRAMES[timeframe]
    df = yf.Ticker(ticker).history(
        period=spec["yf_period"], interval=spec["yf_interval"], auto_adjust=True
    )
    if df.empty:
        return df
    return df.dropna(subset=["Close"])


def upsert_intraday(conn, ticker: str, timeframe: str, df: pd.DataFrame) -> int:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for idx, row in df.iterrows():
        ts = idx.tz_convert("UTC").isoformat() if idx.tzinfo else idx.isoformat()
        rows.append((
            ticker, timeframe, ts,
            float(row["Open"]) if pd.notna(row["Open"]) else None,
            float(row["High"]) if pd.notna(row["High"]) else None,
            float(row["Low"]) if pd.notna(row["Low"]) else None,
            float(row["Close"]),
            int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
            now,
        ))
    conn.executemany(
        """INSERT INTO intraday_prices (ticker, timeframe, ts, open, high, low, close, volume, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(ticker, timeframe, ts) DO UPDATE SET
             open = excluded.open, high = excluded.high, low = excluded.low,
             close = excluded.close, volume = excluded.volume, ingested_at = excluded.ingested_at""",
        rows,
    )
    conn.commit()
    return len(rows)


def load_bars(conn, ticker: str, timeframe: str, limit: int = None) -> pd.DataFrame:
    """Barras ordenadas cronologicamente, indexadas por ts (datetime UTC)."""
    sql = "SELECT ts, open, high, low, close, volume FROM intraday_prices WHERE ticker = ? AND timeframe = ? ORDER BY ts"
    params = [ticker, timeframe]
    if limit:
        sql = (
            "SELECT * FROM (SELECT ts, open, high, low, close, volume FROM intraday_prices "
            "WHERE ticker = ? AND timeframe = ? ORDER BY ts DESC LIMIT ?) ORDER BY ts"
        )
        params = [ticker, timeframe, limit]
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    return df.set_index("ts")


def ingest_all(conn, timeframes=None, tickers=None) -> dict:
    timeframes = timeframes or list(config.PATTERN_TIMEFRAMES)
    tickers = tickers or config.PATTERN_TICKERS
    results = {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for timeframe in timeframes:
        for ticker in tickers:
            try:
                df = fetch_intraday(ticker, timeframe)
                results[(ticker, timeframe)] = upsert_intraday(conn, ticker, timeframe, df) if not df.empty else 0
            except Exception as exc:
                database.log_run(conn, today, "ingest_intraday", "ERROR", f"{ticker}/{timeframe}: {exc}")
                results[(ticker, timeframe)] = None
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", choices=list(config.PATTERN_TIMEFRAMES), default=None)
    args = parser.parse_args()

    connection = database.get_connection()
    database.seed_watchlist(connection)
    tfs = [args.timeframe] if args.timeframe else None
    outcome = ingest_all(connection, timeframes=tfs)
    for (ticker, tf), n in sorted(outcome.items()):
        print(f"  {ticker:6s} {tf:3s}: {n if n is not None else 'FAILED'} bars")
    failed = [k for k, v in outcome.items() if v is None]
    database.log_run(
        connection, datetime.now(timezone.utc).strftime("%Y-%m-%d"), "ingest_intraday",
        "OK" if not failed else "WARN", f"ok={len(outcome) - len(failed)} failed={len(failed)}",
    )
    connection.close()
