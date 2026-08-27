"""Calendario de resultados trimestrais -- o dado que faltava.

A ronda 3 concluiu que o "lucro" do sinal de magnitude vinha de ~5% de
movimentos extremos, e levantou a hipotese de serem eventos agendados
(resultados) que a volatilidade implicita real ja' cobra mas o meu movimento
esperado (media movel de 60 dias) desconhece.

Era uma hipotese por testar, por falta de dados. Com o calendario de resultados
passa a ser testavel, e abre tres perguntas:

  A. o ganho aparente concentra-se mesmo em dias de resultados?
  B. saber as datas melhora a previsao de risco de cauda?
  C. a MAGNITUDE da reacao aos resultados e' previsivel entre empresas?

A pergunta C e' a unica com hipotese de valor real. O mercado de opcoes sabe
obviamente quando ha resultados -- isso esta' no preco. O que pode nao estar
completamente e' QUANTO cada empresa reage.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database

import pandas as pd
import yfinance as yf

SCHEMA = """
CREATE TABLE IF NOT EXISTS earnings_dates (
    ticker         TEXT NOT NULL,
    earnings_ts    TEXT NOT NULL,
    eps_estimate   REAL,
    eps_reported   REAL,
    surprise_pct   REAL,
    fetched_at     TEXT NOT NULL,
    PRIMARY KEY (ticker, earnings_ts)
);
CREATE INDEX IF NOT EXISTS idx_earnings_ticker ON earnings_dates(ticker, earnings_ts);
"""


def fetch_for(ticker: str, limit: int = 60) -> pd.DataFrame:
    frame = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    return frame if frame is not None else pd.DataFrame()


def store(conn, ticker: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for timestamp, row in frame.iterrows():
        stamp = timestamp.tz_convert("UTC") if timestamp.tzinfo else timestamp
        rows.append((
            ticker, stamp.isoformat(),
            float(row["EPS Estimate"]) if pd.notna(row.get("EPS Estimate")) else None,
            float(row["Reported EPS"]) if pd.notna(row.get("Reported EPS")) else None,
            float(row["Surprise(%)"]) if pd.notna(row.get("Surprise(%)")) else None,
            now,
        ))
    conn.executemany(
        """INSERT INTO earnings_dates
             (ticker, earnings_ts, eps_estimate, eps_reported, surprise_pct, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(ticker, earnings_ts) DO UPDATE SET
             eps_estimate = excluded.eps_estimate,
             eps_reported = excluded.eps_reported,
             surprise_pct = excluded.surprise_pct,
             fetched_at = excluded.fetched_at""",
        rows,
    )
    conn.commit()
    return len(rows)


def fetch_all(conn, tickers=None) -> dict:
    tickers = tickers or config.TREE_TICKERS
    conn.executescript(SCHEMA)
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = store(conn, ticker, fetch_for(ticker))
        except Exception as exc:
            results[ticker] = None
            print(f"  {ticker}: falhou ({str(exc)[:70]})")
        time.sleep(0.15)   # cortesia para com a API, evita bloqueios
    return results


def load(conn, tickers=None) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT ticker, earnings_ts, eps_estimate, surprise_pct FROM earnings_dates"
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(r) for r in rows])
    frame["earnings_ts"] = pd.to_datetime(frame["earnings_ts"], utc=True, format="ISO8601")
    frame["earnings_date"] = frame["earnings_ts"].dt.normalize()
    if tickers:
        frame = frame[frame["ticker"].isin(tickers)]
    return frame.sort_values(["ticker", "earnings_ts"]).reset_index(drop=True)


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = fetch_all(connection)
    ok = sum(1 for v in outcome.values() if v)
    total = sum(v for v in outcome.values() if v)
    print(f"\n{ok}/{len(outcome)} tickers com datas de resultados; {total:,} registos")
    frame = load(connection)
    if not frame.empty:
        print(f"intervalo: {frame['earnings_date'].min().date()} a "
              f"{frame['earnings_date'].max().date()}")
        print(f"media de {len(frame) / frame['ticker'].nunique():.1f} datas por ticker")
    connection.close()
