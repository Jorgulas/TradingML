"""Historico diario COMPLETO -- ate' 65 anos por accao.

PORQUE ISTO MUDA O PROJECTO. A ronda 14 acabou em "seriam precisos 13 a 42 anos
de historico" e eu tinha tres. Nunca estendi os dados diarios porque o painel
nasceu de barras de 1 hora, e o yfinance so' da' 730 dias dessas. Mas em barras
DIARIAS da' desde 1962 para muitas destas empresas. E' a maior alavanca que
existe aqui e nunca lhe tinha tocado.

`auto_adjust=True`, e nao e' detalhe. Sobre 65 anos ha' dezenas de
desdobramentos por accao, e sem ajustamento cada um deles aparece como uma queda
de 50% num dia. A ronda 11 perdeu uma experiencia inteira por causa de UM ticker
com desdobramento mal aplicado; a 65 anos sem ajustamento seria tudo lixo.

O QUE O HISTORICO LONGO COMPRA, e o que NAO compra:

  COMPRA replicacao a serio. O modo de falhar deste projecto tem sido sempre o
  mesmo: encontrar alguma coisa e nao a conseguir replicar. Com seis decadas
  posso exigir que um efeito apareca nos anos 70, 80, 90, 2000, 2010 e 2020 --
  seis amostras com mercados, tecnologia, microestrutura e regulacao
  completamente diferentes. Isso e' uma fasquia que nenhum resultado deste
  projecto teve ainda de enfrentar.

  NAO COMPRA universo sem vies. Pelo contrario: AGRAVA-O. A lista e' das grandes
  capitalizacoes de HOJE, e uma empresa com 65 anos de historico que ainda esta'
  no indice e' um sobrevivente do tipo mais extremo que existe. A ronda 8 mediu
  o vies a tres anos e valia metade do efeito; a 65 anos vale muito mais. Toda a
  conclusao que dependa de comparar accoes entre si herda isso, e tem de o
  dizer.

Por isso o historico longo serve, sobretudo, para perguntas sobre a ESTRUTURA
DENTRO de cada serie -- como se comportam os extremos, se persistem, se a sua
forma muda -- onde o vies afecta que accoes se observam mas nao invalida a
relacao medida dentro de cada uma.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.research.fallen_universe import FALLEN

import pandas as pd
import yfinance as yf

TIMEFRAME = "1dfull"     # rotulo proprio: nao mexe nos dados das rondas 8 a 14


def fetch(ticker: str) -> pd.DataFrame:
    frame = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                  "Close": "close", "Volume": "volume"})
    return frame[["open", "high", "low", "close", "volume"]].dropna()


def store(conn, ticker: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO watchlist (ticker, name, sector, active, added_date) "
        "VALUES (?, ?, ?, 0, ?)",
        (ticker, ticker, "History", datetime.now(timezone.utc).date().isoformat()))

    rows = []
    for index, row in frame.iterrows():
        stamp = index.tz_convert("UTC") if index.tzinfo else index.tz_localize("UTC")
        rows.append((ticker, TIMEFRAME, stamp.isoformat(),
                     float(row["open"]), float(row["high"]), float(row["low"]),
                     float(row["close"]), float(row["volume"]), now))
    conn.executemany(
        """INSERT INTO intraday_prices
             (ticker, timeframe, ts, open, high, low, close, volume, ingested_at)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(ticker, timeframe, ts) DO UPDATE SET
             open=excluded.open, high=excluded.high, low=excluded.low,
             close=excluded.close, volume=excluded.volume""",
        rows)
    conn.commit()
    return len(rows)


def universe() -> list:
    """Tudo o que ha': os 161 das arvores, o indice, e as caidas da ronda 8."""
    names = list(config.TREE_TICKERS) + [config.BENCHMARK["ticker"]] + FALLEN
    seen, out = set(), []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def ingest(conn, tickers=None, verbose: bool = True) -> dict:
    tickers = tickers or universe()
    results = {}
    for position, ticker in enumerate(tickers, 1):
        try:
            results[ticker] = store(conn, ticker, fetch(ticker))
        except Exception as exc:
            results[ticker] = 0
            if verbose:
                print(f"  {ticker}: falhou ({str(exc)[:60]})", flush=True)
        if verbose and position % 25 == 0:
            print(f"  {position}/{len(tickers)} tickers, "
                  f"{sum(results.values()):,} barras", flush=True)
        time.sleep(0.05)
    return results


def coverage(conn) -> pd.DataFrame:
    """Quanto historico existe por decada. E' o que decide que perguntas se
    podem fazer -- uma decada com dez accoes nao suporta nada transversal."""
    rows = conn.execute(
        "SELECT ticker, substr(ts, 1, 4) AS ano, COUNT(*) AS n "
        "FROM intraday_prices WHERE timeframe = ? GROUP BY ticker, ano",
        (TIMEFRAME,)).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(r) for r in rows])
    frame["ano"] = frame["ano"].astype(int)
    frame["decada"] = (frame["ano"] // 10) * 10
    return (frame.groupby("decada")
            .agg(tickers=("ticker", "nunique"), barras=("n", "sum"),
                 anos=("ano", "nunique"))
            .reset_index())


if __name__ == "__main__":
    connection = database.get_connection()
    tickers = universe()
    print(f"a buscar historico COMPLETO para {len(tickers)} tickers "
          f"(auto_adjust=True)...")
    results = ingest(connection, tickers)

    ok = sum(1 for v in results.values() if v)
    total = sum(results.values())
    print()
    print(f"{ok}/{len(results)} tickers, {total:,} barras diarias")

    table = coverage(connection)
    if not table.empty:
        print()
        print("COBERTURA POR DECADA -- e' isto que decide que perguntas sao possiveis:")
        print(f"  {'decada':>8} {'tickers':>9} {'barras':>12} {'anos':>6}")
        for _, row in table.iterrows():
            print(f"  {int(row['decada']):>8} {int(row['tickers']):>9} "
                  f"{int(row['barras']):>12,} {int(row['anos']):>6}")
    connection.close()
