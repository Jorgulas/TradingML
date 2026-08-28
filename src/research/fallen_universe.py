"""As accoes que CAIRAM do universo -- e o que o yfinance ja' nao serve.

Serve a experiencia K. O painel da ronda 7 tem 158 grandes capitalizacoes de
HOJE, todas com historico completo e todas vivas no ultimo dia. Zero atrito em
tres anos. Um indice real perde 5 a 8% dos membros por ano.

DUAS DESCOBERTAS SOBRE A PROPRIA FONTE DE DADOS, e a primeira e' mais grave do
que a lista de tickers estar mal escolhida:

1. O yfinance APAGA as empresas delistadas. De 20 candidatas que sabidamente
   sairam do S&P 500 neste periodo por falencia ou aquisicao -- SIVB, FRC,
   ATVI, PXD, HES, MRO, CTLT, JNPR, WBA, DISH, RAD, YELL, WE, FSR, NKLA, MPW,
   SAVE, TUP, BIG -- NENHUMA devolve dados. Nao e' a minha lista que esta'
   enviesada: e' a fonte. Nenhuma reconstrucao completa e' possivel aqui, e
   qualquer conclusao tem de o dizer.

2. Sobram as que CAIRAM mas continuam a negociar: sairam do indice por
   decadencia em vez de por morte. Sao 62, e sao recuperaveis. Nao substituem
   as falidas, mas cobrem exactamente o tipo de accao que falta -- a que teve
   um mau periodo prolongado -- e sao muito mais do que as ~15 que um indice
   real teria perdido.

FILTRO DE LIQUIDEZ, e porque nao e' batota. SBNY negoceia a 0,75 dolares com
volume praticamente nulo (e' a casca do banco falido, com o ticker
reatribuido); BBBY tem 29 dias porque outra empresa herdou o simbolo. Ninguem
vende a descoberto uma accao sem volume, e inclui-las seria fabricar retornos
impossiveis de realizar. O filtro aplica-se de forma IDENTICA a todo o universo,
incluindo as 158 originais, e reporta-se o resultado com e sem ele.
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

# Empresas que sairam do universo de grande capitalizacao entre 2023 e 2026 e
# que o yfinance ainda serve. Cobrem decadencia industrial (VFC, WBA-like),
# desastres sectoriais (SEDG, ENPH, PLUG, CHPT), colapsos pos-euforia (CVNA,
# PTON, ZM, DOCU, ROKU, TDOC) e bancos regionais castigados (ZION, LNC).
FALLEN = [
    "VFC", "AAP", "NWL", "LNC", "DXC", "ZION", "ALK", "SEDG", "ENPH", "PARA",
    "LUMN", "FMC", "CE", "MKTX", "ETSY", "BIO", "XRAY", "QRVO", "MHK", "APA",
    "IVZ", "HII", "CZR", "BWA", "WHR", "HSIC", "CRL", "TFX", "PNW", "NCLH",
    "AAL", "HAS", "GNRC", "MTCH", "ILMN", "MRNA", "CHRW", "OGN", "VTRS", "TAP",
    "CPB", "SJM", "DG", "DLTR", "ULTA", "SPWR", "PLUG", "CVNA", "AMC", "LCID",
    "CHPT", "RIVN", "BYND", "PTON", "WOLF", "AZTA", "TDOC", "ZM", "DOCU", "ROKU",
    "SBNY", "BBBY",
]

# Sabidamente removidas, e sem dados nenhuns no yfinance. Ficam registadas
# porque a ausencia delas E' o resultado: e' o tamanho do buraco que nao se
# consegue tapar.
DELISTED_NO_DATA = [
    "SIVB", "FRC", "ATVI", "PXD", "HES", "MRO", "CTLT", "JNPR", "WBA", "DISH",
    "RAD", "YELL", "WE", "FSR", "NKLA", "MPW", "SAVE", "TUP", "BIG",
]

START = "2023-09-01"
MIN_DOLLAR_VOLUME = 5_000_000.0   # volume-dolar mediano diario
MIN_PRICE = 2.0                   # preco mediano


def fetch_daily(ticker: str) -> pd.DataFrame:
    frame = yf.Ticker(ticker).history(start=START, interval="1d", auto_adjust=False)
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                  "Close": "close", "Volume": "volume"})
    return frame[["open", "high", "low", "close", "volume"]].dropna()


def register(conn, ticker: str) -> None:
    """A tabela intraday_prices tem chave estrangeira para watchlist. As caidas
    entram com active=0: existem para esta experiencia, nao sao transaccionadas
    nem avaliadas em noticias."""
    # `added_date` e' NOT NULL. Sem ele o INSERT OR IGNORE engole a violacao
    # em silencio, o ticker nunca chega a existir, e o erro so' aparece depois
    # como chave estrangeira falhada na tabela de precos -- longe da causa.
    conn.execute(
        "INSERT OR IGNORE INTO watchlist (ticker, name, sector, active, added_date) "
        "VALUES (?, ?, ?, 0, ?)",
        (ticker, ticker, "Fallen", datetime.now(timezone.utc).date().isoformat()))


def store_daily(conn, ticker: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    register(conn, ticker)
    now = datetime.now(timezone.utc).isoformat()
    rows = [(ticker, "1d", index.tz_convert("UTC").isoformat()
             if index.tzinfo else index.tz_localize("UTC").isoformat(),
             float(r["open"]), float(r["high"]), float(r["low"]),
             float(r["close"]), float(r["volume"]), now)
            for index, r in frame.iterrows()]
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


def liquidity(conn, ticker: str) -> dict:
    rows = conn.execute(
        "SELECT close, volume FROM intraday_prices WHERE ticker = ? AND timeframe = '1d'",
        (ticker,)).fetchall()
    if not rows:
        return {"n": 0, "price": 0.0, "dollar_volume": 0.0, "liquid": False}
    frame = pd.DataFrame([dict(r) for r in rows])
    price = float(frame["close"].median())
    dollar_volume = float((frame["close"] * frame["volume"]).median())
    return {
        "n": len(frame), "price": price, "dollar_volume": dollar_volume,
        "liquid": price >= MIN_PRICE and dollar_volume >= MIN_DOLLAR_VOLUME,
    }


def ingest(conn, tickers=None) -> dict:
    """Barras diarias para todo o universo alargado.

    Diarias e nao horarias: o yfinance so' da' 730 dias de barras de 1h, o que
    nao chega para 2023. E como todas as accoes passam a vir da mesma fonte e
    da mesma granularidade, a comparacao entre universos fica limpa."""
    tickers = tickers or (config.TREE_TICKERS + [config.BENCHMARK["ticker"]] + FALLEN)
    out = {}
    for ticker in tickers:
        try:
            out[ticker] = store_daily(conn, ticker, fetch_daily(ticker))
        except Exception as exc:
            out[ticker] = 0
            print(f"  {ticker}: falhou ({str(exc)[:60]})")
        time.sleep(0.05)
    return out


def surviving_fallen(conn) -> list:
    """As caidas que passam o filtro de liquidez."""
    return [t for t in FALLEN if liquidity(conn, t)["liquid"]]


if __name__ == "__main__":
    connection = database.get_connection()
    print(f"a buscar barras diarias para {len(config.TREE_TICKERS) + 1 + len(FALLEN)} "
          f"tickers desde {START}...")
    result = ingest(connection)
    ok = sum(1 for v in result.values() if v)
    print(f"{ok}/{len(result)} tickers com dados; {sum(result.values()):,} barras")

    print()
    print("LIQUIDEZ das accoes caidas (filtro identico ao aplicado ao resto):")
    print(f"  {'ticker':>8} {'dias':>6} {'preco':>10} {'volume-$':>13}  passa?")
    for ticker in FALLEN:
        info = liquidity(connection, ticker)
        mark = "sim" if info["liquid"] else "NAO"
        print(f"  {ticker:>8} {info['n']:>6} ${info['price']:>9.2f} "
              f"${info['dollar_volume'] / 1e6:>11.1f}M  {mark}")

    kept = surviving_fallen(connection)
    print()
    print(f"{len(kept)} de {len(FALLEN)} caidas passam o filtro")
    print(f"{len(DELISTED_NO_DATA)} delistadas sem dados nenhuns -- buraco impossivel de tapar")
    connection.close()
