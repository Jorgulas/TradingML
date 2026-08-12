"""Features tecnicas (calculadas dos precos) + montagem do vetor de input do
modelo por horizonte (subset tecnico + agregacao de noticias na janela do
horizonte). Ver config.HORIZON_PARAMS para as colunas/janela de cada horizonte.

Anti-leakage: todo o dataframe de indicadores e' calculado "as of" a propria
sessao e so' DEPOIS deslocado 1 linha (.shift(1)). Isto garante, de forma
uniforme para todas as colunas, que a linha rotulada com a data D so' contem
informacao conhecida ate' ao fecho de D-1 -- em vez de ter de raciocinar sobre
leakage formula a formula.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from db import database

import numpy as np
import pandas as pd

TECH_COLUMNS = [
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "sma5_ratio", "sma20_ratio", "sma50_ratio", "sma200_ratio",
    "sma_cross_short", "sma_cross_long",
    "rsi14", "rsi50", "vol_20d", "vol_60d", "volume_zscore_20d",
]


def _sma_ratio(close: pd.Series, window: int) -> pd.Series:
    sma = close.rolling(window, min_periods=window).mean()
    return close / sma - 1


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    flat = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~flat, 50.0)
    return rsi


def compute_technical_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """prices_df: indexado por data (ascendente), colunas Close/Volume.
    Devolve um DataFrame indexado por data onde a linha de `date` usa so'
    informacao disponivel ate' date-1 (ver nota anti-leakage no topo do ficheiro)."""
    close = prices_df["Close"]
    volume = prices_df["Volume"]

    raw = pd.DataFrame(index=prices_df.index)
    raw["ret_1d"] = close.pct_change(1)
    raw["ret_5d"] = close.pct_change(5)
    raw["ret_20d"] = close.pct_change(20)
    raw["ret_60d"] = close.pct_change(60)
    raw["sma5_ratio"] = _sma_ratio(close, 5)
    raw["sma20_ratio"] = _sma_ratio(close, 20)
    raw["sma50_ratio"] = _sma_ratio(close, 50)
    raw["sma200_ratio"] = _sma_ratio(close, 200)

    sma5 = close.rolling(5, min_periods=5).mean()
    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    raw["sma_cross_short"] = sma5 / sma20 - 1
    raw["sma_cross_long"] = sma50 / sma200 - 1

    raw["rsi14"] = _rsi(close, 14)
    raw["rsi50"] = _rsi(close, 50)

    daily_ret = close.pct_change()
    raw["vol_20d"] = daily_ret.rolling(20, min_periods=20).std()
    raw["vol_60d"] = daily_ret.rolling(60, min_periods=60).std()

    vol_mean_20 = volume.rolling(20, min_periods=20).mean()
    vol_std_20 = volume.rolling(20, min_periods=20).std()
    raw["volume_zscore_20d"] = (volume - vol_mean_20) / vol_std_20.replace(0, np.nan)

    return raw.shift(1)


def load_price_series(conn, ticker: str) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT date, close, volume FROM prices WHERE ticker = ? ORDER BY date", (ticker,)
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").rename(columns={"close": "Close", "volume": "Volume"})
    return df


def upsert_technical_features(conn, ticker: str, features_df: pd.DataFrame) -> int:
    now = datetime.now(timezone.utc).isoformat()
    col_list = ", ".join(TECH_COLUMNS)
    placeholders = ", ".join("?" for _ in TECH_COLUMNS)
    update_clause = ", ".join(f"{c} = excluded.{c}" for c in TECH_COLUMNS)
    sql = (
        f"INSERT INTO technical_features (ticker, date, {col_list}, computed_at) "
        f"VALUES (?, ?, {placeholders}, ?) "
        f"ON CONFLICT(ticker, date) DO UPDATE SET {update_clause}, computed_at = excluded.computed_at"
    )
    n = 0
    for date_idx, row in features_df.iterrows():
        date_str = date_idx.strftime("%Y-%m-%d")
        values = [None if pd.isna(row[c]) else float(row[c]) for c in TECH_COLUMNS]
        conn.execute(sql, [ticker, date_str, *values, now])
        n += 1
    conn.commit()
    return n


def compute_and_store_technical_features(conn, ticker: str) -> int:
    prices_df = load_price_series(conn, ticker)
    if prices_df.empty:
        return 0
    feats = compute_technical_features(prices_df).dropna(how="all")
    if feats.empty:
        return 0
    return upsert_technical_features(conn, ticker, feats)


def recompute_all_technical_features(conn, tickers=None) -> dict:
    tickers = tickers or config.TICKERS
    return {t: compute_and_store_technical_features(conn, t) for t in tickers}


def get_news_aggregate(conn, ticker: str, date: str, window_days: int) -> dict:
    """Fracao de dias (0..1) em que cada boolean foi True nos ultimos
    `window_days` dias com avaliacao <= date (SHORT usa window_days=1, ou
    seja, reduz-se naturalmente a 'so hoje'). Sem avaliacoes -> tudo 0.0,
    distinto na BD (news_features nao tem linha) mas igual do ponto de vista
    do modelo, por design (ver plano)."""
    placeholders = ", ".join(config.BOOLEAN_FEATURES)
    rows = conn.execute(
        f"""SELECT {placeholders} FROM news_features
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC LIMIT ?""",
        (ticker, date, window_days),
    ).fetchall()
    if not rows:
        return {feat: 0.0 for feat in config.BOOLEAN_FEATURES}
    return {
        feat: sum(r[feat] for r in rows) / len(rows)
        for feat in config.BOOLEAN_FEATURES
    }


def feature_names(horizon: str) -> list:
    return list(config.HORIZON_PARAMS[horizon]["technical_columns"]) + list(config.BOOLEAN_FEATURES)


def build_feature_vector(conn, ticker: str, date: str, horizon: str):
    """Devolve dict {nome_feature: valor} para (ticker, date, horizon), ou
    None se faltar alguma coluna tecnica exigida por este horizonte (janela
    de historico ainda insuficiente nesta data -- nunca imputado)."""
    params = config.HORIZON_PARAMS[horizon]
    tech_row = conn.execute(
        "SELECT * FROM technical_features WHERE ticker = ? AND date = ?", (ticker, date)
    ).fetchone()
    if tech_row is None:
        return None
    vector = {}
    for col in params["technical_columns"]:
        val = tech_row[col]
        if val is None:
            return None
        vector[col] = val
    vector.update(get_news_aggregate(conn, ticker, date, params["news_window_days"]))
    return vector


if __name__ == "__main__":
    connection = database.get_connection()
    result = recompute_all_technical_features(connection)
    for ticker, n in result.items():
        print(f"  {ticker}: {n} rows")
    print(f"technical_features recomputed for {len(result)} tickers.")
    connection.close()
