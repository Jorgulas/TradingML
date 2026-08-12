"""Features de contexto de um padrao, para o classificador multiclasse.

Porque nao se usa directamente o `meta_json` dos detectores: cada tipo de
padrao guarda campos diferentes (um triangulo tem slope_high/slope_low, um
duplo topo tem level_diff/excursion, uma chavena tem depth/rim_diff). Juntar
tudo daria uma matriz quase toda vazia, com colunas que so' existem para 3%
das linhas. Em vez disso extraem-se features UNIVERSAIS, calculaveis para
qualquer padrao a partir das barras -- incluindo volume, que ate' aqui era
ingerido e nunca usado.

Anti-leakage: tudo o que entra aqui e' conhecido no momento em que o padrao
termina. As features de tendencia/volatilidade olham para as barras ANTES do
padrao comecar; as do proprio padrao usam o intervalo [start, end], que ja'
aconteceu. O rotulo (o padrao seguinte) e' estritamente futuro.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "quality",             # confianca geometrica da deteccao (0-1)
    "n_pivots",            # pontos de viragem que a sustentam
    "duration_bars",       # quanto tempo demorou a formar-se
    "pattern_return",      # variacao do preco ao longo do padrao
    "range_pct",           # amplitude (max-min) relativa dentro do padrao
    "prior_trend_20",      # retorno nas 20 barras ANTES do padrao
    "prior_trend_60",      # retorno nas 60 barras antes
    "prior_volatility",    # desvio-padrao dos retornos antes do padrao
    "volume_ratio",        # volume medio no padrao / volume medio antes
    "session_position",    # 0=abertura, 1=fecho da sessao (importa a 5m)
]

FEATURE_NAMES = NUMERIC_FEATURES + [f"is_{p}" for p in config.PATTERN_TYPES]


def _safe_return(series, start: int, end: int) -> float:
    if start < 0 or end >= len(series) or start >= end:
        return 0.0
    first, last = float(series[start]), float(series[end])
    return last / first - 1 if first else 0.0


def pattern_context_features(bars: pd.DataFrame, start_idx: int, end_idx: int,
                             pattern_type: str, quality: float, n_pivots: int) -> dict:
    """Vector de features para um padrao. Usado tanto no treino (sobre padroes
    historicos) como em tempo real (sobre o padrao actual) -- e' a mesma
    funcao nos dois lados, para nao haver hipotese de divergirem."""
    closes = bars["close"].values
    highs = bars["high"].values
    lows = bars["low"].values
    volumes = bars["volume"].values
    n = len(closes)
    start_idx = max(0, min(start_idx, n - 1))
    end_idx = max(start_idx, min(end_idx, n - 1))

    window_closes = closes[start_idx:end_idx + 1]
    window_high = float(np.max(highs[start_idx:end_idx + 1]))
    window_low = float(np.min(lows[start_idx:end_idx + 1]))
    mean_close = float(np.mean(window_closes)) if len(window_closes) else 1.0

    prior_start_20 = max(0, start_idx - 20)
    prior_start_60 = max(0, start_idx - 60)
    prior_closes = closes[prior_start_20:start_idx]
    prior_returns = np.diff(prior_closes) / prior_closes[:-1] if len(prior_closes) > 1 else np.array([0.0])

    prior_volumes = volumes[prior_start_20:start_idx]
    mean_prior_volume = float(np.mean(prior_volumes)) if len(prior_volumes) else 0.0
    mean_window_volume = float(np.mean(volumes[start_idx:end_idx + 1])) if end_idx >= start_idx else 0.0

    # Posicao na sessao: fraccao das barras desse dia ja' decorridas quando o
    # padrao terminou. Calculado por contagem de barras e nao por hora do
    # relogio, para nao depender de fusos nem de horario de verao.
    end_ts = bars.index[end_idx]
    same_day = bars.index.normalize() == end_ts.normalize()
    day_positions = np.flatnonzero(same_day)
    if len(day_positions) > 1:
        session_position = float(
            (end_idx - day_positions[0]) / (day_positions[-1] - day_positions[0])
        )
    else:
        session_position = 0.5

    features = {
        "quality": float(quality),
        "n_pivots": float(n_pivots),
        "duration_bars": float(end_idx - start_idx),
        "pattern_return": _safe_return(closes, start_idx, end_idx),
        "range_pct": (window_high - window_low) / mean_close if mean_close else 0.0,
        "prior_trend_20": _safe_return(closes, prior_start_20, start_idx),
        "prior_trend_60": _safe_return(closes, prior_start_60, start_idx),
        "prior_volatility": float(np.std(prior_returns)) if len(prior_returns) else 0.0,
        "volume_ratio": (mean_window_volume / mean_prior_volume) if mean_prior_volume else 1.0,
        "session_position": session_position,
    }
    for known in config.PATTERN_TYPES:
        features[f"is_{known}"] = 1.0 if known == pattern_type else 0.0
    return features


def build_dataset(conn, timeframe: str, tickers=None) -> pd.DataFrame:
    """Constroi (features do padrao i) -> (tipo do padrao i+1) por ticker.

    A sequencia e' por ticker e nunca se emenda o ultimo padrao de um ticker
    com o primeiro do seguinte. O ultimo padrao de cada ticker fica sem
    rotulo e e' descartado do treino (ainda nao ha padrao seguinte)."""
    from src.patterns.ingest_intraday import load_bars

    tickers = tickers or config.PATTERN_TICKERS
    records = []

    for ticker in tickers:
        rows = conn.execute(
            """SELECT pattern_type, start_ts, end_ts, confirmed_ts, quality, meta_json
               FROM detected_patterns WHERE ticker = ? AND timeframe = ?
               ORDER BY confirmed_ts, end_ts""",
            (ticker, timeframe),
        ).fetchall()
        if len(rows) < 2:
            continue

        bars = load_bars(conn, ticker, timeframe)
        if bars.empty:
            continue
        position_of = {ts: i for i, ts in enumerate(bars.index)}

        for current, following in zip(rows, rows[1:]):
            start_idx = position_of.get(pd.Timestamp(current["start_ts"]))
            end_idx = position_of.get(pd.Timestamp(current["end_ts"]))
            if start_idx is None or end_idx is None:
                continue
            meta = json.loads(current["meta_json"] or "{}")
            features = pattern_context_features(
                bars, start_idx, end_idx, current["pattern_type"],
                current["quality"], meta.get("n_pivots", 3),
            )
            features["ticker"] = ticker
            features["confirmed_ts"] = current["confirmed_ts"]
            features["from_pattern"] = current["pattern_type"]
            features["next_pattern"] = following["pattern_type"]
            records.append(features)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["confirmed_ts", "ticker"]).reset_index(drop=True)
