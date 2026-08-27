"""Dois conjuntos de features para as arvores de decisao, mais o rotulo de
tres classes (SOBE / DESCE / ESPERAR).

MODELO A -- indicadores tecnicos classicos extraidos do grafico: retornos a
varios desfasamentos, medias moveis, MACD, RSI, Bollinger, ATR, volume,
volatilidade. E' a leitura "de analista": o que os indicadores dizem agora.

MODELO B -- forma do percurso recente, sem indicador nenhum. Olha para as
ultimas N barras e descreve-as: quantos movimentos foram positivos e quantos
negativos, qual a inercia (comprimento da corrida actual no mesmo sentido), a
media e o desvio, e o desvio INICIAL vs FINAL da janela. Inclui tambem o sinal
de cada uma das N barras como feature propria, que e' a forma de dar a uma
arvore a sequencia "+ + - + - - + + + -" de que ela consegue ramificar.

Os dois veem a MESMA janela de tempo mas descrevem-na de maneiras diferentes,
por isso nao sao redundantes -- e a comparacao entre eles diz qual das duas
leituras carrega mais informacao.

Anti-leakage: todas as features na linha t usam apenas barras ate' t
(inclusive o fecho de t, que ja' e' conhecido quando t fecha). O rotulo usa
t+1..t+H, estritamente futuro.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

import numpy as np
import pandas as pd

T = config.TREES

# Classes. WAIT no meio de proposito: e' a decisao "o movimento previsto e'
# demasiado pequeno para valer a pena agir".
DOWN, WAIT, UP = 0, 1, 2
CLASS_NAMES = {DOWN: "DESCE", WAIT: "ESPERAR", UP: "SOBE"}


# --------------------------------------------------------------------------
# Indicadores auxiliares
# --------------------------------------------------------------------------

def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(window, min_periods=window).mean()


# --------------------------------------------------------------------------
# MODELO A -- indicadores tecnicos
# --------------------------------------------------------------------------

INDICATOR_FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
    "sma5_ratio", "sma10_ratio", "sma20_ratio", "sma50_ratio",
    "ema12_ratio", "ema26_ratio",
    "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "rsi7", "rsi14",
    "bb_position", "bb_width",
    "atr14_norm", "hl_range_norm",
    "vol_10", "vol_20", "vol_60",
    "volume_z20", "volume_ratio5",
    "stoch_14", "dist_from_high20", "dist_from_low20",
    "session_position",
]


def indicator_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por barra. Tudo calculado com dados ate' a propria barra."""
    close, high, low = bars["close"], bars["high"], bars["low"]
    volume = bars["volume"].astype(float)
    out = pd.DataFrame(index=bars.index)

    for lag in (1, 2, 3, 5, 10, 20):
        out[f"ret_{lag}"] = close.pct_change(lag)

    for window in (5, 10, 20, 50):
        out[f"sma{window}_ratio"] = close / close.rolling(window, min_periods=window).mean() - 1

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["ema12_ratio"] = close / ema12 - 1
    out["ema26_ratio"] = close / ema26 - 1

    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_norm"] = macd / close
    out["macd_signal_norm"] = macd_signal / close
    out["macd_hist_norm"] = (macd - macd_signal) / close

    out["rsi7"] = _rsi(close, 7)
    out["rsi14"] = _rsi(close, 14)

    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    out["bb_position"] = (close - sma20) / (2 * std20).replace(0, np.nan)
    out["bb_width"] = (4 * std20) / sma20

    out["atr14_norm"] = _atr(high, low, close, 14) / close
    out["hl_range_norm"] = (high - low) / close

    returns = close.pct_change()
    for window in (10, 20, 60):
        out[f"vol_{window}"] = returns.rolling(window, min_periods=window).std()

    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std()
    out["volume_z20"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
    out["volume_ratio5"] = volume / volume.rolling(5, min_periods=5).mean().replace(0, np.nan)

    high20 = high.rolling(20, min_periods=20).max()
    low20 = low.rolling(20, min_periods=20).min()
    span14_high = high.rolling(14, min_periods=14).max()
    span14_low = low.rolling(14, min_periods=14).min()
    out["stoch_14"] = (close - span14_low) / (span14_high - span14_low).replace(0, np.nan)
    out["dist_from_high20"] = close / high20 - 1
    out["dist_from_low20"] = close / low20 - 1

    day = bars.index.normalize()
    position = bars.groupby(day).cumcount()
    per_day = bars.groupby(day)["close"].transform("size")
    out["session_position"] = position / (per_day - 1).clip(lower=1)

    return out[INDICATOR_FEATURES]


# --------------------------------------------------------------------------
# MODELO B -- forma da janela recente
# --------------------------------------------------------------------------

def microstructure_feature_names(window: int = None) -> list:
    window = window or T["window_bars"]
    return [
        "n_up", "n_down", "up_ratio",
        "inertia",                # corrida actual no mesmo sentido, com sinal
        "longest_up_run", "longest_down_run",
        "mean_return", "std_return",
        "first_half_mean", "second_half_mean", "momentum_shift",
        "initial_deviation", "final_deviation",
        "slope", "acceleration",
        "cumulative_return", "position_in_range", "range_norm",
    ] + [f"sign_lag{i}" for i in range(1, window + 1)]


def microstructure_features(bars: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """Descreve a FORMA das ultimas `window` barras, sem indicadores.

    Implementado vectorizado (rolling/shift) e nao com um ciclo por linha:
    com ~120 mil linhas x 43 tickers o ciclo demoraria minutos e este demora
    segundos, e o resultado e' identico."""
    window = window or T["window_bars"]
    close = bars["close"]
    returns = close.pct_change()
    sign = np.sign(returns)

    out = pd.DataFrame(index=bars.index)

    positives = (sign > 0).astype(float)
    negatives = (sign < 0).astype(float)
    out["n_up"] = positives.rolling(window, min_periods=window).sum()
    out["n_down"] = negatives.rolling(window, min_periods=window).sum()
    out["up_ratio"] = out["n_up"] / window

    # Inercia: quantas barras seguidas no mesmo sentido, com o sinal desse
    # sentido. Conta-se olhando para tras enquanto o sinal nao muda.
    inertia = pd.Series(0.0, index=bars.index)
    for lag in range(1, window + 1):
        same = (sign.shift(lag - 1) == sign) & (sign != 0)
        inertia = inertia.where(~(same & (inertia == lag - 1)), lag)
    out["inertia"] = inertia * sign.fillna(0)

    out["longest_up_run"] = positives.rolling(window, min_periods=window).apply(_longest_run, raw=True)
    out["longest_down_run"] = negatives.rolling(window, min_periods=window).apply(_longest_run, raw=True)

    out["mean_return"] = returns.rolling(window, min_periods=window).mean()
    out["std_return"] = returns.rolling(window, min_periods=window).std()

    half = window // 2
    out["first_half_mean"] = returns.shift(half).rolling(window - half, min_periods=window - half).mean()
    out["second_half_mean"] = returns.rolling(half, min_periods=half).mean()
    out["momentum_shift"] = out["second_half_mean"] - out["first_half_mean"]

    window_mean = close.rolling(window, min_periods=window).mean()
    out["initial_deviation"] = close.shift(window - 1) / window_mean - 1
    out["final_deviation"] = close / window_mean - 1

    index_positions = np.arange(window, dtype=float)
    centred = index_positions - index_positions.mean()
    denominator = float((centred ** 2).sum())
    out["slope"] = close.rolling(window, min_periods=window).apply(
        lambda values: float(np.dot(values - values.mean(), centred) / denominator), raw=True
    ) / close
    out["acceleration"] = out["slope"].diff()

    out["cumulative_return"] = close / close.shift(window) - 1
    window_high = close.rolling(window, min_periods=window).max()
    window_low = close.rolling(window, min_periods=window).min()
    out["position_in_range"] = (close - window_low) / (window_high - window_low).replace(0, np.nan)
    out["range_norm"] = (window_high - window_low) / window_mean

    # A sequencia de sinais, uma feature por desfasamento: e' assim que se da'
    # a uma arvore o "+ + - + - -" para ela poder ramificar em cada posicao.
    for lag in range(1, window + 1):
        out[f"sign_lag{lag}"] = sign.shift(lag - 1)

    return out[microstructure_feature_names(window)]


def _longest_run(values: np.ndarray) -> float:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best)


# --------------------------------------------------------------------------
# Rotulo de tres classes
# --------------------------------------------------------------------------

def label_three_class(bars: pd.DataFrame, horizon: int = None, band_k: float = None) -> pd.DataFrame:
    """SOBE / DESCE / ESPERAR sobre as proximas `horizon` barras.

    ESPERAR nao e' uma classe de conveniencia -- e' a decisao correcta quando
    o movimento previsto e' pequeno de mais para valer a pena agir. A banda e'
    escalada pela volatilidade recente e nao fixa, porque 5 pontos base sao
    muito num minuto calmo e nada num minuto agitado, e uma banda fixa daria
    quase so' ESPERAR nuns tickers e quase nenhum noutros."""
    horizon = horizon or T["horizon_bars"]
    band_k = T["band_k"] if band_k is None else band_k

    close = bars["close"]
    forward_return = close.shift(-horizon) / close - 1
    volatility = close.pct_change().rolling(T["band_vol_window"],
                                            min_periods=T["band_vol_window"]).std()
    band = band_k * volatility * np.sqrt(horizon)

    label = pd.Series(np.full(len(bars), float(WAIT)), index=bars.index)
    label[forward_return > band] = UP
    label[forward_return < -band] = DOWN
    label[forward_return.isna() | band.isna()] = np.nan

    return pd.DataFrame({
        "label": label,
        "forward_return": forward_return,
        "band": band,
    }, index=bars.index)
