"""Grelha diaria comum a todos os algoritmos, com o alvo neutro ao mercado.

Porque uma grelha comum. Os algoritmos deste projecto vivem em relogios
diferentes -- arvores a 1 minuto, padroes a 1 hora, magnitude a 1 dia -- e por
isso nunca puderam ser comparados nem combinados. A tabela de verdade precisa
de um instante unico onde TODOS tenham opiniao ao mesmo tempo. Esse instante e'
o fecho de cada dia: os sinais de barras intradiarias leem-se na ultima barra
do dia, e os de barras diarias na propria barra.

O ALVO e' o retorno futuro em EXCESSO sobre o SPY. Nao e' preciosismo: a 20
barras, 53.7% das janelas deste universo sao de subida apenas porque o mercado
subiu. Uma tabela de verdade treinada no retorno bruto aprenderia essa deriva e
nada mais, e pareceria boa a fazer isso.

FEATURES TRANSVERSAIS (novas neste modulo). Todo o resto do projecto olhou para
cada accao isoladamente ao longo do tempo, e a direccao temporal mostrou-se
imprevisivel em seis rondas de investigacao. A informacao que nunca foi testada
e' a COMPARACAO entre accoes no mesmo dia: nao "a AAPL subiu 2%", mas "a AAPL
subiu mais do que 80% do universo hoje". E' market-neutral por construcao --
o factor comum a todas as accoes desaparece ao ordenar -- e e' precisamente o
tipo de sinal que pode sobreviver quando o temporal nao sobrevive.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.patterns.ingest_intraday import load_bars
from src.research.daily_horizon import daily_har_features, to_daily, weekday_dummies
from src.trees import features as F

import numpy as np
import pandas as pd

D = config.DECISION

DOWN, WAIT, UP = 0, 1, 2
CLASS_NAMES = {DOWN: "DESCE", WAIT: "ESPERAR", UP: "SOBE"}

# Features que fazem sentido ordenar transversalmente. Sao todas ja'
# normalizadas ao preco, portanto comparaveis entre accoes de precos diferentes.
CROSS_SECTIONAL_BASE = [
    "ret_1", "ret_5", "ret_20",
    "rel_strength_20", "excess_ret_5", "excess_ret_20",
    "vol_20", "volume_z20", "rsi14",
    "dist_from_high20", "dist_from_low20",
]
CROSS_SECTIONAL_FEATURES = [f"xs_{name}" for name in CROSS_SECTIONAL_BASE]


def label_horizon(bars: pd.DataFrame, benchmark: pd.DataFrame, horizon: int,
                  market_neutral: bool = True) -> pd.DataFrame:
    """Retorno futuro (em excesso, se pedido) e a classe de tres valores.

    A banda de indiferenca escala com a volatilidade recente vezes raiz do
    horizonte. Fixa-la seria dar quase so' ESPERAR nas accoes calmas e quase
    nenhum nas agitadas."""
    close = bars["close"]
    forward = close.shift(-horizon) / close - 1

    if market_neutral and benchmark is not None and not benchmark.empty:
        market = benchmark["close"].reindex(bars.index).ffill()
        forward = forward - (market.shift(-horizon) / market - 1)

    volatility = close.pct_change().rolling(D["vol_window"],
                                            min_periods=D["vol_window"]).std()
    band = D["band_k"] * volatility * np.sqrt(horizon)

    label = pd.Series(np.full(len(bars), float(WAIT)), index=bars.index)
    label[forward > band] = UP
    label[forward < -band] = DOWN
    label[forward.isna() | band.isna()] = np.nan

    return pd.DataFrame({"forward_return": forward, "label": label, "band": band},
                        index=bars.index)


def add_cross_sectional(frame: pd.DataFrame) -> pd.DataFrame:
    """Percentil de cada accao dentro do proprio dia, para cada feature base.

    `rank(pct=True)` sobre o grupo do dia: 0 = a mais baixa do universo naquele
    dia, 1 = a mais alta. Nao olha para o futuro (so' usa valores do proprio
    instante) e e' invariante ao nivel geral do mercado nesse dia."""
    frame = frame.copy()
    for name in CROSS_SECTIONAL_BASE:
        if name in frame.columns:
            frame[f"xs_{name}"] = frame.groupby("ts")[name].rank(pct=True)
        else:
            frame[f"xs_{name}"] = np.nan
    return frame


def _daily_bars(conn, ticker: str, timeframe: str) -> pd.DataFrame:
    """Barras diarias, venham elas ja' diarias ou agregadas de 1h.

    O timeframe "1d" existe para a experiencia de sobrevivencia: o yfinance so'
    da' 730 dias de barras horarias, o que nao chega para empresas que sairam do
    universo em 2023-2024. Em barras diarias o historico vai muito mais atras."""
    bars = load_bars(conn, ticker, timeframe)
    return bars if timeframe == "1d" else to_daily(bars)


MAX_PLAUSIBLE_DAILY_RETURN = 1.0   # 100% num dia


def is_corrupt(bars: pd.DataFrame) -> bool:
    """A serie contem movimentos que nenhuma accao faz de verdade?

    Nao e' zelo abstracto -- foi assim que se perdeu uma experiencia inteira.
    Ao alargar o universo para testar o vies de sobrevivencia, a BYND entrou
    com um desdobramento inverso aplicado de forma inconsistente pela fonte:
    oscilava entre 0,55 e 17,85 dolares em dias seguidos, seis vezes, o que da'
    retornos de +2900% seguidos de -97%. Esse unico ticker produziu 99% do
    colapso que quase me levou a concluir que o modelo nao generalizava.

    Depois de ajustada para desdobramentos e dividendos, uma variacao diaria
    acima de 100% numa accao liquida americana e' praticamente sempre um erro de
    ajustamento. Excluir por engano um evento genuino custa uma accao entre
    duzentas; deixar passar uma serie corrompida custa a medicao toda."""
    returns = bars["close"].pct_change().abs()
    return bool((returns > MAX_PLAUSIBLE_DAILY_RETURN).any())


def build(conn, tickers=None, min_days: int = 250,
          timeframe: str = "1h", excluded: list = None) -> pd.DataFrame:
    """Painel diario: uma linha por (ticker, dia) com features e os dois alvos.

    `excluded`, se for passada uma lista, recebe os tickers rejeitados por
    dados impossiveis -- para quem chama poder reporta-los em vez de os perder
    em silencio."""
    tickers = tickers or config.TREE_TICKERS
    benchmark = _daily_bars(conn, config.BENCHMARK["ticker"], timeframe)

    horizon = D["horizon_days"]
    fast = D["fast_horizon_days"]
    neutral = D["market_neutral"]

    frames = []
    for ticker in tickers:
        bars = _daily_bars(conn, ticker, timeframe)
        if bars.empty or len(bars) < min_days:
            continue
        if is_corrupt(bars):
            if excluded is not None:
                excluded.append(ticker)
            continue

        matrix = (F.indicator_features(bars)
                  .join(F.market_context_features(bars, benchmark))
                  .join(daily_har_features(bars))
                  .join(weekday_dummies(bars)))
        # Constante em barras diarias (uma barra por dia), so' acrescenta ruido.
        matrix = matrix.drop(columns=["session_position"], errors="ignore")

        slow_label = label_horizon(bars, benchmark, horizon, neutral)
        fast_label = label_horizon(bars, benchmark, fast, neutral)
        matrix["forward_return"] = slow_label["forward_return"]
        matrix["label"] = slow_label["label"]
        matrix["band"] = slow_label["band"]
        matrix["fast_forward_return"] = fast_label["forward_return"]
        matrix["fast_label"] = fast_label["label"]
        matrix["abs_return"] = slow_label["forward_return"].abs()

        matrix["ticker"] = ticker
        matrix["ts"] = matrix.index
        # Sem isto o sort_values fica ambiguo entre nivel de indice e coluna.
        matrix.index.name = None

        # So' as FEATURES sao obrigatorias. As ultimas `horizon` barras ficam
        # sem rotulo -- o futuro delas ainda nao aconteceu -- e sao exactamente
        # as barras sobre as quais ha' uma decisao para tomar hoje. Deita-las
        # fora aqui obrigaria a pagina a mostrar o veredicto de ha' uma semana
        # como se fosse o de agora.
        frames.append(matrix.dropna(subset=CROSS_SECTIONAL_BASE))

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames).sort_values(["ts", "ticker"]).reset_index(drop=True)
    return add_cross_sectional(panel)


NON_FEATURE_COLUMNS = {
    "ticker", "ts", "forward_return", "label", "band",
    "fast_forward_return", "fast_label", "abs_return",
}


def feature_columns(frame: pd.DataFrame, cross_sectional: bool = True) -> list:
    columns = [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]
    if not cross_sectional:
        columns = [c for c in columns if not c.startswith("xs_")]
    return columns


LABEL_COLUMNS = ["label", "fast_label", "forward_return"]


def labelled(frame: pd.DataFrame) -> pd.DataFrame:
    """So' as linhas cujo futuro ja' e' conhecido. E' o que se pode treinar e
    avaliar; o resto so' serve para decidir.

    Exige apenas as colunas de rotulo que existirem, para o mesmo corte servir
    tambem painéis parciais (um so' horizonte, ou uma fatia de diagnostico)."""
    present = [c for c in LABEL_COLUMNS if c in frame.columns]
    return frame.dropna(subset=present) if present else frame


def split(frame: pd.DataFrame, horizon: int = None):
    """Corte cronologico global com embargo, nao por ticker.

    Cortar por ticker deixaria o treino de um ticker a sobrepor-se ao teste de
    outro no mesmo dia -- e como eles se movem juntos, isso e' fuga de
    informacao disfarcada."""
    horizon = horizon or D["horizon_days"]
    frame = labelled(frame)
    stamps = np.sort(frame["ts"].unique())
    if len(stamps) < 10:
        empty = frame.iloc[0:0]
        return empty, empty, empty
    first = stamps[int(len(stamps) * D["train_fraction"])]
    second = stamps[int(len(stamps) * (D["train_fraction"] + D["validation_fraction"]))]
    step = pd.Series(stamps).diff().median()
    embargo = step * horizon if pd.notna(step) else pd.Timedelta(0)
    return (frame[frame["ts"] < first - embargo],
            frame[(frame["ts"] >= first) & (frame["ts"] < second - embargo)],
            frame[frame["ts"] >= second])


def effective_days(frame: pd.DataFrame) -> int:
    return int(pd.Series(frame["ts"]).nunique())
