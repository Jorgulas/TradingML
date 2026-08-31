"""EXPERIENCIA N -- fluxo assinado aproximado a partir de barras OHLCV.

DE ONDE VEM. A pesquisa sobre a estrategia de order flow do Chris Creamer
deixou uma pergunta em aberto que importa mais do que a estrategia dela: a
literatura de microestrutura (Cont, Kukanov e Stoikov, 2014) mostra que o
DESEQUILIBRIO DE FLUXO DE ORDENS preve variacoes de preco a prazos curtos, de
forma quase linear e estavel entre accoes. A ronda 10 nao encontrou nada ao
minuto. A diferenca entre as duas coisas pode ser exactamente a categoria de
dados que este projecto nunca teve: uma barra OHLCV deita fora quem foi
agressor, a que preco e contra que profundidade.

O que sobra numa barra e' o CLOSE LOCATION VALUE:

    clv = ((fecho - minimo) - (maximo - fecho)) / (maximo - minimo)

Vale +1 se a barra fechou no topo do proprio intervalo e -1 se fechou no fundo.
Multiplicado pelo volume, e' o bloco de construcao da linha de
Acumulacao/Distribuicao e a aproximacao classica de volume assinado a partir de
OHLCV. E' fraco -- nao sabe quem atravessou o spread -- mas e' honesto e e'
computavel com o que ha'.

DUAS ARMADILHAS, e a primeira invalidaria tudo:

1. O VOLUME EM BRUTO NAO SE COMPARA ENTRE ACCOES. A metrica aqui e' a
   correlacao TRANSVERSAL dentro de cada barra, e clv x volume poe sempre a
   AAPL no topo por ela negociar mais, nao por estar a comprar. Ordenar-se-ia
   capitalizacao bolsista disfarcada de fluxo. O volume tem de ser dividido
   pela media recente do PROPRIO ticker, ficando "quantas vezes o seu volume
   normal" -- adimensional e comparavel.

2. O SALTO BID-ASK MENTE PRECISAMENTE AQUI. Uma barra que fecha no topo do
   intervalo fechou provavelmente no preco de venda, e a barra seguinte tende a
   voltar ao preco de compra. Isso produz um CLV com poder preditivo NEGATIVO
   que e' puro artefacto de cotacao e nao se pode negociar. E' o contaminante
   com que este sinal tem mais afinidade de todos os que este projecto testou,
   e por isso mede-se sempre com e sem atraso de entrada.

O protocolo e' o da ronda 10, que foi o que funcionou: sessao fechada, grelha
completa sem escolher celula, e a REPLICA entre 1m, 2m e 5m como teste que
decide. Foi a replica que matou o joelho depois de ele ter passado os intervalos
de confianca em 17 celulas.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.decision.panel import is_corrupt
from src.decision.short_term import block_bootstrap_mean
from src.patterns.ingest_intraday import load_bars
from src.research.tangent_intraday import (GRIDS, STOCK_COST_BPS, bar_ic,
                                           minimum_detectable)

import numpy as np
import pandas as pd

# Janelas de acumulacao, em barras. O desequilibrio de fluxo da literatura e'
# agregado sobre um intervalo, nao lido barra a barra -- daqui as somas.
SUMS = (5, 20, 60)
VOLUME_WINDOW = 60      # barras para normalizar o volume de cada ticker


def close_location_value(bars: pd.DataFrame) -> pd.Series:
    """+1 se fechou no topo do intervalo da barra, -1 se fechou no fundo.

    Barras sem intervalo (maximo == minimo) ficam a 0 e nao a NaN: aconteceu,
    o preco nao se mexeu, e isso e' informacao neutra e nao ausencia de dados.
    Deixa-las NaN eliminaria silenciosamente os minutos mais calmos, que sao
    precisamente aqueles em que nao ha' fluxo."""
    span = bars["high"] - bars["low"]
    raw = ((bars["close"] - bars["low"]) - (bars["high"] - bars["close"]))
    return (raw / span.replace(0, np.nan)).fillna(0.0)


def build_signals(bars: pd.DataFrame, session_aware: bool = True) -> pd.DataFrame:
    """CLV puro, fluxo assinado de uma barra, e as suas somas moveis."""
    clv = close_location_value(bars)
    volume = bars["volume"].astype(float)

    if session_aware:
        session = pd.Series(bars.index.normalize(), index=bars.index)
        reference = volume.groupby(session).transform(
            lambda s: s.rolling(VOLUME_WINDOW, min_periods=20).mean())
    else:
        reference = volume.rolling(VOLUME_WINDOW, min_periods=20).mean()

    # Volume RELATIVO ao proprio ticker. Sem isto a ordenacao transversal seria
    # por dimensao da empresa e nao por fluxo (ver armadilha 1 no cabecalho).
    relative_volume = volume / reference.replace(0, np.nan)
    flow = clv * relative_volume

    out = pd.DataFrame({"clv": clv, "ofi": flow}, index=bars.index)
    if session_aware:
        session = pd.Series(bars.index.normalize(), index=bars.index)
        for window in SUMS:
            out[f"ofi_{window}"] = flow.groupby(session).transform(
                lambda s: s.rolling(window, min_periods=window).sum())
    else:
        for window in SUMS:
            out[f"ofi_{window}"] = flow.rolling(window, min_periods=window).sum()
    return out


SIGNALS = ["clv", "ofi"] + [f"ofi_{w}" for w in SUMS]


def panel(conn, timeframe: str, horizons, tickers=None, lag: int = 1,
          min_bars: int = 200) -> pd.DataFrame:
    """Sinais e retornos futuros para todos os tickers, dentro da sessao."""
    tickers = tickers or config.PATTERN_TICKERS
    benchmark = load_bars(conn, config.BENCHMARK["ticker"], timeframe)
    market = benchmark["close"] if not benchmark.empty else None

    frames = []
    for ticker in tickers:
        bars = load_bars(conn, ticker, timeframe)
        if len(bars) < min_bars or is_corrupt(bars):
            continue

        piece = build_signals(bars, session_aware=True)
        aligned = market.reindex(bars.index).ffill() if market is not None else None
        session = pd.Series(bars.index.normalize(), index=bars.index)

        for horizon in horizons:
            # Entrada atrasada `lag` barras: o sinal ve' o fecho da barra e a
            # posicao so' abre na seguinte. Com lag=0 mede-se o artefacto de
            # cotacao junto com o sinal; a diferenca entre os dois E' o teste.
            entry = bars["close"].shift(-lag)
            exit_price = bars["close"].shift(-(lag + horizon))
            forward = exit_price / entry - 1
            if aligned is not None:
                forward = forward - (aligned.shift(-(lag + horizon))
                                     / aligned.shift(-lag) - 1)
            forward = forward.where(session.shift(-(lag + horizon)) == session)
            piece[f"forward_{horizon}"] = forward

        piece["ticker"] = ticker
        piece["ts"] = piece.index
        piece.index = range(len(piece))
        frames.append(piece)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values(["ts", "ticker"]).reset_index(drop=True)


def grid(conn, timeframe: str, tickers=None, lag: int = 1) -> pd.DataFrame:
    """Todas as combinacoes de sinal e horizonte, sem escolher nenhuma."""
    spec = GRIDS[timeframe]
    frame = panel(conn, timeframe, spec["horizons"], tickers=tickers, lag=lag)
    if frame.empty:
        return pd.DataFrame()

    rows = []
    for horizon in spec["horizons"]:
        for signal in SIGNALS:
            series = bar_ic(frame, signal, horizon)
            if len(series) < 50:
                continue
            block = min(spec["bars_per_day"], max(2, len(series) // 4))
            low, high = block_bootstrap_mean(series.values, block=block)
            rows.append({
                "timeframe": timeframe, "lag": lag, "sinal": signal,
                "horizon": horizon, "ic": float(series.mean()),
                "ci": (low, high), "n_bars": int(len(series)),
                "mde": minimum_detectable(series, block),
                "outside": bool(low > 0 or high < 0),
            })
    return pd.DataFrame(rows)


def net_of_costs(conn, timeframe: str, signal: str, horizon: int,
                 fraction: float = 0.2, tickers=None, lag: int = 1,
                 train_fraction: float = 0.7) -> dict:
    """Carteira long-short sobre o sinal, com o SENTIDO estimado no treino.

    O erro que isto corrige, e que estava na primeira versao. A carteira ia
    sempre comprada no topo do ranking e vendida no fundo. Mas o IC medido
    aqui e' NEGATIVO -- fluxo acumulado antecede reversao -- portanto essa
    carteira estava a negociar ao contrario, e reportava prejuizo naquilo que
    era exactamente o sinal a funcionar. Um bruto de -0.06% lido como fracasso
    era +0.06% lido ao direito.

    Corrigido, mas nao a olhar para a resposta: o sentido e' estimado na
    primeira fatia do periodo e aplicado a' segunda. Escolhe-lo pelo IC da
    amostra inteira seria garantir lucro por construcao, que e' precisamente a
    forma mais facil de este projecto se enganar a si proprio."""
    spec = GRIDS[timeframe]
    frame = panel(conn, timeframe, spec["horizons"], tickers=tickers, lag=lag)
    if frame.empty:
        return {}

    target = f"forward_{horizon}"
    work = frame[["ts", "ticker", signal, target]].dropna()
    if work.empty:
        return {}

    stamps = np.sort(work["ts"].unique())
    boundary = stamps[int(len(stamps) * train_fraction)]
    train = work[work["ts"] < boundary]
    direction = np.sign(bar_ic(train, signal, horizon).mean()) if not train.empty else 1.0
    if direction == 0 or not np.isfinite(direction):
        direction = 1.0

    work = work[work["ts"] >= boundary].copy()
    if work.empty:
        return {}

    grouped = work.groupby("ts")
    ranked = grouped[signal].rank(pct=True)
    size = grouped[signal].transform("size")
    work = work.assign(rank=ranked, names=size)
    work = work[work["names"] >= 15]

    weight = direction * np.where(work["rank"] >= 1 - fraction, 1.0,
                                  np.where(work["rank"] <= fraction, -1.0, 0.0))
    work = work.assign(weight=weight)

    per_bar = (work.assign(contribution=work["weight"] * work[target])
               .groupby("ts")
               .apply(lambda g: float(g["contribution"].sum()
                                      / max((g["weight"] != 0).sum(), 1)),
                      include_groups=False))

    # Rotacao entre decisoes espacadas pelo horizonte.
    stamps = sorted(work["ts"].unique())[::horizon]
    previous, changes = None, []
    for stamp in stamps:
        slice_ = work[(work["ts"] == stamp) & (work["weight"] != 0)]
        names = set(zip(slice_["ticker"], slice_["weight"]))
        if previous and names:
            changes.append(1 - len(names & previous) / len(previous))
        previous = names or previous
    turnover = float(np.mean(changes)) if changes else 1.0

    cycles = len(per_bar) / horizon if horizon else len(per_bar)
    gross_per_cycle = float(per_bar.sum() / cycles) if cycles else np.nan
    cost = 2 * (STOCK_COST_BPS / 10_000) * turnover
    return {
        "signal": signal, "horizon": horizon, "direction": float(direction),
        "train_days": int(pd.DatetimeIndex(train["ts"]).normalize().nunique()),
        "test_days": int(pd.DatetimeIndex(work["ts"]).normalize().nunique()),
        "gross_per_cycle": gross_per_cycle, "turnover": turnover,
        "cost_per_cycle": cost, "net_per_cycle": gross_per_cycle - cost,
        "typical_move": float(work[target].abs().mean()),
        "n_bars": int(len(per_bar)),
    }


def run(conn, timeframes=("1m", "2m", "5m")) -> dict:
    tables = {}
    for timeframe in timeframes:
        for lag in (0, 1):
            table = grid(conn, timeframe, lag=lag)
            if not table.empty:
                tables[(timeframe, lag)] = table

    days = {}
    for timeframe in timeframes:
        rows = conn.execute(
            "SELECT COUNT(DISTINCT substr(ts, 1, 10)) FROM intraday_prices "
            "WHERE timeframe = ?", (timeframe,)).fetchone()
        days[timeframe] = rows[0] if rows else 0

    return {"tables": tables, "days": days}


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection)

    print("=" * 96)
    print("EXPERIENCIA N -- fluxo assinado aproximado (CLV x volume) ao minuto")
    print("  grelha completa, sessao fechada, sem escolher celula")

    print()
    print("  O TESTE DO SALTO BID-ASK -- entrada imediata contra entrada uma barra depois:")
    print(f"    {'prazo':>6} {'sinal':>8} {'IC (lag 0)':>12} {'IC (lag 1)':>12} "
          f"{'sobrevive?':>12}")
    for timeframe in ("1m", "2m", "5m"):
        zero = outcome["tables"].get((timeframe, 0))
        one = outcome["tables"].get((timeframe, 1))
        if zero is None or one is None:
            continue
        for signal in SIGNALS:
            a = zero[zero["sinal"] == signal]["ic"]
            b = one[one["sinal"] == signal]["ic"]
            if a.empty or b.empty:
                continue
            # "Sobrevive" = mantem o sinal e pelo menos metade da magnitude.
            keeps = (np.sign(a.mean()) == np.sign(b.mean())
                     and abs(b.mean()) > 0.5 * abs(a.mean()))
            print(f"    {timeframe:>6} {signal:>8} {a.mean():>+12.5f} "
                  f"{b.mean():>+12.5f} {'sim' if keeps else 'NAO':>12}")

    print()
    print("  A GRELHA com entrada atrasada (o unico numero negociavel):")
    print(f"    {'prazo':>6} {'dias':>5} {'sinal':>8} {'IC medio':>11} "
          f"{'|IC| max':>10} {'MDE':>9} {'fora do zero':>13}")
    for timeframe in ("1m", "2m", "5m"):
        table = outcome["tables"].get((timeframe, 1))
        if table is None:
            continue
        for signal in SIGNALS:
            part = table[table["sinal"] == signal]
            if part.empty:
                continue
            print(f"    {timeframe:>6} {outcome['days'][timeframe]:>5} {signal:>8} "
                  f"{part['ic'].mean():>+11.5f} {part['ic'].abs().max():>10.5f} "
                  f"{part['mde'].median():>9.5f} "
                  f"{int(part['outside'].sum()):>8}/{len(part):<4}")

    print()
    print("  O TESTE QUE DECIDE -- o que aparece a 1m replica onde ha' mais dias?")
    print(f"    {'sinal':>8} {'1m (9d)':>11} {'2m (30d)':>11} {'5m (71d)':>11} "
          f"{'mesmo sinal?':>13}")
    for signal in SIGNALS:
        values = []
        for timeframe in ("1m", "2m", "5m"):
            table = outcome["tables"].get((timeframe, 1))
            part = table[table["sinal"] == signal] if table is not None else None
            values.append(part["ic"].mean() if part is not None and not part.empty
                          else np.nan)
        finite = [v for v in values if np.isfinite(v)]
        agree = (len(finite) == 3 and len(set(np.sign(finite))) == 1)
        print(f"    {signal:>8} {values[0]:>+11.5f} {values[1]:>+11.5f} "
              f"{values[2]:>+11.5f} {'sim' if agree else 'NAO':>13}")

    print()
    print("  LIQUIDO DE CUSTOS -- sentido estimado no treino, medido no resto.")
    print("  Sinal fixado em ofi_20: e' o unico com o mesmo sentido nos tres prazos.")
    print(f"    {'prazo':>6} {'treino':>7} {'teste':>6} {'sent.':>6} "
          f"{'bruto/ciclo':>12} {'rotacao':>8} {'custo':>9} {'liquido':>10}")
    for timeframe in ("1m", "2m", "5m"):
        spec = GRIDS[timeframe]
        result = net_of_costs(connection, timeframe, "ofi_20", spec["horizons"][1])
        if not result:
            continue
        print(f"    {timeframe:>6} {result['train_days']:>6}d {result['test_days']:>5}d "
              f"{result['direction']:>+6.0f} {result['gross_per_cycle']:>+12.4%} "
              f"{result['turnover']:>8.0%} {result['cost_per_cycle']:>9.4%} "
              f"{result['net_per_cycle']:>+10.4%}")
    connection.close()
