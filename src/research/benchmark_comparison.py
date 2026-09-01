"""EXPERIENCIA O -- tudo o que construimos contra comprar e ficar quieto.

A LACUNA QUE ISTO FECHA. Onze rondas mediram sinais uns contra os outros, contra
o acaso, contra o HAR-RV, contra votacoes simples. Nunca contra a alternativa que
qualquer pessoa tem sem escrever uma linha de codigo: comprar o indice e nao lhe
tocar. E' a referencia que decide se alguma coisa aqui valeu a pena, e faltava.

DUAS COMPARACOES, porque respondem a perguntas diferentes:

1. O LONGO PRAZO, com o historico todo que existe do SPY (desde 1993). Tres anos
   nao dizem nada sobre investir a longo prazo, e o periodo deste projecto calhou
   ser um mercado em alta -- o que favorece comprar e esperar de uma forma que
   nao e' representativa. Trinta anos incluem 2000-2002, 2008-2009 e 2020, e e'
   la' que se ve' o que "seguro" custa de verdade.

2. O CONFRONTO DIRECTO, no periodo em que ha' dados para todos os algoritmos.
   Mesmas datas, mesmos custos, mesma aritmetica.

A ESTRATEGIA E' SIMULADA A SERIO, e nao pela media dos retornos futuros como nas
rondas anteriores. Rebalanceia-se de cinco em cinco dias sem sobreposicao,
compra-se o decil de topo, vende-se o de baixo, paga-se o spread, e compoe-se. O
que sai e' uma curva de capital comparavel a' do indice, em vez de uma media de
janelas que se sobrepoem e nao se pode compor.

NAO E' ACONSELHAMENTO. E' a medicao do que estes algoritmos fizeram neste
historico, com as ressalvas que as rondas 8 a 11 estabeleceram.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.decision import panel as P
from src.decision import short_term as ST
from src.research.fallen_universe import surviving_fallen

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252
COST_BPS = 3.0            # spread por ponta
REBALANCE_DAYS = 5        # o horizonte do sinal de curto prazo
FRACTION = 0.1            # decil de topo e de baixo


def describe(returns: pd.Series, label: str) -> dict:
    """Estatisticas de uma serie de retornos periodicos.

    O drawdown maximo esta' aqui de proposito ao lado do retorno: e' a metade da
    resposta a "seguro e com maior retorno" que costuma ficar de fora, e no caso
    de comprar e esperar e' a metade que magoa."""
    values = returns.dropna().values
    if len(values) < 2:
        return {"name": label}
    curve = np.cumprod(1 + values)
    drawdown = curve / np.maximum.accumulate(curve) - 1
    years = len(values) / TRADING_DAYS
    volatility = values.std(ddof=1) * np.sqrt(TRADING_DAYS)
    cagr = curve[-1] ** (1 / years) - 1 if years > 0 else np.nan
    return {
        "name": label,
        "total": float(curve[-1] - 1),
        "cagr": float(cagr),
        "vol": float(volatility),
        "sharpe": float(cagr / volatility) if volatility > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "worst_day": float(values.min()),
        "positive_days": float((values > 0).mean()),
        "years": float(years),
        "n": int(len(values)),
    }


def long_run_index(ticker: str = "SPY", start: str = "1993-01-01") -> pd.Series:
    """Retornos diarios do indice com todo o historico disponivel."""
    frame = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=True)
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    return frame["Close"].pct_change().dropna()


def decade_by_decade(returns: pd.Series) -> pd.DataFrame:
    """O mesmo indice, ano a ano. E' onde se ve' que a media anual esconde
    -37% em 2008 e -19% em 2022, e que "longo prazo" quer dizer aguentar isso."""
    yearly = returns.groupby(pd.DatetimeIndex(returns.index).year).apply(
        lambda s: float(np.prod(1 + s.values) - 1))
    return yearly


def strategy_curve(conn, panel_frame: pd.DataFrame, min_train_quarters: int = 6,
                   fraction: float = FRACTION) -> pd.Series:
    """Simulacao a serio do sinal transversal de curto prazo.

    Rebalanceamento de 5 em 5 dias SEM sobreposicao, modelo re-treinado todos os
    trimestres so' com o passado, custo cobrado sobre a rotacao efectiva. O
    retorno de cada periodo e' o do decil de topo menos o do decil de baixo,
    dividido por dois porque metade do capital esta' de cada lado."""
    frame = panel_frame.copy()
    frame["quarter"] = (pd.DatetimeIndex(pd.to_datetime(frame["ts"]))
                        .tz_localize(None).to_period("Q").astype(str))
    labels = sorted(frame["quarter"].unique())
    columns = ST.feature_sets(frame)["transversal (percentis do dia)"]

    pieces, previous = [], set()
    for index in range(min_train_quarters, len(labels)):
        label = labels[index]
        evaluation = P.labelled(frame[frame["quarter"] == label])
        if evaluation.empty:
            continue
        cutoff = evaluation["ts"].min() - pd.Timedelta(days=REBALANCE_DAYS + 2)
        history = P.labelled(frame[frame["ts"] <= cutoff])
        if len(history) < 5000:
            continue

        model = ST.fit(history, columns)
        evaluation = evaluation.copy()
        evaluation["confidence"] = ST.score(model, evaluation, columns)

        # Datas de rebalanceamento: de REBALANCE_DAYS em REBALANCE_DAYS, sem
        # sobreposicao, para os retornos poderem ser compostos.
        stamps = sorted(evaluation["ts"].unique())[::REBALANCE_DAYS]
        for stamp in stamps:
            group = evaluation[evaluation["ts"] == stamp]
            if len(group) < 20:
                continue
            size = max(1, int(len(group) * fraction))
            ordered = group.sort_values("confidence")
            longs = ordered.iloc[-size:]
            shorts = ordered.iloc[:size]

            gross = (longs["forward_return"].mean()
                     - shorts["forward_return"].mean()) / 2

            names = (set(zip(longs["ticker"], [1] * len(longs)))
                     | set(zip(shorts["ticker"], [-1] * len(shorts))))
            turnover = (1 - len(names & previous) / len(previous)) if previous else 1.0
            previous = names
            # Duas pontas (abrir e fechar) sobre a exposicao bruta.
            cost = 2 * (COST_BPS / 10_000) * turnover

            pieces.append({"ts": stamp, "ret": float(gross - cost)})

    if not pieces:
        return pd.Series(dtype=float)
    out = pd.DataFrame(pieces).set_index("ts")["ret"]
    return out[~out.index.duplicated()].sort_index()


def buy_and_hold(panel_frame: pd.DataFrame, stamps: pd.Index,
                 index_returns: pd.Series) -> pd.Series:
    """Carteira de peso igual em todo o universo, em retorno TOTAL.

    Cuidado que custou uma leitura errada: o `forward_return` do painel e' o
    retorno em EXCESSO sobre o indice, porque e' isso que os modelos preveem.
    Usa-lo aqui media' "o universo menos o SPY" e rotulava-o de "comprar o
    universo" -- deu -4.2% onde o indice deu +35.9%, e a diferenca era
    inteiramente a subtraccao esquecida, nao o desempenho.

    Como excesso = accao - indice, o total recupera-se somando o indice de
    volta. E' exacto e nao obriga a reconstruir o painel."""
    work = panel_frame[["ts", "forward_return"]].dropna()
    excess = work.groupby("ts")["forward_return"].mean().reindex(stamps)
    return (excess + index_returns.reindex(stamps)).dropna()


def run(conn) -> dict:
    excluded = []
    universe = list(config.TREE_TICKERS) + surviving_fallen(conn)
    frame = P.build(conn, tickers=universe, min_days=120, timeframe="1d",
                    excluded=excluded)

    strategy = strategy_curve(conn, frame)
    if strategy.empty:
        return {}

    # O indice nas MESMAS datas, para o confronto ser periodo a periodo.
    from src.patterns.ingest_intraday import load_bars
    spy = load_bars(conn, config.BENCHMARK["ticker"], "1d")["close"]
    spy_periodic = (spy.reindex(pd.DatetimeIndex(strategy.index), method="ffill")
                    .pct_change(fill_method=None)
                    .shift(-1).dropna())

    equal_weight = buy_and_hold(frame, strategy.index, spy_periodic)

    # Escala anual: cada observacao e' um periodo de REBALANCE_DAYS dias.
    periods = TRADING_DAYS / REBALANCE_DAYS

    def scaled(returns, label):
        values = returns.dropna().values
        if len(values) < 2:
            return {"name": label}
        curve = np.cumprod(1 + values)
        drawdown = curve / np.maximum.accumulate(curve) - 1
        years = len(values) / periods
        volatility = values.std(ddof=1) * np.sqrt(periods)
        cagr = curve[-1] ** (1 / years) - 1 if years > 0 else np.nan
        low, high = ST.block_bootstrap_mean(values, block=1)
        return {"name": label, "total": float(curve[-1] - 1), "cagr": float(cagr),
                "vol": float(volatility),
                "sharpe": float(cagr / volatility) if volatility > 0 else np.nan,
                "max_drawdown": float(drawdown.min()),
                "worst_period": float(values.min()),
                "positive": float((values > 0).mean()),
                "mean": float(values.mean()), "ci": (low, high),
                "significant": low > 0, "n": int(len(values)),
                "years": float(years)}

    head_to_head = [
        scaled(spy_periodic, "comprar o indice (SPY)"),
        scaled(equal_weight, "comprar as 214 accoes, peso igual"),
        scaled(strategy, "sinal transversal long-short (liquido)"),
    ]

    index_returns = long_run_index()
    return {
        "excluded": excluded,
        "n_tickers": int(frame["ticker"].nunique()),
        "head_to_head": head_to_head,
        "long_run": describe(index_returns, "SPY desde 1993"),
        "yearly": decade_by_decade(index_returns),
        "start": str(pd.Timestamp(strategy.index.min()).date()),
        "end": str(pd.Timestamp(strategy.index.max()).date()),
    }


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection)
    if not outcome:
        print("dados insuficientes")
        raise SystemExit

    print("=" * 98)
    print("EXPERIENCIA O -- tudo contra comprar e ficar quieto")

    long_run = outcome["long_run"]
    print()
    print(f"  1. O LONGO PRAZO -- {long_run['name']}, {long_run['years']:.0f} anos:")
    print(f"     retorno total {long_run['total']:>10.0%}   "
          f"anualizado {long_run['cagr']:>7.2%}   "
          f"volatilidade {long_run['vol']:>6.2%}")
    print(f"     Sharpe {long_run['sharpe']:>7.2f}   "
          f"DRAWDOWN MAXIMO {long_run['max_drawdown']:>7.1%}   "
          f"pior dia {long_run['worst_day']:>7.1%}")

    yearly = outcome["yearly"]
    worst = yearly.nsmallest(5)
    print()
    print("     Os cinco piores anos, que e' o que 'seguro' custa de verdade:")
    print("       " + "   ".join(f"{year}: {value:+.1%}"
                                 for year, value in worst.items()))

    print()
    print(f"  2. CONFRONTO DIRECTO -- {outcome['start']} a {outcome['end']}, "
          f"{outcome['n_tickers']} accoes, rebalanceamento de {REBALANCE_DAYS} dias")
    print(f"     {'estrategia':>38} {'total':>8} {'anual':>8} {'vol':>7} "
          f"{'Sharpe':>7} {'drawdown':>9} {'periodos +':>11}")
    for row in outcome["head_to_head"]:
        if "total" not in row:
            continue
        print(f"     {row['name']:>38} {row['total']:>8.1%} {row['cagr']:>8.2%} "
              f"{row['vol']:>7.2%} {row['sharpe']:>7.2f} "
              f"{row['max_drawdown']:>9.1%} {row['positive']:>11.1%}")

    print()
    print("     Retorno medio por periodo, com intervalo de confianca:")
    for row in outcome["head_to_head"]:
        if "ci" not in row:
            continue
        low, high = row["ci"]
        mark = "*" if row["significant"] else " "
        print(f"     {row['name']:>38} {row['mean']:>+9.4%}{mark} "
              f"[{low:>+8.4%}, {high:>+8.4%}]")

    print()
    print("  NOTA sobre o que se esta' a comparar: as duas primeiras linhas tem")
    print("  exposicao total ao mercado; a terceira e' neutra por construcao (metade")
    print("  comprada, metade vendida). Sao perfis de risco diferentes, e o periodo")
    print("  do confronto e' curto e de mercado em alta -- o que favorece as duas")
    print("  primeiras de uma forma que 34 anos de historico mostram nao ser tipica.")
    print()
    print("  * = intervalo inteiramente acima de zero")
    connection.close()
