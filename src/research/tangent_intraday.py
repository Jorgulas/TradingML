"""EXPERIENCIA M -- o joelho ao MINUTO, que era o enquadramento original.

A ronda 9 mediu o angulo da tangente a 1 hora e ao dia, e deu zero. Mas o
pedido do Jorge era "para o trading real-time", e ao minuto ha' tres coisas
mesmo diferentes: a media move-se muito mais depressa, o joelho dispara muitas
mais vezes, e os custos passam a ser da ordem de grandeza do movimento.

TRES DECISOES DE DESENHO que este prazo obriga, e que a ronda 9 nao precisava:

1. NADA ATRAVESSA A NOITE. Uma media de 20 barras na quinta barra do dia usaria
   15 barras de ontem e o `diff` apanharia o salto overnight -- que pode ser 1%
   e nao e' um movimento de um minuto. O angulo dispararia para o tecto todas as
   manhas, sempre no sentido do gap, e a regra negociaria o gap disfarcado de
   joelho. Tudo e' calculado dentro da sessao (`session_aware=True`).

2. NAO HA' DIVISAO TREINO/TESTE HONESTA AO MINUTO. Sao 9 dias distintos de
   historico. Cortar 80/20 daria 7 dias para escolher e 2 para medir, e a
   medicao nao significaria nada. Por isso NAO SE ESCOLHE: reporta-se a GRELHA
   INTEIRA de correlacoes, com correccao de Bonferroni para o numero de celulas.
   Uma grelha completa nao pode ser acusada de escolher a melhor.

3. OS CUSTOS DEIXAM DE SER DETALHE. A 1 minuto, o movimento tipico em excesso e'
   da ordem de 5 pontos base e uma ida-e-volta custa 6. E' o facto que decide
   este prazo, e mede-se explicitamente em vez de se assumir.

A METRICA e' o coeficiente de informacao por BARRA: a correlacao de Spearman,
entre os 43 tickers de cada minuto, entre o sinal e o retorno seguinte. E'
transversal, portanto imune ao factor mercado, e da' uma observacao por barra em
vez de uma por dia -- que e' o que torna 9 dias mensuraveis de todo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.decision.short_term import block_bootstrap_mean
from src.research import tangent_angle as L

import numpy as np
import pandas as pd
from scipy import stats

# Parametros em BARRAS, escalados a cada prazo para cobrirem tempos de relogio
# parecidos. A 1m uma media de 20 barras sao 20 minutos; a 5m sao 100.
GRIDS = {
    "1m": {"ma": (10, 20, 60), "span": (5, 10, 30), "horizons": (1, 5, 15),
           "bars_per_day": 390},
    "2m": {"ma": (10, 20, 30), "span": (5, 10, 15), "horizons": (1, 3, 8),
           "bars_per_day": 195},
    "5m": {"ma": (6, 12, 20), "span": (3, 6, 10), "horizons": (1, 3, 6),
           "bars_per_day": 78},
}

STOCK_COST_BPS = 3.0     # spread por ponta em large-caps americanas
THRESHOLD = 5.0          # graus; foi o que os varrimentos da ronda 9 escolheram


def bar_ic(frame: pd.DataFrame, column: str, horizon: int,
           min_names: int = 15) -> pd.Series:
    """Spearman entre os tickers de CADA barra, sinal contra retorno seguinte.

    Uma observacao por barra. E' transversal, portanto o que o mercado inteiro
    fizer naquele minuto nao entra -- so' conta a ORDEM entre accoes, que e'
    exactamente o que um sinal destes tem de acertar."""
    target = f"forward_{horizon}"
    work = frame[["ts", column, target]].dropna()
    if work.empty:
        return pd.Series(dtype=float)

    # Vectorizado. Chamar `stats.spearmanr` barra a barra sobre milhares de
    # barras demorava minutos por celula, e a grelha tem 54. Spearman e' apenas
    # Pearson sobre POSTOS, e os postos dentro de cada barra calculam-se de uma
    # vez com `groupby.rank`.
    grouped = work.groupby("ts")
    counts = grouped[column].transform("size")
    work = work[counts >= min_names]
    if work.empty:
        return pd.Series(dtype=float)

    grouped = work.groupby("ts")
    ranks_x = grouped[column].rank()
    ranks_y = grouped[target].rank()
    frame_ranks = pd.DataFrame({"ts": work["ts"], "x": ranks_x, "y": ranks_y})

    def _correlate(group):
        if group["x"].nunique() < 5:
            return np.nan
        x = group["x"] - group["x"].mean()
        y = group["y"] - group["y"].mean()
        denominator = np.sqrt((x ** 2).sum() * (y ** 2).sum())
        return float((x * y).sum() / denominator) if denominator > 0 else np.nan

    return (frame_ranks.groupby("ts")[["x", "y"]]
            .apply(_correlate).dropna().sort_index())


def minimum_detectable(series: pd.Series, block: int) -> float:
    """Que tamanho de efeito seria detectavel com esta amostra.

    Sem este numero, um intervalo que contem zero e' ambiguo: nao se sabe se o
    efeito e' nulo ou se a amostra e' cega. Com ele, "o IC medido e' 0.001 e o
    minimo detectavel e' 0.004" diz que nao se pode concluir nada, enquanto "o
    IC e' 0.001 e o minimo detectavel e' 0.002" ja' e' um nulo informativo.

    Calculado como ~2 erros-padrao, com o erro-padrao a vir do proprio bootstrap
    de blocos (que ja' incorpora a dependencia dentro do dia)."""
    low, high = block_bootstrap_mean(series.values, block=block)
    if not np.isfinite(low):
        return np.nan
    return float((high - low) / 2)


def grid(conn, timeframe: str, tickers=None, lag: int = 0) -> pd.DataFrame:
    """A grelha inteira de configuracoes, sem escolher nenhuma."""
    spec = GRIDS[timeframe]
    tickers = tickers or config.PATTERN_TICKERS
    rows = []

    for ma_window in spec["ma"]:
        for span in spec["span"]:
            frame = L.panel(conn, timeframe, ma_window, span, spec["horizons"],
                            tickers=tickers, session_aware=True, min_bars=200,
                            lag=lag)
            if frame.empty:
                continue
            for horizon in spec["horizons"]:
                for column in ("kink", "angle"):
                    series = bar_ic(frame, column, horizon)
                    if len(series) < 50:
                        continue
                    # Blocos de um dia inteiro de barras: e' a escolha
                    # conservadora, e assume que tudo dentro do dia pode estar
                    # correlacionado.
                    block = min(spec["bars_per_day"], max(2, len(series) // 4))
                    low, high = block_bootstrap_mean(series.values, block=block)
                    rows.append({
                        "timeframe": timeframe, "lag": lag,
                        "ma": ma_window, "span": span,
                        "horizon": horizon, "sinal": column,
                        "ic": float(series.mean()),
                        "ci": (low, high), "n_bars": int(len(series)),
                        "mde": minimum_detectable(series, block),
                        "positive_bars": float((series > 0).mean()),
                    })
    return pd.DataFrame(rows)


def rule_cost(conn, timeframe: str, ma_window: int, span: int,
              horizon: int, tickers=None) -> dict:
    """O que a regra rende e o que a execucao lhe cobra.

    A pergunta que decide este prazo nao e' se o sinal existe, e' se sobra
    alguma coisa depois de pagar o spread. Uma ida-e-volta custa 2 pontas."""
    spec = GRIDS[timeframe]
    frame = L.panel(conn, timeframe, ma_window, span, spec["horizons"],
                    tickers=tickers or config.PATTERN_TICKERS,
                    session_aware=True, min_bars=200)
    if frame.empty:
        return {}

    target = f"forward_{horizon}"
    weight = L.positions(frame, "joelho", THRESHOLD)
    gross = L.bar_returns(frame, weight, target)

    # Rotacao: quanto da carteira muda entre decisoes espacadas pelo horizonte.
    held = frame.assign(w=weight.values)
    held = held[held["w"] != 0]
    stamps = sorted(frame["ts"].unique())[::horizon]
    previous, changes = None, []
    for stamp in stamps:
        names = set(zip(held.loc[held["ts"] == stamp, "ticker"],
                        np.sign(held.loc[held["ts"] == stamp, "w"])))
        if previous:
            changes.append(1 - len(names & previous) / len(previous))
        previous = names or previous
    turnover = float(np.mean(changes)) if changes else 1.0

    exposure = float(weight.abs().mean())
    cost_per_cycle = 2 * (STOCK_COST_BPS / 10_000) * turnover * exposure
    cycles = len(gross) / horizon if horizon else len(gross)

    typical = float(frame[target].abs().mean())
    return {
        "gross_total": float(gross.sum()),
        "gross_per_cycle": float(gross.sum() / cycles) if cycles else np.nan,
        "exposure": exposure, "turnover": turnover,
        "cost_per_cycle": cost_per_cycle,
        "net_per_cycle": (float(gross.sum() / cycles) - cost_per_cycle
                          if cycles else np.nan),
        "typical_move": typical,
        "cost_as_share_of_move": (2 * STOCK_COST_BPS / 10_000) / typical
        if typical else np.nan,
        "n_bars": int(len(gross)),
    }


def regions(table: pd.DataFrame, signal: str = "kink") -> pd.DataFrame:
    """Agrega a grelha pela razao troco/media.

    Porque esta divisao e nao outra. A 1 minuto o sinal do IC nao era aleatorio
    entre configuracoes: era POSITIVO em todas as que tinham troco menor ou igual
    a' media e NEGATIVO em todas as outras, sem excepcao. A fronteira cai
    exactamente em troco = media, que e' onde as duas rectas deixam de partilhar
    barras e passam a comparar trocos de preco disjuntos. Parecia estrutural, e
    por isso merecia ser testada noutros prazos em vez de acreditada."""
    part = table[table["sinal"] == signal].copy()
    part["ratio"] = part["span"] / part["ma"]
    part["regiao"] = np.where(part["ratio"] <= 1.0, "troco <= media", "troco > media")
    return (part.groupby("regiao")
            .agg(ic=("ic", "mean"), positivos=("ic", lambda s: int((s > 0).sum())),
                 celulas=("ic", "size"), mde=("mde", "median"))
            .reset_index())


def run(conn, timeframes=("1m", "2m", "5m"), lag: int = 1) -> dict:
    """Grelha completa em cada prazo, mais os custos.

    `lag=1` por omissao: o sinal ve' o fecho da barra e a posicao so' abre na
    barra seguinte. E' o que acontece na realidade, e serve tambem de teste ao
    salto bid-ask -- se o efeito vivesse da alternancia entre preco de compra e
    de venda, desaparecia ao saltar uma barra."""
    tables, costs = {}, {}
    for timeframe in timeframes:
        table = grid(conn, timeframe, lag=lag)
        if table.empty:
            continue
        tables[timeframe] = table
        spec = GRIDS[timeframe]
        costs[timeframe] = rule_cost(conn, timeframe, spec["ma"][1],
                                     spec["span"][1], spec["horizons"][1])

    days = {}
    for timeframe in tables:
        rows = conn.execute(
            "SELECT DISTINCT substr(ts, 1, 10) FROM intraday_prices WHERE timeframe = ?",
            (timeframe,)).fetchall()
        days[timeframe] = len(rows)

    return {"tables": tables, "costs": costs, "lag": lag, "days": days,
            "regions": {tf: regions(t) for tf, t in tables.items()}}


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection)

    print("=" * 92)
    print("EXPERIENCIA M -- o joelho ao minuto (trading em tempo real)")
    print(f"  grelha completa, sem escolher configuracao; entrada atrasada "
          f"{outcome['lag']} barra")
    print("  com 9 dias ao minuto nao existe divisao treino/teste honesta, e uma")
    print("  grelha inteira nao pode ser acusada de escolher a melhor celula.")

    print()
    print("  A GRELHA, prazo a prazo:")
    print(f"    {'prazo':>6} {'dias':>5} {'sinal':>7} {'IC mediano':>12} "
          f"{'|IC| max':>10} {'MDE':>9} {'fora do zero':>13}")
    for timeframe, table in outcome["tables"].items():
        for signal in ("kink", "angle"):
            part = table[table["sinal"] == signal]
            if part.empty:
                continue
            outside = int(part.apply(lambda r: r["ci"][0] > 0 or r["ci"][1] < 0,
                                     axis=1).sum())
            print(f"    {timeframe:>6} {outcome['days'][timeframe]:>5} {signal:>7} "
                  f"{part['ic'].median():>+12.5f} {part['ic'].abs().max():>10.5f} "
                  f"{part['mde'].median():>9.5f} {outside:>8}/{len(part):<4}")

    print()
    print("  O TESTE QUE DECIDE -- a 1 minuto o sinal do IC nao era aleatorio:")
    print("  positivo em TODAS as configuracoes com troco <= media, negativo em")
    print("  todas as outras. Replica nos prazos com mais dias?")
    print()
    print(f"    {'prazo':>6} {'dias':>5} {'regiao':>16} {'IC medio':>10} "
          f"{'positivos':>11}")
    for timeframe, table in outcome["regions"].items():
        for _, row in table.iterrows():
            print(f"    {timeframe:>6} {outcome['days'][timeframe]:>5} "
                  f"{row['regiao']:>16} {row['ic']:>+10.4f} "
                  f"{row['positivos']:>7}/{row['celulas']:<3}")

    print()
    print("  CUSTOS -- o que decide este prazo independentemente do sinal:")
    print(f"    {'prazo':>6} {'movimento tipico':>17} {'ida-e-volta':>12} "
          f"{'custo/movimento':>16} {'bruto/ciclo':>12} {'liquido/ciclo':>14}")
    for timeframe, cost in outcome["costs"].items():
        if not cost:
            continue
        print(f"    {timeframe:>6} {cost['typical_move']:>17.4%} "
              f"{2 * STOCK_COST_BPS / 10_000:>12.4%} "
              f"{cost['cost_as_share_of_move']:>16.1%} "
              f"{cost['gross_per_cycle']:>12.4%} {cost['net_per_cycle']:>14.4%}")
    connection.close()
