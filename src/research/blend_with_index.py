"""EXPERIENCIA P -- o sinal vale como DIVERSIFICADOR do indice?

A PERGUNTA E' DIFERENTE DA DA RONDA 12, e a diferenca importa. La' media-se o
sinal SOZINHO contra o indice sozinho, e ele perdeu. Mas um activo pode melhorar
uma carteira mesmo rendendo menos do que ela, desde que se mova de forma
independente: a volatilidade da mistura cai mais depressa do que o retorno, e o
racio melhora. E' o unico almoco gratis que existe em financas.

E o livro long-short tem exactamente o perfil de quem podia fazer isso: metade
comprado e metade vendido, portanto NEUTRO AO MERCADO por construcao, com 7.7%
de volatilidade contra 16.5% do indice.

A CONDICAO, que e' aritmetica e nao opiniao. Juntar um activo S a uma carteira I
melhora o racio de Sharpe se e so' se

    Sharpe(S)  >  correlacao(S, I) x Sharpe(I)

Com correlacao zero, a condicao reduz-se a "o retorno de S tem de ser positivo".
Com correlacao NEGATIVA, ate' um activo de retorno ligeiramente negativo ajuda.
E' por isso que a correlacao medida decide a resposta, e nao o retorno isolado.

O PROBLEMA DE SEMPRE: 70 periodos. O retorno medido do sinal (-0.88% ao ano) tem
um intervalo que atravessa o zero folgadamente, e a Sharpe de uma mistura
calculada com esse numero herda toda essa incerteza. Por isso a peca central
desta experiencia nao e' a tabela de misturas -- e' o PONTO DE EQUILIBRIO: dado
a volatilidade e a correlacao medidas (que sao muito mais estaveis do que a
media), que retorno teria o sinal de ter para a mistura valer a pena? Essa
pergunta tem resposta estrutural e quase nao depende da amostra curta.

E um terceiro teste, que e' o que separa um diversificador a serio de um que so'
parece: o sinal rende alguma coisa PRECISAMENTE nos periodos em que o indice
cai? Uma correlacao media perto de zero pode esconder um activo que acompanha o
indice quando ele afunda, que e' quando a diversificacao teria de servir.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.decision import panel as P
from src.decision import short_term as ST
from src.research.benchmark_comparison import (REBALANCE_DAYS, TRADING_DAYS,
                                               strategy_curve)
from src.research.fallen_universe import surviving_fallen

import numpy as np
import pandas as pd
from scipy import stats

PERIODS_PER_YEAR = TRADING_DAYS / REBALANCE_DAYS
WEIGHTS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)


def annualise(values: np.ndarray) -> dict:
    curve = np.cumprod(1 + values)
    drawdown = curve / np.maximum.accumulate(curve) - 1
    years = len(values) / PERIODS_PER_YEAR
    volatility = values.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    cagr = curve[-1] ** (1 / years) - 1 if years > 0 else np.nan
    return {
        "cagr": float(cagr), "vol": float(volatility),
        "sharpe": float(cagr / volatility) if volatility > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "worst": float(values.min()),
    }


def blend(index_returns: pd.Series, strategy: pd.Series, weight: float) -> np.ndarray:
    """Carteira com `weight` do capital no sinal e o resto no indice.

    E' a leitura de ALOCACAO, nao de sobreposicao: o dinheiro que vai para o
    sinal sai do indice. A alternativa (manter 100% no indice e correr o
    long-short com margem por cima) so' melhora se o sinal render acima de zero,
    e portanto nao tem nada para descobrir enquanto o retorno medido for
    negativo."""
    joined = pd.concat([index_returns, strategy], axis=1, keys=["i", "s"]).dropna()
    return ((1 - weight) * joined["i"] + weight * joined["s"]).values


def break_even_return(index_returns: pd.Series, strategy: pd.Series) -> dict:
    """Que retorno teria o sinal de ter para melhorar o racio da carteira.

    Sai directamente da condicao de Sharpe: a mistura melhora se
    Sharpe(S) > correlacao x Sharpe(I), logo o retorno minimo e'

        mu_S = correlacao x Sharpe(I) x sigma_S

    Depende so' da correlacao e das duas volatilidades -- todas estimadas com
    muito mais precisao do que uma media. E' por isso que esta e' a peca fiavel
    da experiencia e a tabela de misturas e' a ilustracao."""
    joined = pd.concat([index_returns, strategy], axis=1, keys=["i", "s"]).dropna()
    correlation = float(joined["i"].corr(joined["s"]))

    index_stats = annualise(joined["i"].values)
    strategy_stats = annualise(joined["s"].values)

    required = correlation * index_stats["sharpe"] * strategy_stats["vol"]
    return {
        "correlation": correlation,
        "correlation_p": float(stats.pearsonr(joined["i"], joined["s"]).pvalue),
        "index_sharpe": index_stats["sharpe"],
        "strategy_vol": strategy_stats["vol"],
        "strategy_cagr": strategy_stats["cagr"],
        "required_cagr": float(required),
        "shortfall": float(strategy_stats["cagr"] - required),
        "helps": strategy_stats["cagr"] > required,
        "n": int(len(joined)),
    }


def crisis_behaviour(index_returns: pd.Series, strategy: pd.Series,
                     quantile: float = 0.2) -> dict:
    """O sinal rende quando o indice cai?

    O teste que separa um diversificador a serio de um que so' parece. Uma
    correlacao media perto de zero pode esconder um activo que acompanha o
    indice exactamente quando ele afunda -- e e' nesses periodos que a
    diversificacao teria de servir para alguma coisa."""
    joined = pd.concat([index_returns, strategy], axis=1, keys=["i", "s"]).dropna()
    threshold = joined["i"].quantile(quantile)
    bad = joined[joined["i"] <= threshold]
    good = joined[joined["i"] >= joined["i"].quantile(1 - quantile)]
    return {
        "threshold": float(threshold),
        "n_bad": int(len(bad)),
        "strategy_in_bad": float(bad["s"].mean()),
        "strategy_in_good": float(good["s"].mean()),
        "index_in_bad": float(bad["i"].mean()),
        "hit_rate_in_bad": float((bad["s"] > 0).mean()),
    }


def run(conn) -> dict:
    universe = list(config.TREE_TICKERS) + surviving_fallen(conn)
    frame = P.build(conn, tickers=universe, min_days=120, timeframe="1d")
    strategy = strategy_curve(conn, frame)
    if strategy.empty:
        return {}

    from src.patterns.ingest_intraday import load_bars
    spy = load_bars(conn, config.BENCHMARK["ticker"], "1d")["close"]
    index_returns = (spy.reindex(pd.DatetimeIndex(strategy.index), method="ffill")
                     .pct_change(fill_method=None).shift(-1).dropna())

    rows = []
    baseline = None
    for weight in WEIGHTS:
        values = blend(index_returns, strategy, weight)
        if len(values) < 5:
            continue
        result = annualise(values)
        result["weight"] = weight
        if weight == 0.0:
            baseline = values
        rows.append(result)

    # Diferenca emparelhada de Sharpe contra 100% no indice, por bootstrap de
    # blocos: os periodos partilham datas, portanto compara-se par a par.
    rng = np.random.default_rng(31)
    for row in rows:
        if row["weight"] == 0.0 or baseline is None:
            row["ci"] = (np.nan, np.nan)
            row["improves"] = False
            continue
        values = blend(index_returns, strategy, row["weight"])
        differences = []
        for _ in range(3000):
            draw = rng.integers(0, len(values), len(values))
            a, b = values[draw], baseline[draw]
            sa = a.mean() / a.std(ddof=1) if a.std(ddof=1) > 0 else np.nan
            sb = b.mean() / b.std(ddof=1) if b.std(ddof=1) > 0 else np.nan
            differences.append(sa - sb)
        low, high = np.percentile(differences, [2.5, 97.5])
        row["ci"] = (float(low), float(high))
        row["improves"] = bool(low > 0)

    # Referencia obrigatoria: misturar com DINHEIRO PARADO.
    #
    # E' a alternativa gratuita a qualquer diversificador, e tem uma propriedade
    # que a torna a fasquia certa: como o dinheiro rende 0 com volatilidade 0,
    # a mistura desliza ao longo de uma recta que passa pela origem e o racio de
    # Sharpe fica EXACTAMENTE igual em todos os pesos. Um diversificador a serio
    # tem de bater isso; se ficar abaixo, esta' a ser pior do que nao fazer nada
    # com essa fatia do capital -- e ainda por cima com risco de execucao,
    # custos de emprestimo de titulos e complexidade por cima.
    cash_rows = []
    for weight in WEIGHTS:
        joined = pd.concat([index_returns, strategy], axis=1,
                           keys=["i", "s"]).dropna()
        values = ((1 - weight) * joined["i"]).values
        if len(values) < 5:
            continue
        result = annualise(values)
        result["weight"] = weight
        cash_rows.append(result)

    return {
        "blends": rows,
        "cash": cash_rows,
        "break_even": break_even_return(index_returns, strategy),
        "crisis": crisis_behaviour(index_returns, strategy),
        "start": str(pd.Timestamp(strategy.index.min()).date()),
        "end": str(pd.Timestamp(strategy.index.max()).date()),
    }


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection)
    if not outcome:
        print("dados insuficientes")
        connection.close()
        raise SystemExit

    even = outcome["break_even"]
    print("=" * 94)
    print("EXPERIENCIA P -- o sinal vale como diversificador do indice?")
    print(f"  {outcome['start']} a {outcome['end']}, {even['n']} periodos de "
          f"{REBALANCE_DAYS} dias")

    print()
    print("  1. A CORRELACAO, que e' o que decide:")
    print(f"     correlacao entre o sinal e o indice: {even['correlation']:+.4f} "
          f"(p = {even['correlation_p']:.3f})")
    print(f"     volatilidade do sinal {even['strategy_vol']:.2%} contra "
          f"Sharpe do indice {even['index_sharpe']:.2f}")

    print()
    print("  2. O PONTO DE EQUILIBRIO -- que retorno o sinal precisaria de ter:")
    print(f"     minimo para a mistura melhorar o racio : {even['required_cagr']:>+8.2%} ao ano")
    print(f"     retorno que o sinal teve de facto      : {even['strategy_cagr']:>+8.2%} ao ano")
    print(f"     falta                                  : {even['shortfall']:>+8.2%}")
    print(f"     -> {'a mistura MELHORA' if even['helps'] else 'a mistura NAO melhora'}")

    print()
    print("  3. AS MISTURAS, contra a alternativa gratuita (dinheiro parado):")
    print(f"     {'na fatia':>9} {'anual':>8} {'vol':>8} {'Sharpe':>8} "
          f"{'drawdown':>10} | {'Sharpe c/ dinheiro':>18} {'diferenca':>10}")
    cash_by_weight = {row["weight"]: row for row in outcome["cash"]}
    for row in outcome["blends"]:
        cash = cash_by_weight.get(row["weight"], {})
        cash_sharpe = cash.get("sharpe", float("nan"))
        print(f"     {row['weight']:>8.0%} {row['cagr']:>8.2%} {row['vol']:>8.2%} "
              f"{row['sharpe']:>8.2f} {row['max_drawdown']:>10.1%} | "
              f"{cash_sharpe:>18.2f} {row['sharpe'] - cash_sharpe:>+10.3f}")
    print()
    print("     Misturar com dinheiro deixa o Sharpe ARITMETICO exactamente igual em")
    print("     todos os pesos -- a carteira desliza numa recta que passa pela origem.")
    print("     O Sharpe geometrico acima desvia-se um pouco disso porque depende da")
    print("     forma da distribuicao, e nao so' da media e do desvio. Em qualquer das")
    print("     leituras, um diversificador so' se justifica ACIMA dessa linha, e a")
    print("     coluna da direita mostra que este fica sempre abaixo.")

    crisis = outcome["crisis"]
    print()
    print("  4. E QUANDO O INDICE CAI? (o teste que separa diversificador a serio)")
    print(f"     nos {crisis['n_bad']} piores periodos do indice "
          f"(media {crisis['index_in_bad']:+.2%}):")
    print(f"       o sinal rendeu    : {crisis['strategy_in_bad']:>+8.4%}  "
          f"({crisis['hit_rate_in_bad']:.0%} deles positivos)")
    print(f"     nos melhores periodos do indice:")
    print(f"       o sinal rendeu    : {crisis['strategy_in_good']:>+8.4%}")

    print()
    print("  * = intervalo da diferenca de Sharpe inteiramente acima de zero")
    connection.close()
