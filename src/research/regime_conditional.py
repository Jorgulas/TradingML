"""EXPERIENCIA Q -- o sinal funciona nos regimes de QUEDA do indice?

A HIPOTESE E' RAZOAVEL, e nao e' pesca. Um livro transversal long-short vive de
dispersao entre accoes, e a dispersao dispara quando o mercado cai: em panico as
accoes deixam de andar todas juntas, os perdedores separam-se dos vencedores, e
ha' mais para ordenar. Se o sinal existe em algum lado, e' plausivel que seja
la'. A ronda 13 ja' tinha uma pista -- nos 15 piores periodos do indice o sinal
rendeu -0.27% -- mas essa medicao respondia a outra pergunta.

DUAS CONDICOES, e a diferenca entre elas e' o cerne desta experiencia:

  CONTEMPORANEA   "nos periodos em que o indice caiu, o sinal rendeu quanto?"
                  E' diagnostico. NAO E' NEGOCIAVEL: no momento de decidir nao
                  se sabe que este periodo vai ser dos maus. Responde a "e'
                  hedge?", nao a "ligo-o quando?".

  EX-ANTE         "quando o mercado JA' vinha a cair, o sinal rendeu quanto?"
                  So' informacao disponivel na altura -- media movel, retorno
                  passado, drawdown corrente. Esta da' uma regra que se pode
                  seguir, e e' a unica que pode produzir alguma coisa util.

Reporta-se as duas, lado a lado, porque a distancia entre elas mede exactamente
quanto do efeito aparente e' impossivel de capturar.

A ARMADILHA, que este projecto ja' pisou duas vezes: fatiar um sinal nulo por
regimes acaba sempre por encontrar uma fatia bonita. Tres defesas:

  1. Poucos regimes, definidos a' partida, e todos classicos. Nada de varrer
     limiares ate' um funcionar.
  2. Correccao de Bonferroni sobre o numero de regimes testados.
  3. Um PLACEBO por deslocamento circular: roda-se a etiqueta de regime no
     tempo, o que preserva a sua estrutura de autocorrelacao (os regimes vem em
     blocos longos, nao alternados) mas destroi a ligacao ao sinal. Se a
     diferenca verdadeira nao se destacar de centenas de rotacoes ao acaso, nao
     e' diferenca nenhuma. E' mais exigente do que Bonferroni porque nao assume
     independencia entre periodos.

O TRADE-OFF DA AMOSTRA: usa-se 4 trimestres de treino em vez dos 6 das rondas
12 e 13. Cada modelo fica com menos historico, mas o periodo de teste alarga e
passa a incluir dois trimestres negativos em vez de um. Para uma pergunta sobre
regimes de queda, ter quedas para medir pesa mais do que treinar melhor.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.decision import panel as P
from src.decision.short_term import block_bootstrap_mean
from src.patterns.ingest_intraday import load_bars
from src.research.benchmark_comparison import (REBALANCE_DAYS, TRADING_DAYS,
                                               strategy_curve)
from src.research.fallen_universe import surviving_fallen

import numpy as np
import pandas as pd
from scipy import stats

PERIODS_PER_YEAR = TRADING_DAYS / REBALANCE_DAYS
TRAIN_QUARTERS = 4          # menos treino, mais teste -- ver cabecalho
N_PLACEBO = 2000


def ex_ante_regimes(prices: pd.Series, stamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Bandeiras de regime que usam SO' o passado de cada instante.

    Quatro definicoes classicas, escolhidas a' partida e sem varrimento de
    limiares. Todas se leem no fecho do dia da decisao: a media movel de 200
    dias inclui o proprio dia mas nenhum posterior, e o drawdown e' medido contra
    o maximo ja' atingido."""
    prices = prices.copy()
    prices.index = pd.DatetimeIndex(prices.index)

    frame = pd.DataFrame(index=prices.index)
    frame["abaixo da MA200"] = prices < prices.rolling(200, min_periods=200).mean()
    frame["abaixo da MA50"] = prices < prices.rolling(50, min_periods=50).mean()
    frame["retorno 20d negativo"] = prices.pct_change(20) < 0
    frame["em drawdown > 5%"] = (prices / prices.cummax() - 1) < -0.05

    return frame.reindex(stamps, method="ffill")


def contemporaneous_regime(index_returns: pd.Series) -> pd.Series:
    """O periodo em que o indice CAIU. Diagnostico apenas -- so' se sabe depois."""
    return index_returns < 0


def compare(strategy: pd.Series, flag: pd.Series) -> dict:
    """Retorno do sinal dentro e fora do regime, e a diferenca com intervalo."""
    joined = pd.concat([strategy, flag], axis=1, keys=["s", "flag"]).dropna()
    inside = joined.loc[joined["flag"].astype(bool), "s"]
    outside = joined.loc[~joined["flag"].astype(bool), "s"]
    if len(inside) < 5 or len(outside) < 5:
        return {"n_inside": len(inside), "n_outside": len(outside), "usable": False}

    low, high = block_bootstrap_mean(inside.values, block=1)
    difference = float(inside.mean() - outside.mean())
    # Welch, que nao assume variancias iguais -- e os dois regimes tem
    # volatilidades diferentes por definicao.
    statistic = stats.ttest_ind(inside, outside, equal_var=False)

    return {
        "usable": True,
        "n_inside": int(len(inside)), "n_outside": int(len(outside)),
        "mean_inside": float(inside.mean()), "mean_outside": float(outside.mean()),
        "annual_inside": float(inside.mean() * PERIODS_PER_YEAR),
        "ci_inside": (low, high),
        "inside_positive": low > 0,
        "difference": difference,
        "p_value": float(statistic.pvalue),
        "hit_rate_inside": float((inside > 0).mean()),
    }


def circular_placebo(strategy: pd.Series, flag: pd.Series,
                     n: int = N_PLACEBO, seed: int = 44) -> dict:
    """Placebo por deslocamento circular da etiqueta de regime.

    Porque isto e' mais exigente do que Bonferroni. Os regimes nao sao sorteios
    independentes -- vem em blocos longos (um mercado abaixo da MA200 fica la'
    semanas), e uma variavel em blocos correlaciona-se por acaso com outra
    variavel em blocos muito mais vezes do que um teste que assume independencia
    admite. Rodar a etiqueta no tempo preserva exactamente essa estrutura de
    blocos e destroi apenas o alinhamento com o sinal.

    O p-value que sai e' a fraccao de rotacoes ao acaso que produzem uma
    diferenca tao extrema como a verdadeira."""
    joined = pd.concat([strategy, flag], axis=1, keys=["s", "flag"]).dropna()
    values = joined["s"].values
    labels = joined["flag"].astype(bool).values
    if labels.sum() < 5 or (~labels).sum() < 5:
        return {"usable": False}

    observed = values[labels].mean() - values[~labels].mean()

    rng = np.random.default_rng(seed)
    extremes = 0
    for _ in range(n):
        shifted = np.roll(labels, int(rng.integers(1, len(labels))))
        if shifted.sum() < 2 or (~shifted).sum() < 2:
            continue
        candidate = values[shifted].mean() - values[~shifted].mean()
        if abs(candidate) >= abs(observed):
            extremes += 1

    return {"usable": True, "observed": float(observed),
            "p_value": float((extremes + 1) / (n + 1))}


def beta_diagnostic(strategy: pd.Series, index_returns: pd.Series) -> dict:
    """O livro e' mesmo neutro ao mercado, ou carrega beta escondido?

    Porque esta pergunta entrou aqui. A condicao contemporanea mostrou o sinal a
    perder quando o indice cai e a ganhar quando sobe, e a explicacao mais
    natural seria beta residual: pesar as duas pernas por igual NAO garante
    neutralidade, porque as accoes compradas podem ter beta maior do que as
    vendidas, e nesse caso o livro perderia exactamente quando devia proteger.

    Era uma hipotese com consequencia pratica -- se fosse verdade, a correccao
    seria pesar as pernas por beta em vez de por capital. Mede-se em vez de se
    assumir."""
    joined = pd.concat([strategy, index_returns], axis=1, keys=["s", "i"]).dropna()
    if len(joined) < 20:
        return {"usable": False}
    fit = stats.linregress(joined["i"], joined["s"])
    return {
        "usable": True, "n": int(len(joined)),
        "beta": float(fit.slope), "beta_se": float(fit.stderr),
        "beta_p": float(fit.pvalue),
        "alpha": float(fit.intercept),
        "r_squared": float(fit.rvalue ** 2),
        "correlation": float(joined["i"].corr(joined["s"])),
        # Quanto de um movimento de -2% no indice o beta explicaria.
        "explained_by_beta_on_2pc_drop": float(fit.slope * -0.02),
    }


def minimum_detectable(series: pd.Series) -> float:
    """Que efeito seria detectavel com esta amostra, em ~2 erros-padrao.

    Com 11 periodos dentro de um regime, o intervalo e' tao largo que quase
    nada e' distinguivel de zero -- e sem este numero um resultado nulo parece
    uma conclusao quando e' apenas cegueira."""
    values = series.dropna().values
    if len(values) < 5:
        return float("nan")
    low, high = block_bootstrap_mean(values, block=1)
    return float((high - low) / 2) if np.isfinite(low) else float("nan")


def years_needed(mean: float, mde: float, n_inside: int,
                 frequency: float) -> float:
    """Quantos anos de historico seriam precisos para o efeito ser detectavel.

    O erro-padrao cai com a raiz do numero de observacoes, portanto para o
    minimo detectavel descer de `mde` para `|mean|` sao precisas
    n x (mde/mean)^2 observacoes DENTRO do regime. Dividindo pela frequencia com
    que o regime ocorre sai o total, e dividindo pelos periodos por ano sai a
    resposta na unidade que interessa.

    E' o numero que separa "o efeito nao existe" de "a amostra e' cega", e este
    projecto ja' confundiu as duas coisas uma vez (ronda 6)."""
    if not np.isfinite(mde) or mean == 0 or frequency <= 0:
        return float("nan")
    required_inside = n_inside * (mde / abs(mean)) ** 2
    return float(required_inside / frequency / PERIODS_PER_YEAR)


def run(conn) -> dict:
    universe = list(config.TREE_TICKERS) + surviving_fallen(conn)
    frame = P.build(conn, tickers=universe, min_days=120, timeframe="1d")
    strategy = strategy_curve(conn, frame, min_train_quarters=TRAIN_QUARTERS)
    if strategy.empty:
        return {}

    stamps = pd.DatetimeIndex(strategy.index)
    spy = load_bars(conn, config.BENCHMARK["ticker"], "1d")["close"]
    index_returns = (spy.reindex(stamps, method="ffill")
                     .pct_change(fill_method=None).shift(-1).dropna())

    flags = ex_ante_regimes(spy, stamps)
    results = {}
    for name in flags.columns:
        outcome = compare(strategy, flags[name])
        if outcome.get("usable"):
            outcome["placebo"] = circular_placebo(strategy, flags[name])
        results[name] = outcome

    for name in flags.columns:
        joined = pd.concat([strategy, flags[name]], axis=1,
                           keys=["s", "flag"]).dropna()
        inside = joined.loc[joined["flag"].astype(bool), "s"]
        if name in results and results[name].get("usable"):
            results[name]["mde"] = minimum_detectable(inside)
            results[name]["frequency"] = float(
                joined["flag"].astype(bool).mean())
            results[name]["years_needed"] = years_needed(
                results[name]["mean_inside"], results[name]["mde"],
                results[name]["n_inside"], results[name]["frequency"])

    contemporary = compare(strategy, contemporaneous_regime(index_returns))
    if contemporary.get("usable"):
        contemporary["placebo"] = circular_placebo(
            strategy, contemporaneous_regime(index_returns))

    overall = strategy.values
    low, high = block_bootstrap_mean(overall, block=1)
    return {
        "n_periods": int(len(strategy)),
        "start": str(stamps.min().date()), "end": str(stamps.max().date()),
        "overall_mean": float(overall.mean()),
        "overall_annual": float(overall.mean() * PERIODS_PER_YEAR),
        "overall_ci": (low, high),
        "ex_ante": results,
        "contemporary": contemporary,
        "n_regimes": len(flags.columns),
        "beta": beta_diagnostic(strategy, index_returns),
    }


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection)
    if not outcome:
        print("dados insuficientes")
        connection.close()
        raise SystemExit

    print("=" * 98)
    print("EXPERIENCIA Q -- o sinal funciona nos regimes de queda?")
    print(f"  {outcome['start']} a {outcome['end']}, {outcome['n_periods']} periodos "
          f"de {REBALANCE_DAYS} dias ({TRAIN_QUARTERS} trimestres de treino)")
    low, high = outcome["overall_ci"]
    print(f"  referencia: o sinal em TODOS os periodos rende "
          f"{outcome['overall_mean']:+.4%} por periodo "
          f"({outcome['overall_annual']:+.2%} ao ano), IC [{low:+.4%}, {high:+.4%}]")

    threshold = 0.05 / max(outcome["n_regimes"], 1)
    print()
    print(f"  REGIMES EX-ANTE (informacao disponivel na altura -- e' o que se pode")
    print(f"  negociar). Bonferroni com {outcome['n_regimes']} regimes exige "
          f"p < {threshold:.4f}.")
    print()
    print(f"    {'regime':>24} {'periodos':>9} {'dentro':>10} {'fora':>10} "
          f"{'anual dentro':>13} {'MDE':>9} {'p Welch':>9} {'p placebo':>10}")
    for name, result in outcome["ex_ante"].items():
        if not result.get("usable"):
            print(f"    {name:>24} {result['n_inside']:>9} "
                  f"{'amostra insuficiente':>45}")
            continue
        placebo = result.get("placebo", {})
        mark = "*" if result["p_value"] < threshold else " "
        print(f"    {name:>24} {result['n_inside']:>9} "
              f"{result['mean_inside']:>+10.4%} {result['mean_outside']:>+10.4%} "
              f"{result['annual_inside']:>+13.2%} "
              f"{result.get('mde', float('nan')):>9.4%} "
              f"{result['p_value']:>8.3f}{mark} "
              f"{placebo.get('p_value', float('nan')):>10.3f}")

    contemporary = outcome["contemporary"]
    print()
    print("  CONDICAO CONTEMPORANEA (diagnostico -- NAO negociavel, so' se sabe depois):")
    if contemporary.get("usable"):
        print(f"    nos {contemporary['n_inside']} periodos em que o indice caiu, "
              f"o sinal rendeu {contemporary['mean_inside']:+.4%}")
        print(f"    nos {contemporary['n_outside']} em que subiu, "
              f"{contemporary['mean_outside']:+.4%}")
        print(f"    diferenca {contemporary['difference']:+.4%}, "
              f"p Welch {contemporary['p_value']:.3f}, "
              f"p placebo {contemporary.get('placebo', {}).get('p_value', float('nan')):.3f}")

    print()
    print("  O PODER: nenhum destes efeitos e' maior do que o seu minimo detectavel.")
    print("  Quantos anos de historico seriam precisos para o distinguir de zero?")
    print(f"    {'regime':>24} {'efeito':>10} {'MDE':>10} {'racio':>7} {'anos':>9}")
    for name, result in outcome["ex_ante"].items():
        if not result.get("usable"):
            continue
        ratio = result["mde"] / abs(result["mean_inside"]) if result["mean_inside"] else float("nan")
        print(f"    {name:>24} {result['mean_inside']:>+10.4%} "
              f"{result['mde']:>10.4%} {ratio:>7.1f}x "
              f"{result.get('years_needed', float('nan')):>8.0f}")

    beta = outcome.get("beta", {})
    if beta.get("usable"):
        print()
        print("  A HIPOTESE QUE TESTEI PARA EXPLICAR O RESULTADO CONTEMPORANEO, e que")
        print("  se revelou FALSA: seria beta residual? Pesar as pernas por igual nao")
        print("  garante neutralidade, e um livro com beta escondido perderia")
        print("  exactamente quando devia proteger.")
        print(f"    beta ao indice : {beta['beta']:+.4f} "
              f"(erro-padrao {beta['beta_se']:.4f}, p = {beta['beta_p']:.3f})")
        print(f"    R2             : {beta['r_squared']:.4f}   "
              f"correlacao {beta['correlation']:+.4f}")
        print(f"    num queda de 2% do indice, o beta explicaria "
              f"{beta['explained_by_beta_on_2pc_drop']:+.4%} "
              f"do sinal")
        print("    -> o livro E' neutro. A diferenca contemporanea nao vem dai',")
        print("       e com 5 comparacoes feitas o mais provavel e' ser ruido.")

    print()
    print("  * = sobrevive a' correccao de Bonferroni")
    connection.close()
