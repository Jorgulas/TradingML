"""EXPERIENCIA S -- a concentracao do risco subiu por causa das NOTICIAS AGENDADAS?

DE ONDE VEM. A ronda 15 mediu, nas mesmas 115 accoes presentes em todas as
decadas, a concentracao de risco a subir de 0.197 (anos 80) para 0.230 (anos
2020) com a volatilidade parada. Tendencia +0.00088/ano, p = 0.0003, e sem o ano
de 2020 mantem-se -- nao e' COVID.

A literatura tem a versao de MERCADO deste facto. Baker et al. (NBER w28687),
sobre 90 anos, mostram que os saltos do indice passaram de ser causados por
noticias NAO agendadas (guerras, nas decadas antigas) para noticias AGENDADAS,
sobretudo politica monetaria, nas recentes.

Mas isso e' o indice. O meu numero e' por EMPRESA, e ao nivel da empresa a
noticia agendada dominante nao e' a Reserva Federal -- sao os RESULTADOS
TRIMESTRAIS. Isso da' uma previsao mecanistica que se pode testar com os dados
que ha', e e' o que esta experiencia faz.

A DECOMPOSICAO QUE DECIDE:

    concentracao em TODOS os dias          <- o que a ronda 15 mediu
    concentracao EXCLUINDO dias de resultados

  Se a subida for feita de resultados, a segunda serie nao tem tendencia
  nenhuma. Se a tendencia sobreviver a retirar os resultados, o mecanismo e'
  outro -- e isso seria mais interessante do que a confirmacao.

E uma medida directa, mais facil de ler do que qualquer indice:

    intensidade = (quota da variancia anual que cai em dias de resultados)
                / (quota dos dias que sao de resultados)

  Vale 1 se um dia de resultados for um dia como os outros. Se valer 4, cada dia
  de resultados carrega quatro vezes a variancia de um dia normal. A pergunta e'
  se esse numero subiu ao longo de vinte anos.

RESSALVAS. O calendario de resultados so' existe a partir de 2005 com cobertura
decente, portanto testa-se 21 anos e nao 65 -- e a subida da ronda 15 e' medida
sobre decadas. Vinte e um anos chegam para ver a tendencia recente, nao para
explicar a de 1980. E as datas vem do yfinance como estao: um erro de um dia
numa data desloca a janela, o que ATENUA o efeito em vez de o inventar.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.research.deep_history import TIMEFRAME
from src.research.earnings_data import load as load_earnings
from src.research.tail_shape import load_returns, tail_measures

import numpy as np
import pandas as pd
from scipy import stats

WINDOW_DAYS = 1        # +-1 dia em torno da data anunciada
FIRST_YEAR = 2005      # antes disto o calendario tem cobertura fraca
MIN_DAYS = 200


def earnings_mask(stamps: pd.DatetimeIndex, dates: np.ndarray) -> np.ndarray:
    """Marca os dias dentro de +-WINDOW_DAYS de um anuncio de resultados.

    Janela e nao dia exacto: a empresa pode anunciar depois do fecho, e nesse
    caso a reaccao cai no dia seguinte. Uma janela de tres dias apanha as duas
    convencoes sem ter de adivinhar qual delas o yfinance usou."""
    if len(dates) == 0:
        return np.zeros(len(stamps), dtype=bool)
    days = stamps.normalize().tz_localize(None).values.astype("datetime64[D]")
    announced = np.unique(np.asarray(dates, dtype="datetime64[D]"))
    mask = np.zeros(len(days), dtype=bool)
    for offset in range(-WINDOW_DAYS, WINDOW_DAYS + 1):
        mask |= np.isin(days, announced + np.timedelta64(offset, "D"))
    return mask


def build(conn, tickers=None) -> pd.DataFrame:
    """Um registo por (ticker, ano) com a decomposicao dentro/fora de resultados."""
    earnings = load_earnings(conn)
    if earnings.empty:
        return pd.DataFrame()
    by_ticker = {t: g["earnings_date"].dt.tz_localize(None).values
                 for t, g in earnings.groupby("ticker")}

    tickers = tickers or sorted(by_ticker)
    rows = []
    for ticker in tickers:
        dates = by_ticker.get(ticker)
        if dates is None:
            continue
        returns = load_returns(conn, ticker)
        if returns.empty:
            continue

        stamps = pd.DatetimeIndex(returns.index)
        mask = earnings_mask(stamps, dates)
        years = stamps.year

        for year in np.unique(years):
            if year < FIRST_YEAR or year > 2025:
                continue
            selection = years == year
            values = returns.values[selection]
            flags = mask[selection]
            if len(values) < MIN_DAYS or flags.sum() < 4:
                continue

            squares = values ** 2
            total = squares.sum()
            if total <= 0:
                continue

            variance_share = float(squares[flags].sum() / total)
            day_share = float(flags.mean())
            outside = tail_measures(values[~flags])
            everything = tail_measures(values)
            if not outside or not everything:
                continue

            rows.append({
                "ticker": ticker, "period": int(year),
                "variance_share": variance_share,
                "day_share": day_share,
                # Quantas vezes a variancia de um dia normal carrega um dia de
                # resultados. E' o numero legivel desta experiencia.
                "intensity": variance_share / day_share if day_share > 0 else np.nan,
                "concentration": everything["concentration"],
                "concentration_ex": outside["concentration"],
                "volatility": everything["volatility"],
                "volatility_ex": outside["volatility"],
            })

    return pd.DataFrame(rows)


def constant_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """So' as accoes presentes em TODOS os anos.

    Sem isto, uma subida podia ser apenas composicao -- entraram empresas
    diferentes. E' o mesmo controlo da ronda 15, e aqui e' ainda mais preciso
    porque a cobertura do calendario cresce com o tempo."""
    years = set(frame["period"].unique())
    counts = frame.groupby("ticker")["period"].nunique()
    complete = counts[counts == len(years)].index
    return frame[frame["ticker"].isin(complete)]


def trend(frame: pd.DataFrame, column: str) -> dict:
    """Regressao da media anual transversal contra o ano."""
    yearly = frame.groupby("period")[column].mean().dropna()
    if len(yearly) < 8:
        return {}
    fit = stats.linregress(yearly.index, yearly.values)
    return {
        "column": column, "slope": float(fit.slope), "p_value": float(fit.pvalue),
        "r_squared": float(fit.rvalue ** 2),
        "first": float(yearly.iloc[0]), "last": float(yearly.iloc[-1]),
        "change": float(yearly.iloc[-1] - yearly.iloc[0]),
        "n_years": int(len(yearly)), "series": yearly,
    }


def slope_difference(frame: pd.DataFrame, first: str, second: str,
                     n: int = 4000, seed: int = 55) -> dict:
    """O declive de `first` e' mesmo diferente do de `second`?

    CORRIGE UM ERRO ESTATISTICO CLASSICO que esta experiencia cometeu na
    primeira versao: comparar os p-values das duas regressoes e concluir "uma e'
    significativa e a outra nao, logo sao diferentes". Nao sao a mesma coisa. Um
    declive pode cair 30% e perder significancia apenas por o intervalo ser
    largo, sem que a diferenca entre os dois seja distinguivel de zero.

    O teste certo e' sobre a DIFERENCA. Reamostram-se os anos em conjunto -- os
    dois declives sao medidos nos mesmos anos, portanto a comparacao e'
    emparelhada -- e ve-se se a diferenca exclui o zero."""
    left = frame.groupby("period")[first].mean().dropna()
    right = frame.groupby("period")[second].mean().dropna()
    years = left.index.intersection(right.index)
    if len(years) < 8:
        return {}

    left, right = left.loc[years], right.loc[years]
    observed = (stats.linregress(years, left.values).slope
                - stats.linregress(years, right.values).slope)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(years), len(years))
        sample_years = np.asarray(years)[pick]
        if len(np.unique(sample_years)) < 5:
            continue
        draws.append(stats.linregress(sample_years, left.values[pick]).slope
                     - stats.linregress(sample_years, right.values[pick]).slope)
    if not draws:
        return {}
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"difference": float(observed), "ci": (float(low), float(high)),
            "distinguishable": bool(low > 0 or high < 0)}


def run(conn) -> dict:
    frame = build(conn)
    if frame.empty:
        return {}
    stable = constant_panel(frame)
    if stable.empty or stable["ticker"].nunique() < 30:
        return {"n_all": int(frame["ticker"].nunique()), "insufficient": True}

    return {
        "n_all": int(frame["ticker"].nunique()),
        "n_stable": int(stable["ticker"].nunique()),
        "years": (int(stable["period"].min()), int(stable["period"].max())),
        "trends": {name: trend(stable, name)
                   for name in ("intensity", "variance_share", "concentration",
                                "concentration_ex", "volatility")},
        "slope_test": slope_difference(stable, "concentration", "concentration_ex"),
        "frame": stable,
    }


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection)
    if not outcome or outcome.get("insufficient"):
        print("calendario de resultados insuficiente -- corre "
              "src/research/earnings_data.py")
        connection.close()
        raise SystemExit

    trends = outcome["trends"]
    print("=" * 94)
    print("EXPERIENCIA S -- a concentracao subiu por causa das noticias agendadas?")
    print(f"  {outcome['n_stable']} accoes presentes em TODOS os anos "
          f"(de {outcome['n_all']} com calendario), "
          f"{outcome['years'][0]} a {outcome['years'][1]}")

    intensity = trends.get("intensity", {})
    if intensity:
        print()
        print("  1. QUANTO PESA UM DIA DE RESULTADOS?")
        print(f"     intensidade = variancia que cai em dias de resultados, "
              f"dividida pela")
        print(f"     fraccao de dias que sao de resultados. Vale 1 se fosse um "
              f"dia normal.")
        print(f"       {outcome['years'][0]}: {intensity['first']:.2f}x   "
              f"{outcome['years'][1]}: {intensity['last']:.2f}x   "
              f"({intensity['change']:+.2f})")
        print(f"       tendencia {intensity['slope']:+.4f}/ano, "
              f"p = {intensity['p_value']:.4f}, R2 = {intensity['r_squared']:.3f}")

    print()
    print("  2. A DECOMPOSICAO QUE DECIDE:")
    print(f"     {'serie':>24} {'inicio':>9} {'fim':>9} {'variacao':>10} "
          f"{'por ano':>10} {'p':>9}")
    for name, label in (("concentration", "concentracao (tudo)"),
                        ("concentration_ex", "SEM dias de resultados"),
                        ("volatility", "volatilidade (controlo)")):
        row = trends.get(name)
        if not row:
            continue
        print(f"     {label:>24} {row['first']:>9.4f} {row['last']:>9.4f} "
              f"{row['change']:>+10.4f} {row['slope']:>+10.5f} "
              f"{row['p_value']:>9.4f}")

    whole = trends.get("concentration", {})
    without = trends.get("concentration_ex", {})
    test = outcome.get("slope_test", {})
    if whole and without and test:
        share = 1 - without["slope"] / whole["slope"] if whole["slope"] else np.nan
        low, high = test["ci"]
        print()
        print(f"     Os resultados explicam {share:.0%} do declive. Mas a pergunta")
        print(f"     nao e' se cada serie e' significativa -- e' se as duas SAO")
        print(f"     DIFERENTES uma da outra, que e' outro teste:")
        print(f"       diferenca de declives {test['difference']:+.5f}/ano, "
              f"IC95% [{low:+.5f}, {high:+.5f}]")
        if test["distinguishable"]:
            verdict = ("as duas series sobem a ritmos mesmo diferentes: os "
                       "resultados carregam parte real da subida")
        else:
            verdict = ("as duas series nao se distinguem uma da outra. Retirar os "
                       "dias de resultados NAO faz a tendencia desaparecer, e a "
                       "intensidade deles esta' plana -- portanto o mecanismo de "
                       "noticias agendadas NAO explica esta subida")
        print(f"     -> {verdict}")

    frame = outcome["frame"]
    print()
    print("  3. ANO A ANO (intensidade dos dias de resultados):")
    series = intensity.get("series")
    if series is not None:
        for year, value in series.items():
            bar = "#" * int(round(value * 6))
            print(f"     {int(year)}  {value:>5.2f}x  {bar}")

    connection.close()
