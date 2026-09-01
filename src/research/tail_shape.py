"""EXPERIENCIA R -- a FORMA da cauda e' uma propriedade persistente da accao?

A PERGUNTA, e porque acho que vale a pena. Catorze rondas estabeleceram que a
direccao nao e' previsivel e que a magnitude e'. Toda a gente mede magnitude como
VOLATILIDADE. Mas volatilidade e' um nivel, e ha' uma dimensao separada que este
projecto nunca mediu: a FORMA com que essa volatilidade e' entregue.

Duas accoes podem ter exactamente a mesma volatilidade anual e comportar-se de
maneiras completamente diferentes:

    A entrega-a em muitos movimentos medios     (cauda fina)
    B entrega-a em poucos movimentos enormes    (cauda grossa)

Para quem gere risco isto nao e' o mesmo. Volatilidade igual, risco de ruina
diferente. E a pergunta e': a forma e' CARACTER da accao -- estavel, previsivel,
uma propriedade dela -- ou e' so' ruido que muda de periodo para periodo?

PORQUE ACHO QUE PODE SER REAL. Ha' mecanismo: a forma da cauda reflecte o
negocio (um ensaio clinico e' binario, uma electrica nao), a estrutura de
accionistas, a liquidez. Essas coisas mudam devagar. E a ronda 6 mostrou que os
extremos se concentram em EVENTOS AGENDADOS -- mas isso era sobre QUANDO. Esta
pergunta e' sobre QUEM.

O QUE A TORNA TESTAVEL AGORA. Ate' aqui tinha tres anos de historico. Com o
historico completo tenho ate' 65 anos, e isso permite a fasquia que nenhum
resultado deste projecto enfrentou: o efeito tem de aparecer nos anos 70, 80,
90, 2000, 2010 e 2020 -- cinco transicoes de decada independentes, com mercados,
tecnologia, microestrutura e regulacao completamente diferentes. O modo de
falhar deste projecto foi sempre "encontrei e nao replicou"; aqui a replicacao
esta' construida dentro do desenho.

AS TRES DEFESAS QUE ESTA EXPERIENCIA PRECISA:

1. AS MEDIDAS TEM DE SER SEM ESCALA. Se a medida de cauda subir com a
   volatilidade, estou a remedir persistencia de volatilidade -- que e' conhecida
   desde os anos 80 e nao seria novidade nenhuma. Todas as medidas aqui sao
   racios ou quotas, e a correlacao com a volatilidade e' medida e reportada.

2. TEM DE BATER A VOLATILIDADE como preditor da cauda futura. E' a pergunta de
   sempre neste projecto: acrescenta alguma coisa a' alternativa mais simples?

3. PLACEBO POR TROCA DE IDENTIDADE. Baralham-se os tickers entre periodos. Se a
   persistencia sobreviver a isso, nao e' persistencia -- e' estrutura comum a
   todo o mercado a passar por identidade individual.

RESSALVA QUE NAO SE PODE OMITIR: o universo sao as grandes capitalizacoes de
HOJE, e uma empresa com 65 anos de historico que ainda esta' no indice e' um
sobrevivente extremo. A conclusao possivel e' "entre accoes que sobreviveram, a
forma da cauda persiste". Nao e' fatal -- mede-se uma propriedade DENTRO de cada
serie, nao uma comparacao de retornos -- mas limita a generalizacao.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.research.deep_history import TIMEFRAME

import numpy as np
import pandas as pd
from scipy import stats

MIN_DAYS_YEAR = 200         # um ano util tem ~252 dias
MIN_DAYS_DECADE = 1200      # meia decada de negociacao chega
# Referencia gaussiana para q99/q75 de |r|, e o valor certo NAO e' obvio.
#
# A primeira versao usava 2.326/1.150 = 2.02, com 2.326 o percentil 99
# UNILATERAL da normal. Mas a medida e' sobre o VALOR ABSOLUTO, e para |X| com
# X ~ N(0,1) tem-se P(|X| < c) = 2*Phi(c) - 1, logo o percentil 99 e'
# Phi^-1(0.995) = 2.576 e nao 2.326. O 1.150 do denominador ja' estava certo
# (Phi^-1(0.875)). Confirmado por simulacao com 2 milhoes de amostras.
#
# Nao e' detalhe: com a referencia errada a 2.02, uma cauda observada de 2.97
# parecia 47% acima da normal quando esta' 33% acima.
GAUSSIAN_TAIL_RATIO = 2.576 / 1.150   # o que uma normal daria: ~2.24


def load_returns(conn, ticker: str) -> pd.Series:
    """Retornos logaritmicos diarios, ja' ajustados a desdobramentos."""
    rows = conn.execute(
        "SELECT ts, close FROM intraday_prices WHERE ticker = ? AND timeframe = ? "
        "ORDER BY ts", (ticker, TIMEFRAME)).fetchall()
    if len(rows) < 100:
        return pd.Series(dtype=float)
    frame = pd.DataFrame([dict(r) for r in rows])
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, format="ISO8601")
    price = frame.set_index("ts")["close"]
    return np.log(price).diff().dropna()


def tail_measures(returns: np.ndarray) -> dict:
    """Tres leituras da FORMA, todas sem escala.

    Sem escala e' a exigencia central: se subissem com a volatilidade, isto
    seria um estudo de persistencia de volatilidade com outro nome.

      tail_ratio    q99(|r|) / q75(|r|). Quanto maior e' um dia de 1-em-100 face
                    a um dia agitado normal. Uma normal da' ~2.02; acima disso e'
                    cauda gorda. Numerador e denominador escalam ambos com sigma,
                    logo o racio nao.

      concentration Quota da variancia total entregue pelo 1% de dias piores.
                    "Quao concentrado esta' o risco desta accao." Uma quota e'
                    adimensional por construcao.

      hill          Estimador de Hill sobre os 5% maiores. E' a ferramenta
                    classica da teoria de valores extremos, e mede o expoente da
                    cauda: MENOR = cauda mais gorda. Entra por ser independente
                    das outras duas -- se as tres concordarem, e' sinal de que
                    ha' mesmo uma propriedade por baixo e nao um artefacto de
                    uma definicao.
    """
    # Filtra-se UMA vez, e tudo o que se segue vem do mesmo vector. A primeira
    # versao filtrava `absolute` e indexava `squares` construido do vector
    # original -- mascaras de tamanhos diferentes, e o erro so' aparecia nos
    # tickers que tinham dias de retorno exactamente zero.
    returns = returns[np.isfinite(returns)]
    if len(returns) < 100:
        return {}
    absolute = np.abs(returns)
    squares = returns ** 2

    # Dias de retorno ZERO ficam. Sao observacoes a serio -- o preco nao se
    # mexeu -- e retira-los deslocaria todos os quantis para cima, inflacionando
    # artificialmente a cauda das accoes mais paradas.
    q75, q99 = np.quantile(absolute, [0.75, 0.99])
    concentration = squares[absolute >= q99].sum() / squares.sum()

    # O estimador de Hill e' o unico que precisa de valores estritamente
    # positivos, porque toma logaritmos de racios. Filtra-se so' aqui.
    positive = np.sort(absolute[absolute > 0])[::-1]
    hill = np.nan
    if len(positive) > 40:
        k = max(10, int(len(positive) * 0.05))
        if k < len(positive) and positive[k] > 0:
            hill = float(k / np.sum(np.log(positive[:k] / positive[k])))

    return {
        "tail_ratio": float(q99 / q75) if q75 > 0 else np.nan,
        "concentration": float(concentration),
        "hill": hill,
        "volatility": float(returns.std(ddof=1)),
        "n": int(len(absolute)),
    }


def panel(conn, tickers=None, by: str = "year") -> pd.DataFrame:
    """Uma linha por (ticker, periodo) com as medidas de forma e a volatilidade."""
    tickers = tickers or [r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM intraday_prices WHERE timeframe = ?",
        (TIMEFRAME,)).fetchall()]
    minimum = MIN_DAYS_YEAR if by == "year" else MIN_DAYS_DECADE

    rows = []
    for ticker in tickers:
        returns = load_returns(conn, ticker)
        if returns.empty:
            continue
        stamps = pd.DatetimeIndex(returns.index)
        key = stamps.year if by == "year" else (stamps.year // 10) * 10

        for period in np.unique(key):
            values = returns.values[key == period]
            if len(values) < minimum:
                continue
            measures = tail_measures(values)
            if not measures:
                continue
            measures.update({"ticker": ticker, "period": int(period)})
            rows.append(measures)

    return pd.DataFrame(rows)


def is_it_just_volatility(frame: pd.DataFrame) -> pd.DataFrame:
    """Correlacao transversal entre cada medida de forma e a volatilidade.

    Se for alta, a medida nao e' sem escala na pratica e o estudo colapsa em
    persistencia de volatilidade. E' a primeira coisa a verificar, antes de
    qualquer resultado."""
    rows = []
    for measure in ("tail_ratio", "concentration", "hill"):
        by_period = []
        for period, group in frame.groupby("period"):
            if len(group) < 20:
                continue
            by_period.append(stats.spearmanr(group[measure],
                                             group["volatility"]).statistic)
        by_period = [v for v in by_period if np.isfinite(v)]
        if by_period:
            rows.append({"medida": measure,
                         "correlacao_media_com_vol": float(np.mean(by_period)),
                         "periodos": len(by_period)})
    return pd.DataFrame(rows)


def persistence(frame: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Correlacao de postos entre o periodo t e o t+1, periodo a periodo.

    E' transversal dentro de cada transicao: pergunta se as accoes MANTEM a sua
    ordem, nao se o nivel geral se mantem. Assim o que for comum ao mercado
    inteiro naquele periodo nao conta."""
    periods = sorted(frame["period"].unique())
    rows = []
    for earlier, later in zip(periods, periods[1:]):
        left = frame[frame["period"] == earlier].set_index("ticker")[measure]
        right = frame[frame["period"] == later].set_index("ticker")[measure]
        common = left.index.intersection(right.index)
        if len(common) < 20:
            continue
        rho = stats.spearmanr(left.loc[common], right.loc[common]).statistic
        rows.append({"de": earlier, "para": later, "n_tickers": len(common),
                     "spearman": float(rho)})
    return pd.DataFrame(rows)


def beats_volatility(frame: pd.DataFrame, measure: str) -> pd.DataFrame:
    """A forma passada preve a forma futura melhor do que a volatilidade passada?

    A pergunta de sempre deste projecto: acrescenta alguma coisa a' alternativa
    mais simples? Se a volatilidade de ontem previr a cauda de amanha tao bem
    como a cauda de ontem, nao ha' dimensao nova nenhuma."""
    periods = sorted(frame["period"].unique())
    rows = []
    for earlier, later in zip(periods, periods[1:]):
        left = frame[frame["period"] == earlier].set_index("ticker")
        right = frame[frame["period"] == later].set_index("ticker")
        common = left.index.intersection(right.index)
        if len(common) < 20:
            continue
        target = right.loc[common, measure]
        rows.append({
            "de": earlier, "para": later, "n_tickers": len(common),
            "pela_forma": float(stats.spearmanr(left.loc[common, measure],
                                                target).statistic),
            "pela_volatilidade": float(stats.spearmanr(left.loc[common, "volatility"],
                                                       target).statistic),
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["vantagem"] = table["pela_forma"] - table["pela_volatilidade"]
    return table


def identity_placebo(frame: pd.DataFrame, measure: str, n: int = 500,
                     seed: int = 77) -> dict:
    """Baralha os tickers entre periodos e refaz a persistencia.

    Se ela sobreviver, o que se estava a medir nao era identidade individual --
    era estrutura comum ao mercado a passar por propriedade da accao."""
    rng = np.random.default_rng(seed)
    observed = persistence(frame, measure)["spearman"].mean()

    extremes = 0
    for _ in range(n):
        shuffled = frame.copy()
        # Baralha o ticker DENTRO de cada periodo: preserva a distribuicao de
        # cada periodo e destroi apenas a ligacao entre periodos.
        shuffled["ticker"] = (shuffled.groupby("period")["ticker"]
                              .transform(lambda s: rng.permutation(s.values)))
        candidate = persistence(shuffled, measure)["spearman"].mean()
        if np.isfinite(candidate) and abs(candidate) >= abs(observed):
            extremes += 1

    return {"observed": float(observed),
            "p_value": float((extremes + 1) / (n + 1))}


def decade_terciles(frame: pd.DataFrame, measure: str) -> pd.DataFrame:
    """O teste de fora da amostra: ordena numa decada, mede na seguinte."""
    periods = sorted(frame["period"].unique())
    rows = []
    for earlier, later in zip(periods, periods[1:]):
        left = frame[frame["period"] == earlier].set_index("ticker")[measure]
        right = frame[frame["period"] == later].set_index("ticker")[measure]
        common = left.index.intersection(right.index)
        if len(common) < 30:
            continue
        groups = pd.qcut(left.loc[common], 3, labels=["fina", "media", "grossa"],
                         duplicates="drop")
        outcome = right.loc[common].groupby(groups, observed=True).mean()
        if len(outcome) < 3:
            continue
        rows.append({"de": earlier, "para": later, "n": len(common),
                     "fina": float(outcome.get("fina", np.nan)),
                     "media": float(outcome.get("media", np.nan)),
                     "grossa": float(outcome.get("grossa", np.nan))})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["separacao"] = table["grossa"] - table["fina"]
    return table


# ---------------------------------------------------------------------------
# As defesas: causas mecanicas, residualizacao, decadencia e tendencia secular
# ---------------------------------------------------------------------------

CONTROLS = ["zeros", "log_price", "log_dollar_volume", "log_volatility"]


def add_controls(conn, frame: pd.DataFrame) -> pd.DataFrame:
    """Junta ao painel as variaveis que poderiam explicar tudo mecanicamente.

    A mais perigosa e' a LIQUIDEZ. Uma accao com muitos dias parados tem um q75
    baixo, o que infla o racio de cauda sem que nada na cauda tenha mudado -- e a
    iliquidez e' persistente. Se fosse esse o motor, isto seria um estudo de
    persistencia de liquidez com outro nome."""
    frame = frame.copy()
    extra = []
    for ticker, group in frame.groupby("ticker"):
        rows = conn.execute(
            "SELECT ts, close, volume FROM intraday_prices "
            "WHERE ticker = ? AND timeframe = ? ORDER BY ts",
            (ticker, TIMEFRAME)).fetchall()
        if not rows:
            continue
        prices = pd.DataFrame([dict(r) for r in rows])
        prices["ts"] = pd.to_datetime(prices["ts"], utc=True, format="ISO8601")
        prices["year"] = pd.DatetimeIndex(prices["ts"]).year
        returns = np.log(prices.set_index("ts")["close"]).diff()

        for _, row in group.iterrows():
            window = prices[prices["year"] == row["period"]]
            slice_ = returns[pd.DatetimeIndex(returns.index).year == row["period"]]
            if window.empty:
                continue
            extra.append({
                "ticker": ticker, "period": row["period"],
                "zeros": float((slice_.abs() < 1e-12).mean()),
                "log_price": float(np.log(max(window["close"].median(), 0.01))),
                "log_dollar_volume": float(np.log(max(
                    (window["close"] * window["volume"]).median(), 1.0))),
            })

    if not extra:
        return frame
    merged = frame.merge(pd.DataFrame(extra), on=["ticker", "period"], how="left")
    merged["log_volatility"] = np.log(merged["volatility"].clip(lower=1e-6))
    return merged


def mechanical_controls(frame: pd.DataFrame) -> pd.DataFrame:
    """Correlacao transversal de cada medida com cada causa mecanica."""
    rows = []
    for measure in ("tail_ratio", "concentration", "hill"):
        entry = {"medida": measure}
        for control in CONTROLS:
            if control not in frame.columns:
                continue
            values = [stats.spearmanr(g[measure], g[control]).statistic
                      for _, g in frame.groupby("period")
                      if len(g) >= 20 and g[control].nunique() > 3]
            values = [v for v in values if np.isfinite(v)]
            entry[control] = float(np.mean(values)) if values else np.nan
        rows.append(entry)
    return pd.DataFrame(rows)


def residualise(frame: pd.DataFrame, measure: str) -> pd.Series:
    """Retira de cada medida, dentro de cada periodo, tudo o que os controlos
    explicam. O que sobra e' forma de cauda e mais nada."""
    available = [c for c in CONTROLS if c in frame.columns]
    if not available:
        return frame[measure]

    def _fit(group):
        design = np.column_stack([np.ones(len(group))]
                                 + [group[c].values for c in available])
        target = group[measure].values
        usable = np.isfinite(target) & np.isfinite(design).all(axis=1)
        out = pd.Series(np.nan, index=group.index)
        if usable.sum() < 20:
            return out
        beta, *_ = np.linalg.lstsq(design[usable], target[usable], rcond=None)
        out[usable] = target[usable] - design[usable] @ beta
        return out

    return frame.groupby("period", group_keys=False).apply(_fit)


def persistence_decay(frame: pd.DataFrame, measure: str,
                      lags=(1, 2, 3, 5, 10, 20, 30)) -> pd.DataFrame:
    """Persistencia a varias distancias no tempo.

    E' o que separa "propriedade da empresa" de "memoria curta". Uma coisa que
    ainda se ve' vinte anos depois e' caracter; uma que morre em dois anos e'
    inercia."""
    rows = []
    periods = set(frame["period"].unique())
    for lag in lags:
        values = []
        for earlier in sorted(periods):
            later = earlier + lag
            if later not in periods:
                continue
            left = frame[frame["period"] == earlier].set_index("ticker")[measure].dropna()
            right = frame[frame["period"] == later].set_index("ticker")[measure].dropna()
            common = left.index.intersection(right.index)
            if len(common) < 20:
                continue
            values.append(stats.spearmanr(left[common], right[common]).statistic)
        values = [v for v in values if np.isfinite(v)]
        if values:
            rows.append({"lag_anos": lag, "spearman": float(np.mean(values)),
                         "pares": len(values)})
    return pd.DataFrame(rows)


def secular_trend(frame: pd.DataFrame, measure: str = "concentration",
                  exclude=(2020,)) -> dict:
    """A concentracao do risco mudou ao longo das decadas?

    Com CONTROLO DE COMPOSICAO, que e' o erro obvio: ha' 19 accoes nos anos 60 e
    219 nos 2020, e a lista mais recente tem empresas de outro tipo. Usam-se
    apenas as accoes presentes em TODAS as decadas comparadas, para a diferenca
    ser sobre o tempo e nao sobre quem entrou.

    `exclude` tira anos excepcionais -- 2020 por omissao. Se a tendencia morrer
    sem a COVID, era a COVID."""
    frame = frame.copy()
    frame["decade"] = (frame["period"] // 10) * 10
    decades = [1980, 1990, 2000, 2010, 2020]
    present = [set(frame.loc[frame["decade"] == d, "ticker"]) for d in decades]
    constant = set.intersection(*present) if all(present) else set()
    if len(constant) < 30:
        return {}

    stable = frame[frame["ticker"].isin(constant)]
    by_decade = stable.groupby("decade").agg(
        measure=(measure, "mean"), volatility=("volatility", "mean")).to_dict("index")

    yearly = stable.groupby("period")[measure].mean()
    full = stats.linregress(yearly.index, yearly.values)

    trimmed = stable[~stable["period"].isin(exclude)]
    trimmed_yearly = trimmed.groupby("period")[measure].mean()
    without = stats.linregress(trimmed_yearly.index, trimmed_yearly.values)

    return {
        "n_constant": len(constant), "by_decade": by_decade,
        "slope": float(full.slope), "p_value": float(full.pvalue),
        "slope_excluding": float(without.slope),
        "p_value_excluding": float(without.pvalue),
        "r_squared_excluding": float(without.rvalue ** 2),
        "excluded": list(exclude),
    }


if __name__ == "__main__":
    connection = database.get_connection()

    yearly = panel(connection, by="year")
    if yearly.empty:
        print("sem historico suficiente -- corre src/research/deep_history.py")
        connection.close()
        raise SystemExit
    yearly = add_controls(connection, yearly)

    print("=" * 96)
    print("EXPERIENCIA R -- a forma da cauda e' propriedade persistente da accao?")
    print(f"  {len(yearly):,} pares (ticker, ano), {yearly['ticker'].nunique()} accoes, "
          f"{yearly['period'].min()} a {yearly['period'].max()}")
    print(f"  uma normal daria tail_ratio {GAUSSIAN_TAIL_RATIO:.2f}; "
          f"a mediana observada e' {yearly['tail_ratio'].median():.2f}")

    print()
    print("  1. E' SO' VOLATILIDADE, LIQUIDEZ OU DIMENSAO DISFARCADAS?")
    print("     (correlacao transversal media; alta = a medida e' artefacto disso)")
    controls = mechanical_controls(yearly)
    header = "".join(f"{c.replace('log_', ''):>18}" for c in CONTROLS if c in yearly.columns)
    print(f"     {'medida':>15}{header}")
    for _, row in controls.iterrows():
        cells = "".join(f"{row.get(c, np.nan):>+18.3f}"
                        for c in CONTROLS if c in yearly.columns)
        print(f"     {row['medida']:>15}{cells}")

    print()
    print("  2. PERSISTENCIA ANO A ANO, antes e depois de retirar os controlos:")
    print(f"     {'medida':>15} {'bruta':>9} {'residual':>10} {'retido':>8} "
          f"{'transicoes':>11} {'positivas':>10}")
    for measure in ("volatility", "tail_ratio", "concentration", "hill"):
        raw = persistence(yearly, measure)["spearman"]
        work = yearly.copy()
        work["residual"] = residualise(work, measure)
        residual = persistence(work, "residual")["spearman"]
        if raw.empty or residual.empty:
            continue
        print(f"     {measure:>15} {raw.mean():>+9.3f} {residual.mean():>+10.3f} "
              f"{residual.mean() / raw.mean() * 100:>7.0f}% {len(residual):>11} "
              f"{(residual > 0).mean():>10.0%}")

    print()
    print("  3. BATE A VOLATILIDADE a prever a forma futura?")
    for measure in ("tail_ratio", "concentration"):
        table = beats_volatility(yearly, measure)
        if table.empty:
            continue
        print(f"     {measure:>15}: pela forma {table['pela_forma'].mean():+.3f}, "
              f"pela volatilidade {table['pela_volatilidade'].mean():+.3f}, "
              f"vantagem {table['vantagem'].mean():+.3f} "
              f"({(table['vantagem'] > 0).mean():.0%} das transicoes)")

    print()
    print("  4. QUANTO TEMPO DURA?")
    print(f"     {'lag':>6} {'tail_ratio':>12} {'concentration':>15} {'pares':>7}")
    ratio_decay = persistence_decay(yearly, "tail_ratio").set_index("lag_anos")
    conc_decay = persistence_decay(yearly, "concentration").set_index("lag_anos")
    for lag in ratio_decay.index:
        print(f"     {lag:>5}a {ratio_decay.loc[lag, 'spearman']:>+12.3f} "
              f"{conc_decay.loc[lag, 'spearman']:>+15.3f} "
              f"{int(ratio_decay.loc[lag, 'pares']):>7}")

    decade = panel(connection, by="decade")
    if not decade.empty:
        print()
        print("  5. FORA DA AMOSTRA -- ordenar numa DECADA, medir na seguinte:")
        for measure in ("tail_ratio", "concentration"):
            table = decade_terciles(decade, measure)
            if table.empty:
                continue
            separations = table["separacao"]
            print(f"     {measure:>15}: separacao media {separations.mean():+.4f}, "
                  f"{(separations > 0).sum()}/{len(separations)} transicoes no "
                  f"sentido certo")

    print()
    print("  6. PLACEBO -- baralhar a identidade das accoes entre periodos:")
    for measure in ("tail_ratio", "concentration"):
        result = identity_placebo(yearly, measure, n=200)
        print(f"     {measure:>15}: observado {result['observed']:+.3f}, "
              f"p = {result['p_value']:.4f}")

    trend = secular_trend(yearly)
    if trend:
        print()
        print(f"  7. TENDENCIA SECULAR, nas MESMAS {trend['n_constant']} accoes "
              f"presentes em todas as decadas:")
        print(f"     {'decada':>8} {'concentracao':>14} {'volatilidade':>14}")
        for decade_key, values in sorted(trend["by_decade"].items()):
            print(f"     {int(decade_key):>7}s {values['measure']:>14.4f} "
                  f"{values['volatility'] * np.sqrt(252):>13.1%}")
        print(f"     tendencia {trend['slope']:+.5f}/ano (p = {trend['p_value']:.4f}); "
              f"sem {trend['excluded']} fica {trend['slope_excluding']:+.5f} "
              f"(p = {trend['p_value_excluding']:.4f})")

    connection.close()
