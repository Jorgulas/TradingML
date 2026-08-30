"""EXPERIENCIA L -- o angulo da tangente a' media movel (ideia do Jorge).

A PROPOSTA. Desenha-se a media movel sobre o grafico. No ultimo ponto -- o
momento actual -- mede-se o angulo entre a recta do troco anterior e a recta do
troco recente. Media plana que passa a subir da' um "joelho" de +30 graus; se
voltar a ficar plana, o joelho e' -30. Ha' um limiar (25 graus) e uma zona morta
no meio, e acima do limiar lanca-se uma moeda enviesada para o lado do joelho
em vez de decidir a seco.

DUAS CORRECCOES TECNICAS que a ideia obriga, e nenhuma delas a enfraquece:

1. O ANGULO PRECISA DE NORMALIZACAO. Um angulo exige os dois eixos nas mesmas
   unidades, e preco e tempo nao sao. Uma accao de 500 dolares e uma de 5 com a
   mesma subida percentual dao angulos completamente diferentes, e o "25 graus"
   passaria a depender da escala do grafico e do zoom. Aqui o declive e' medido
   em VOLATILIDADE POR BARRA: 45 graus quer dizer "a media sobe exactamente um
   movimento tipico por barra". Assim o limiar significa a mesma coisa na AAPL e
   na KO.

2. O ANGULO ABSOLUTO SOZINHO NAO ACRESCENTA NADA AO DECLIVE. `arctan` e'
   monotona, portanto "angulo > 25" e' identico a "declive normalizado > 0.466".
   Testar o angulo absoluto seria testar momentum com outro nome.

   Mas o ANGULO ENTRE DOIS TROCOS ja' nao e' transformacao monotona de nada:
   depende dos dois declives ao mesmo tempo e mede CURVATURA -- aceleracao da
   media, nao a sua direccao. Isso e' conteudo genuinamente novo neste projecto,
   e e' o que esta experiencia mede.

TRES REGRAS EM COMPETICAO, todas com a mesma zona morta e a mesma avaliacao:

    declive    age no sinal do angulo actual        (momentum, a referencia)
    joelho     age no sinal da VARIACAO do angulo   (a proposta)
    ambos      so' age quando os dois concordam

E a MOEDA: acima do limiar a posicao e' tomada com probabilidade crescente no
tamanho do joelho, em vez de ser tomada sempre. Mede-se contra a versao
determinista com a mesma esperanca -- a moeda nao pode mudar o retorno medio,
so' a variancia, e vale a pena ver quanto.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.decision.panel import is_corrupt
from src.decision.short_term import block_bootstrap_mean
from src.patterns.ingest_intraday import load_bars

import numpy as np
import pandas as pd
from scipy import stats

# Grelha de configuracoes. Pequena de proposito: cada combinacao e' mais uma
# comparacao, e varrer centenas garante encontrar a mais sortuda.
MA_WINDOWS = (10, 20, 50)
SPANS = (5, 10, 20)            # barras de cada troco de recta
THRESHOLDS = (5.0, 10.0, 15.0, 20.0, 25.0, 35.0)
# Exposicao minima para uma configuracao poder ser escolhida. Sem este chao, o
# criterio "retorno por exposicao" premeia sempre a regra que quase nunca age:
# duas dezenas de posicoes sortudas dao um racio enorme e nao significam nada.
MIN_EXPOSURE = 0.10
SATURATION_DEGREES = 20.0      # quantos graus acima do limiar levam a moeda a 100%


def normalised_slope(series: pd.Series, span: int, scale: pd.Series) -> pd.Series:
    """Declive da media em unidades de VOLATILIDADE POR BARRA.

    Divide-se pela volatilidade de uma barra para o numero ficar sem unidades:
    1.0 quer dizer que a media avanca um movimento tipico por barra. E' isto que
    torna o angulo comparavel entre accoes, que e' a correccao (1) do cabecalho."""
    rise = series.diff(span) / span
    return rise / scale.replace(0, np.nan)


def build(bars: pd.DataFrame, ma_window: int, span: int,
          session_aware: bool = False) -> pd.DataFrame:
    """Angulo actual, angulo anterior, e o joelho entre eles.

    EM LOG-PRECO, e nao em preco. A primeira versao media a media sobre o preco
    e dividia por `volatilidade x preco`. Um teste apanhou o problema: numa
    subida perfeitamente constante o joelho devia ser zero e dava 16.7 graus. O
    denominador andava -- o preco sobe durante a tendencia -- e o angulo caia
    sozinho, criando um joelho que nao existia no grafico.

    Em log-preco o declive ja' e' um retorno por barra e o denominador e' a
    volatilidade desse mesmo retorno: a razao e' verdadeiramente adimensional e
    nao deriva com o nivel do preco. Numa tendencia constante da' zero, como
    tem de dar."""
    close = bars["close"]
    log_price = np.log(close)

    if session_aware:
        # INTRADIA: nada pode atravessar a noite.
        #
        # Sem isto, uma media de 20 barras na quinta barra do dia usa 15 barras
        # de ontem, e o `diff` apanha o salto overnight -- que pode ser 1% e nao
        # e' um movimento de um minuto. O angulo resultante dispara para o tecto
        # todas as manhas, sempre no mesmo sentido do gap, e a regra passaria a
        # negociar o gap disfarcado de joelho.
        session = pd.Series(bars.index.normalize(), index=bars.index)
        grouped = log_price.groupby(session)
        average = grouped.transform(
            lambda s: s.rolling(ma_window, min_periods=ma_window).mean())
        returns = grouped.transform(lambda s: s.diff())
        scale = returns.groupby(session).transform(
            lambda s: s.rolling(60, min_periods=20).std())
        rise = average.groupby(session).transform(lambda s: s.diff(span)) / span
        slope_now = rise / scale.replace(0, np.nan)
        angle_now = np.degrees(np.arctan(slope_now))
        angle_previous = angle_now.groupby(session).shift(span)
    else:
        average = log_price.rolling(ma_window, min_periods=ma_window).mean()
        # Escala: desvio-padrao do retorno logaritmico de uma barra. Janela
        # longa para a propria escala nao oscilar e fabricar sinal.
        scale = log_price.diff().rolling(60, min_periods=60).std()
        slope_now = normalised_slope(average, span, scale)
        angle_now = np.degrees(np.arctan(slope_now))
        # O troco ANTERIOR e' o mesmo declive atrasado `span` barras: a recta
        # que ia ate' onde a actual comeca, sem sobreposicao entre as duas.
        angle_previous = angle_now.shift(span)

    return pd.DataFrame({
        "close": close,
        "angle": angle_now,
        "angle_previous": angle_previous,
        "kink": angle_now - angle_previous,
    }, index=bars.index)


def coin_probability(magnitude: pd.Series, threshold: float) -> pd.Series:
    """Probabilidade de agir: zero na zona morta, a subir ate' 1.

    E' a "moeda enviesada" da proposta, escrita como probabilidade em vez de
    como sorteio. Um joelho de exactamente 25 graus quase nao move a mao; um de
    45 move-a por inteiro."""
    return ((magnitude.abs() - threshold) / SATURATION_DEGREES).clip(0.0, 1.0)


def positions(frame: pd.DataFrame, rule: str, threshold: float) -> pd.Series:
    """Posicao desejada em [-1, 1]. Zero e' nao fazer nada, e e' a decisao por
    omissao dentro da zona morta."""
    if rule == "declive":
        driver = frame["angle"]
    elif rule == "joelho":
        driver = frame["kink"]
    elif rule == "ambos":
        # So' age quando o angulo e o joelho apontam ao mesmo lado: a media ja'
        # sobe E esta' a acelerar. O tamanho fica pelo mais fraco dos dois.
        agree = np.sign(frame["angle"]) == np.sign(frame["kink"])
        weaker = np.minimum(frame["angle"].abs(), frame["kink"].abs())
        driver = np.sign(frame["kink"]) * weaker.where(agree, 0.0)
    else:
        raise ValueError(f"regra desconhecida: {rule}")

    return np.sign(driver) * coin_probability(driver, threshold)


def panel(conn, timeframe: str, ma_window: int, span: int,
          horizons, tickers=None, session_aware: bool = False,
          min_bars: int = 300, lag: int = 0) -> pd.DataFrame:
    """Um painel com o sinal e o retorno futuro, para todos os tickers.

    O retorno e' em EXCESSO sobre o indice. Sem isso, uma regra que esteja
    comprada a maior parte do tempo herda a deriva do mercado e parece boa por
    causa dela -- foi o que a ronda 7 mostrou custar 53.7% das janelas."""
    tickers = tickers or config.TREE_TICKERS
    benchmark = load_bars(conn, config.BENCHMARK["ticker"], timeframe)
    market = benchmark["close"] if not benchmark.empty else None

    frames = []
    for ticker in tickers:
        bars = load_bars(conn, ticker, timeframe)
        if len(bars) < min_bars or is_corrupt(bars):
            continue

        piece = build(bars, ma_window, span, session_aware=session_aware)
        aligned = market.reindex(bars.index).ffill() if market is not None else None
        session = pd.Series(bars.index.normalize(), index=bars.index)
        for horizon in horizons:
            # `lag` atrasa a ENTRADA: o sinal ve' o fecho da barra t e a posicao
            # so' abre em t+lag. Com lag=1 e' o que acontece na realidade -- nao
            # se transacciona no preco que se acabou de observar -- e e' tambem
            # o teste ao salto bid-ask, que a 1 minuto e' o contaminante
            # classico: o preco alterna entre compra e venda e fabrica
            # previsibilidade que desaparece assim que se salta uma barra.
            entry = bars["close"].shift(-lag)
            exit_price = bars["close"].shift(-(lag + horizon))
            forward = exit_price / entry - 1
            if aligned is not None:
                forward = forward - (aligned.shift(-(lag + horizon))
                                     / aligned.shift(-lag) - 1)
            if session_aware:
                # O retorno futuro tambem nao pode saltar para o dia seguinte:
                # seria creditar ao sinal um gap overnight que ele nao previu.
                forward = forward.where(session.shift(-(lag + horizon)) == session)
            piece[f"forward_{horizon}"] = forward

        piece["ticker"] = ticker
        piece["ts"] = piece.index
        piece.index = range(len(piece))
        frames.append(piece.dropna(subset=["kink"] + [f"forward_{h}" for h in horizons]))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values(["ts", "ticker"]).reset_index(drop=True)


def bar_returns(frame: pd.DataFrame, weight: pd.Series,
                target: str = "forward_return") -> pd.Series:
    """Retorno da carteira em cada instante: media ponderada pelas posicoes.

    Divide-se pelo numero de accoes DISPONIVEIS e nao pelo numero de posicoes
    abertas. E' a diferenca entre "que rendeu o capital" e "que rendeu quem
    apostou": normalizar pelas posicoes abertas faria um dia com uma unica
    aposta valer tanto como um dia com cem."""
    # Vectorizado de proposito. A versao com `groupby.apply` e um lambda
    # demorava minutos por avaliacao sobre 800 mil linhas, e o varrimento faz
    # perto de mil avaliacoes -- passava de exequivel a impraticavel.
    contribution = weight.values * frame[target].values
    grouped = pd.Series(contribution, index=frame["ts"].values).groupby(level=0)
    return (grouped.sum() / grouped.size()).sort_index()


def daily(series: pd.Series) -> pd.Series:
    """Agrega para o DIA. A amostra efectiva sao dias, nao barras: 160 accoes a
    mover-se juntas numa hora nao sao 160 observacoes, e sete horas do mesmo dia
    nao sao sete."""
    return series.groupby(pd.DatetimeIndex(series.index).normalize()).sum()


def evaluate(frame: pd.DataFrame, rule: str, threshold: float,
             horizon: int, seed: int = 5, invert: bool = False,
             full: bool = True) -> dict:
    """`full=False` salta o bootstrap e o sorteio da moeda.

    Nao e' atalho de conveniencia: o varrimento faz perto de mil avaliacoes e o
    bootstrap de blocos sao 4000 reamostragens em cada uma. No varrimento so'
    interessa ordenar configuracoes pelo retorno; o intervalo de confianca so'
    tem significado para a configuracao ESCOLHIDA, medida no teste -- calcula-lo
    mil vezes durante a escolha seria caro e nao entraria em decisao nenhuma."""
    weight = positions(frame, rule, threshold)
    if invert:
        weight = -weight
    target = f"forward_{horizon}"
    series = daily(bar_returns(frame, weight, target))
    values = series.values
    exposure = float(weight.abs().mean())

    if not full:
        return {
            "rule": rule, "threshold": threshold, "invert": invert,
            "mean": float(values.mean()) if len(values) else np.nan,
            "exposure": exposure, "n_days": int(len(values)),
            "hit_days": float((values > 0).mean()) if len(values) else np.nan,
        }

    low, high = block_bootstrap_mean(values, block=horizon)

    # A moeda a serio: sorteio de Bernoulli com a mesma probabilidade. Mesma
    # esperanca, outra variancia -- e' exactamente isso que se quer medir.
    rng = np.random.default_rng(seed)
    tossed = np.sign(weight) * (rng.random(len(weight)) < weight.abs())
    tossed_series = daily(bar_returns(frame, pd.Series(tossed, index=weight.index), target))

    return {
        "rule": rule, "threshold": threshold, "invert": invert,
        "mean": float(values.mean()) if len(values) else np.nan,
        "ci": (low, high), "significant": low > 0,
        "n_days": int(len(values)),
        "hit_days": float((values > 0).mean()) if len(values) else np.nan,
        "exposure": exposure,
        # Retorno por unidade de exposicao: sem isto uma regra que quase nunca
        # age parece pior do que e', so' por estar fora do mercado.
        "per_exposure": float(values.mean() / exposure) if exposure > 0 else np.nan,
        "coin_mean": float(tossed_series.values.mean()) if len(tossed_series) else np.nan,
        "coin_std": float(tossed_series.values.std(ddof=1)) if len(tossed_series) > 1 else np.nan,
        "deterministic_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
    }


def split_frame(frame, horizon):
    """Corte cronologico global com embargo. O sinal nao tem parametros
    ajustados aos dados, mas a CONFIGURACAO e o LIMIAR sao escolhidos, e isso
    conta como ajuste."""
    stamps = np.sort(frame["ts"].unique())
    first = stamps[int(len(stamps) * 0.8)]
    step = pd.Series(stamps).diff().median()
    embargo = step * horizon if pd.notna(step) else pd.Timedelta(0)
    return frame[frame["ts"] < first - embargo], frame[frame["ts"] >= first]


RULES = ("declive", "joelho", "ambos")


def run(conn, timeframe: str = "1h", horizons=(1, 3, 7), tickers=None) -> dict | None:
    """Varrimento na porcao de treino, uma escolha por regra, medicao no teste.

    A escolha e' pelo RETORNO no treino, com chao de exposicao -- e nao pelo
    retorno por exposicao, que premiava sempre a configuracao que quase nunca
    agia e cujo racio enorme vinha de duas dezenas de posicoes sortudas.

    Testa-se tambem a regra INVERTIDA. Nao e' pesca: e' uma hipotese com
    conteudo, porque a aceleracao de uma media pode muito bem anteceder
    reversao em vez de continuacao, e a versao invertida responde a essa
    pergunta directamente. Conta como comparacao a mais e esta' contabilizada."""
    results, cache = [], {}
    for ma_window in MA_WINDOWS:
        for span in SPANS:
            frame = panel(conn, timeframe, ma_window, span, horizons, tickers)
            if frame.empty:
                continue
            cache[(ma_window, span)] = frame
            train, _ = split_frame(frame, max(horizons))
            for horizon in horizons:
                for rule in RULES:
                    for threshold in THRESHOLDS:
                        for invert in (False, True):
                            outcome = evaluate(train, rule, threshold, horizon,
                                               invert=invert, full=False)
                            outcome.update({"ma_window": ma_window, "span": span,
                                            "horizon": horizon})
                            results.append(outcome)

    if not results:
        return None

    sweep = pd.DataFrame(results)
    usable = sweep[sweep["exposure"] >= MIN_EXPOSURE]

    chosen = {}
    for rule in RULES:
        subset = usable[usable["rule"] == rule]
        if subset.empty:
            continue
        best = subset.loc[subset["mean"].idxmax()]
        frame = cache[(int(best["ma_window"]), int(best["span"]))]
        _, test = split_frame(frame, max(horizons))
        horizon = int(best["horizon"])
        on_test = evaluate(test, rule, float(best["threshold"]), horizon,
                           invert=bool(best["invert"]))
        on_test.update({"ma_window": int(best["ma_window"]), "span": int(best["span"]),
                        "horizon": horizon, "train_mean": float(best["mean"])})
        chosen[rule] = on_test

    # Referencia: comprado em tudo, sempre, no mesmo horizonte que a melhor
    # regra escolheu. Um sinal que nao bata isto nao serve, por mais elegante
    # que seja a construcao.
    reference_horizon = (max(chosen.values(), key=lambda r: r["mean"])["horizon"]
                         if chosen else horizons[0])
    any_frame = next(iter(cache.values()))
    _, test_any = split_frame(any_frame, max(horizons))
    always_series = daily(bar_returns(test_any, pd.Series(1.0, index=test_any.index),
                                      f"forward_{reference_horizon}"))
    low, high = block_bootstrap_mean(always_series.values, block=reference_horizon)
    always = {
        "rule": "sempre comprado (referencia)", "threshold": np.nan, "invert": False,
        "mean": float(always_series.values.mean()), "ci": (low, high),
        "significant": low > 0, "exposure": 1.0, "horizon": reference_horizon,
        "per_exposure": float(always_series.values.mean()),
        "hit_days": float((always_series.values > 0).mean()),
        "n_days": int(len(always_series)),
        "coin_mean": np.nan, "coin_std": np.nan, "deterministic_std": np.nan,
    }

    # Diagnostico numa configuracao FIXA e central (media 20, troco 20), nao na
    # que o varrimento escolheu: se usasse a escolhida, herdava a seleccao e
    # deixava de ser uma medicao independente do sinal.
    reference = cache.get((20, 20), any_frame)
    _, reference_test = split_frame(reference, max(horizons))

    return {"timeframe": timeframe, "horizons": list(horizons), "sweep": sweep,
            "chosen": chosen, "always": always, "n_configs": len(sweep),
            "diagnosis": diagnose(reference_test, horizons),
            "n_tickers": int(any_frame["ticker"].nunique())}


def deciles(frame: pd.DataFrame, column: str, horizon: int,
            n: int = 10) -> pd.DataFrame:
    """Retorno futuro medio por decil do sinal, com IC ao nivel do DIA.

    E' a medicao mais justa que se pode fazer a uma ideia de sinal, e separa
    duas perguntas que a conta de lucros mistura: o sinal CONTEM informacao, e a
    regra sabe EXTRAI-LA. Uma zona morta mal posta, um limiar mal escolhido ou
    um dimensionamento infeliz destroem uma regra sem que o sinal tenha
    culpa. Se os decis forem monotonos, ha' informacao e o problema e' a regra;
    se forem planos, nao ha' nada para extrair e nenhuma regra o salvaria."""
    work = frame[["ts", column, f"forward_{horizon}"]].dropna()
    work = work.assign(bucket=pd.qcut(work[column], n, labels=False, duplicates="drop"))

    rows = []
    for bucket, piece in work.groupby("bucket"):
        # Uma observacao por dia, nao por linha: 158 accoes na mesma hora
        # movem-se juntas e nao sao 158 medicoes independentes.
        by_day = piece.groupby(pd.DatetimeIndex(piece["ts"]).normalize())[
            f"forward_{horizon}"].mean()
        low, high = block_bootstrap_mean(by_day.values, block=horizon)
        rows.append({"decil": int(bucket), "n": len(piece),
                     "sinal_medio": float(piece[column].mean()),
                     "retorno": float(by_day.mean()),
                     "ci": (low, high), "significant": low > 0 or high < 0})
    return pd.DataFrame(rows)


def diagnose(frame: pd.DataFrame, horizons) -> dict:
    """O sinal CONTEM informacao, independentemente da regra que a tenta extrair?

    A conta de lucros de uma regra mistura duas perguntas. Um limiar mal posto,
    uma zona morta mal dimensionada ou um horizonte infeliz afundam uma regra sem
    que o sinal tenha culpa nenhuma. Estas duas medicoes separam-nas:

      DIRECCAO   Spearman entre o sinal e o retorno FUTURO COM SINAL. Se houver
                 estrutura monotona nos decis, ha' direccao para extrair.
      MAGNITUDE  Spearman entre |sinal| e |retorno|. Um sinal pode nao dizer
                 para que lado e ainda assim dizer QUANTO -- e' o padrao que
                 este projecto encontrou repetidamente."""
    out = {}
    for horizon in horizons:
        target = f"forward_{horizon}"
        work = frame[["ts", "angle", "kink", target]].dropna()
        if work.empty:
            continue
        signed = work[target]
        magnitude = signed.abs()
        row = {}
        for name in ("angle", "kink"):
            values = work[name]
            table = deciles(work, name, horizon)
            row[name] = {
                "spearman_direction": float(stats.spearmanr(values, signed).statistic),
                "spearman_magnitude": float(stats.spearmanr(values.abs(),
                                                            magnitude).statistic),
                "deciles": table["retorno"].tolist(),
                "top_minus_bottom": float(table["retorno"].iloc[-1]
                                          - table["retorno"].iloc[0]),
            }
        out[horizon] = row
    return out


def consistency(train: pd.DataFrame, test: pd.DataFrame, column: str,
                horizon: int, absolute: bool = False) -> dict:
    """A mesma medicao nas DUAS particoes, lado a lado.

    Foi esta verificacao que apanhou o unico falso positivo desta experiencia.
    A forca da tendencia (|angulo|) dava, no teste, um topo-menos-fundo de
    +1.53% a 5 dias com intervalo inteiramente acima de zero -- e no TREINO dava
    +0.04% com o zero la' dentro. O treino tem quatro vezes mais dados, logo
    quatro vezes mais poder: se o efeito fosse real, aparecia primeiro la'. Um
    efeito que so' existe nos ultimos 20% da amostra e' uma propriedade daquele
    periodo, nao um sinal.

    A regra pratica que daqui sai: antes de acreditar num resultado de teste,
    perguntar o que diz o treino. Nao para o escolher -- para o descartar."""
    out = {}
    for name, part in (("treino", train), ("teste", test)):
        work = part[["ts", column, f"forward_{horizon}"]].dropna().copy()
        driver = work[column].abs() if absolute else work[column]
        work = work.assign(bucket=pd.qcut(driver, 10, labels=False, duplicates="drop"))

        edges = {}
        for bucket in (0, work["bucket"].max()):
            piece = work[work["bucket"] == bucket]
            edges[bucket] = piece.groupby(
                pd.DatetimeIndex(piece["ts"]).normalize())[f"forward_{horizon}"].mean()
        joined = pd.concat([edges[work["bucket"].max()], edges[0]], axis=1,
                           sort=False).dropna()
        difference = (joined.iloc[:, 0] - joined.iloc[:, 1]).values
        low, high = block_bootstrap_mean(difference, block=horizon)
        out[name] = {
            "spearman": float(stats.spearmanr(driver, work[f"forward_{horizon}"]).statistic),
            "top_minus_bottom": float(difference.mean()),
            "ci": (low, high), "significant": low > 0 or high < 0,
            "n_days": int(len(difference)),
        }
    out["agree"] = (out["treino"]["significant"] and out["teste"]["significant"]
                    and np.sign(out["treino"]["top_minus_bottom"])
                    == np.sign(out["teste"]["top_minus_bottom"]))
    return out


def _row(outcome: dict, label: str = None) -> str:
    low, high = outcome["ci"]
    mark = "*" if outcome["significant"] else " "
    name = label or outcome["rule"]
    return (f"    {name:>44} {outcome['mean']:>+10.4%}{mark} "
            f"[{low:>+9.4%}, {high:>+9.4%}] {outcome['exposure']:>9.1%} "
            f"{outcome['hit_days']:>8.1%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--horizons", default="1,3,7")
    args = parser.parse_args()
    horizons = tuple(int(h) for h in args.horizons.split(","))

    connection = database.get_connection()
    outcome = run(connection, args.timeframe, horizons)
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print("=" * 104)
    print(f"EXPERIENCIA L -- angulo da tangente a' media movel @ {outcome['timeframe']}")
    print(f"  {outcome['n_tickers']} accoes | angulo em unidades de volatilidade por barra")
    print(f"  {outcome['n_configs']} configuracoes varridas no treino, "
          f"horizontes {outcome['horizons']} barras")

    print()
    print("  MELHOR CONFIGURACAO DE CADA REGRA (escolhida no treino, medida no teste):")
    print(f"    {'regra':>44} {'retorno':>10} {'IC95% (blocos)':>26} "
          f"{'exposic.':>9} {'dias +':>8}")
    print(_row(outcome["always"], "sempre comprado (referencia)"))
    for rule, result in outcome["chosen"].items():
        label = (f"{'-' if result['invert'] else ''}{rule} "
                 f"(media {result['ma_window']}, troco {result['span']}, "
                 f"{result['threshold']:.0f} graus, h={result['horizon']})")
        print(_row(result, label))

    print()
    print("  A MOEDA custa o que? (mesma esperanca, outra variancia)")
    print(f"    {'regra':>12} {'determinista':>13} {'com moeda':>11} "
          f"{'desvio det.':>12} {'desvio moeda':>13}")
    for rule, result in outcome["chosen"].items():
        print(f"    {rule:>12} {result['mean']:>+13.4%} {result['coin_mean']:>+11.4%} "
              f"{result['deterministic_std']:>12.4%} {result['coin_std']:>13.4%}")

    print()
    print("  QUEM DOMINA O VARRIMENTO NO TREINO (so' configuracoes accionaveis):")
    sweep = outcome["sweep"]
    usable = sweep[sweep["exposure"] >= MIN_EXPOSURE]
    top = usable.nlargest(12, "mean")
    print(f"    {'regra':>10} {'inv':>4} {'media':>6} {'troco':>6} {'graus':>6} "
          f"{'h':>3} {'exposic.':>9} {'retorno':>10}")
    for _, r in top.iterrows():
        print(f"    {r['rule']:>10} {'sim' if r['invert'] else 'nao':>4} "
              f"{r['ma_window']:>6} {r['span']:>6} {r['threshold']:>6.0f} "
              f"{r['horizon']:>3} {r['exposure']:>9.1%} {r['mean']:>+10.4%}")

    print()
    counts = usable.nlargest(30, "mean")["rule"].value_counts().to_dict()
    print(f"    nas 30 melhores do treino: " +
          "  ".join(f"{k}={v}" for k, v in counts.items()))

    print()
    print("  O SINAL CONTEM INFORMACAO? (independente da regra que a extrai)")
    print(f"    {'horizonte':>10} {'sinal':>8} {'direccao':>10} {'magnitude':>10} "
          f"{'topo-fundo':>12}")
    for horizon, row in outcome["diagnosis"].items():
        for name, stats_row in row.items():
            print(f"    {horizon:>10} {name:>8} "
                  f"{stats_row['spearman_direction']:>+10.4f} "
                  f"{stats_row['spearman_magnitude']:>+10.4f} "
                  f"{stats_row['top_minus_bottom']:>+12.4%}")
    print()
    print("    Retorno por decil do sinal (do mais negativo ao mais positivo):")
    for horizon, row in outcome["diagnosis"].items():
        for name, stats_row in row.items():
            cells = "  ".join(f"{v:+.3%}" for v in stats_row["deciles"])
            print(f"      h={horizon} {name:>6}: {cells}")

    print()
    print("  * = intervalo de confianca inteiramente acima de zero")
    connection.close()
