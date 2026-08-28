"""EXPERIENCIA K -- o vies de sobrevivencia explica o resultado da ronda 7?

A suspeita, e porque e' seria. As 158 accoes do painel sao as grandes
capitalizacoes de HOJE. Nenhuma empresa que tenha colapsado, sido comprada ou
caido do indice entre 2023 e 2026 esta' la'. E os numeros confirmam-no de forma
crua: das 158, TODAS tem historico completo e TODAS chegam vivas ao ultimo dia.
Um indice real perde 5 a 8% dos membros por ano. Zero atrito em tres anos nao e'
o mundo -- e' a lista a ser escolhida no fim.

Num long-short transversal isto nao e' obviamente mau nem obviamente bom, e e'
por isso que precisa de teste em vez de ressalva. As accoes em falta sao
sobretudo desastres (falencia, queda longa) e aquisicoes (choque positivo). Se
o modelo as tivesse posto do lado VENDIDO, a ausencia delas esta' a PREJUDICAR
o resultado medido. Se as tivesse comprado, esta' a inflaciona-lo.

QUATRO TESTES, do mais barato ao decisivo:

  1. DE QUE PERNA VEM O RESULTADO. Se vier toda da perna comprada, a ausencia de
     falidos -- que so' afecta a vendida -- nao o pode explicar. E' o teste mais
     barato e ja' elimina metade das hipoteses.

  2. O MODELO EVITA OS DESASTRES QUE EXISTEM? Entre as accoes que ficaram,
     houve quedas violentas. O que e' que o modelo tinha dito na vespera? Se
     acerta a por as futuras quedas do lado vendido, acrescentar mais quedas
     (as que faltam) MELHORARIA o resultado.

  3. O EFEITO CONCENTRA-SE NAS ACCOES SEGURAS? Estratificar por drawdown
     maximo: as accoes que mais sofreram sao as analogas vivas das que
     morreram. Se o efeito so' existir entre as mega-caps tranquilas e
     desaparecer nas fragilizadas, e' mau sinal.

  4. REPOR AS QUE FALTAM. Ha' empresas que sairam deste universo por decadencia
     e continuam a negociar, portanto o yfinance ainda as serve. Repo-las no
     painel e repete-se a medicao. E' o unico teste directo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.decision import panel as P
from src.decision import short_term as ST
from src.research.fallen_universe import (
    DELISTED_NO_DATA, surviving_fallen)

import numpy as np
import pandas as pd
from scipy import stats


# As caidas nao sao todas do mesmo tipo, e a distincao decide a interpretacao.
#
# Umas ERAM membros do indice de grande capitalizacao e sairam por decadencia:
# sao exactamente o que falta ao painel, e repo-las e' reparar o vies de
# sobrevivencia.
#
# Outras NUNCA foram grandes capitalizacoes -- sao nomes especulativos de alta
# volatilidade e forte presenca retalhista. Repo-las nao repara vies nenhum:
# ALARGA o universo para outro tipo de empresa. Se o efeito morrer so' com
# estas, o problema nao e' sobrevivencia, e' o modelo nao generalizar para fora
# das grandes capitalizacoes -- que e' um problema diferente e menos grave.
#
# A classificacao vem do meu conhecimento da composicao do S&P 500 e pode ter
# erros nas margens. O que nao tem duvida e' que os dois grupos sao de natureza
# diferente: o primeiro tem preco mediano de dezenas de dolares e negocio
# estabelecido, o segundo inclui RIVN, LCID, AMC e PLUG.
INDEX_REMOVED = [
    "VFC", "AAP", "NWL", "LNC", "DXC", "ZION", "ALK", "SEDG", "ENPH", "PARA",
    "LUMN", "FMC", "CE", "MKTX", "ETSY", "BIO", "XRAY", "QRVO", "MHK", "APA",
    "IVZ", "HII", "CZR", "BWA", "WHR", "HSIC", "CRL", "TFX", "PNW", "NCLH",
    "AAL", "HAS", "GNRC", "MTCH", "ILMN", "MRNA", "CHRW", "OGN", "VTRS", "TAP",
    "CPB", "SJM", "DG", "DLTR", "ULTA",
]

NEVER_LARGE_CAP = [
    "SPWR", "PLUG", "CVNA", "AMC", "LCID", "CHPT", "RIVN", "BYND", "PTON",
    "WOLF", "AZTA", "TDOC", "ZM", "DOCU", "ROKU",
]


def legs(frame: pd.DataFrame, confidence: np.ndarray, fraction: float = 0.1) -> dict:
    """Separa o retorno long-short nas suas duas pernas.

    A perna comprada e' a media do decil de topo, a vendida e' MENOS a media do
    decil de baixo. Somam para o resultado total. Como o alvo ja' e' retorno em
    excesso sobre o indice, cada perna e' interpretavel sozinha: uma perna
    vendida positiva quer dizer que o decil de baixo rendeu menos do que o
    mercado, que e' exactamente o que se lhe pede."""
    work = frame[["ts", "forward_return"]].copy()
    work["confidence"] = confidence
    long_leg, short_leg = {}, {}
    for stamp, group in work.groupby("ts"):
        if len(group) < 20:
            continue
        size = max(1, int(len(group) * fraction))
        ordered = group.sort_values("confidence")
        long_leg[stamp] = float(ordered["forward_return"].iloc[-size:].mean())
        short_leg[stamp] = -float(ordered["forward_return"].iloc[:size].mean())
    return {"comprada": pd.Series(long_leg).sort_index(),
            "vendida": pd.Series(short_leg).sort_index()}


def crash_response(frame: pd.DataFrame, confidence: np.ndarray,
                   quantile: float = 0.02) -> dict:
    """O modelo poe as quedas violentas do lado vendido?

    Toma-se o percentil `quantile` mais baixo dos retornos realizados -- as
    quedas a serio -- e pergunta-se em que percentil de confianca o modelo as
    tinha colocado na vespera. Se a resposta for abaixo de 0.5, o modelo
    inclina-se a vende-las, e as quedas que faltam no painel teriam ajudado."""
    work = frame[["ts", "forward_return"]].copy()
    work["confidence"] = confidence
    # Percentil da confianca dentro do dia: e' assim que a decisao e' tomada.
    work["rank"] = work.groupby("ts")["confidence"].rank(pct=True)

    threshold = work["forward_return"].quantile(quantile)
    crashes = work[work["forward_return"] <= threshold]
    spikes = work[work["forward_return"] >= work["forward_return"].quantile(1 - quantile)]

    return {
        "crash_threshold": float(threshold),
        "n_crashes": int(len(crashes)),
        "crash_mean_rank": float(crashes["rank"].mean()),
        "spike_mean_rank": float(spikes["rank"].mean()),
        "normal_mean_rank": float(work["rank"].mean()),
        # Teste formal: a distribuicao dos postos das quedas e' diferente da
        # uniforme? Wilcoxon contra 0.5 e' o mais directo.
        "crash_p_value": float(stats.wilcoxon(crashes["rank"] - 0.5).pvalue)
        if len(crashes) > 20 else None,
    }


def fragility_strata(frame: pd.DataFrame, confidence: np.ndarray,
                     n_groups: int = 3) -> pd.DataFrame:
    """IC dentro de cada grupo de fragilidade.

    Fragilidade medida pelo DRAWDOWN MAXIMO de cada accao no periodo: as que
    mais caíram sao as analogas vivas das que morreram. A pergunta e' se o
    efeito sobrevive la' dentro ou se so' existe entre as tranquilas.

    Nota sobre o drawdown: e' calculado sobre a amostra inteira, portanto usa
    informacao futura. Nao ha' problema porque nao entra em nenhuma decisao --
    e' apenas a variavel que define os grupos da analise, como um estrato
    demografico. Usa-la para negociar seria fuga; usa-la para estratificar e'
    o proprio objectivo."""
    work = frame[["ticker", "ts", "forward_return"]].copy()
    work["confidence"] = confidence

    # Drawdown a partir do retorno cumulativo de cada accao no painel.
    depth = {}
    for ticker, group in frame.sort_values("ts").groupby("ticker"):
        curve = (1 + group["ret_1"].fillna(0)).cumprod()
        depth[ticker] = float((curve / curve.cummax() - 1).min())
    work["drawdown"] = work["ticker"].map(depth)

    edges = work.groupby("ticker")["drawdown"].first().quantile(
        np.linspace(0, 1, n_groups + 1)).values
    labels = ["mais fragil", "intermedio", "mais solido"][:n_groups]
    # Bordas de menor para maior drawdown: o mais negativo e' o mais fragil.
    work["grupo"] = pd.cut(work["drawdown"], bins=np.unique(edges),
                           labels=labels, include_lowest=True)

    rows = []
    for group_name, piece in work.groupby("grupo", observed=True):
        if piece.empty:
            continue
        daily_ic = piece.groupby("ts").apply(
            lambda g: stats.spearmanr(g["confidence"], g["forward_return"]).statistic
            if len(g) >= 10 and g["confidence"].nunique() > 2 else np.nan,
            include_groups=False)
        daily_ic = daily_ic.dropna()
        low, high = ST.block_bootstrap_mean(daily_ic.values)
        rows.append({
            "grupo": str(group_name),
            "n_tickers": int(piece["ticker"].nunique()),
            "drawdown_medio": float(piece.groupby("ticker")["drawdown"].first().mean()),
            "ic": float(daily_ic.mean()),
            "ci": (low, high),
            "significant": low > 0,
            "n_days": int(len(daily_ic)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TESTE 4 -- repor as accoes que faltam
# ---------------------------------------------------------------------------

def quarterly_walkforward(frame: pd.DataFrame, min_train_quarters: int = 6) -> dict:
    """Walk-forward trimestral so' com o modelo transversal de curto prazo.

    So' este modelo, e nao a tabela inteira, porque e' nele que a ronda 7
    localizou toda a evidencia. Acrescentar a tabela por cima traria os sinais
    de padrao e de resultados, que dependem de dados que as accoes caidas nao
    tem, e a comparacao entre universos deixaria de ser sobre o universo."""
    frame = frame.copy()
    frame["quarter"] = (pd.DatetimeIndex(pd.to_datetime(frame["ts"]))
                        .tz_localize(None).to_period("Q").astype(str))
    labels = sorted(frame["quarter"].unique())
    columns = ST.feature_sets(frame)["transversal (percentis do dia)"]

    daily_pieces, ic_pieces, rows = [], [], []
    for index in range(min_train_quarters, len(labels)):
        label = labels[index]
        evaluation = P.labelled(frame[frame["quarter"] == label])
        if evaluation.empty:
            continue
        cutoff = evaluation["ts"].min() - pd.Timedelta(days=config.DECISION["horizon_days"] + 2)
        history = P.labelled(frame[frame["ts"] <= cutoff])
        if len(history) < 5000:
            continue

        model = ST.fit(history, columns)
        confidence = ST.score(model, evaluation, columns)
        daily = ST.long_short_return(evaluation, confidence)
        ic = ST.information_coefficient(evaluation, confidence)
        daily_pieces.append(daily)
        ic_pieces.append(ic)
        rows.append({"quarter": label, "n_tickers": int(evaluation["ticker"].nunique()),
                     "mean": float(daily.mean()), "ic": float(ic.mean())})

    if not daily_pieces:
        return {}
    values = pd.concat(daily_pieces).sort_index()
    ic_values = pd.concat(ic_pieces).sort_index()
    low, high = ST.block_bootstrap_mean(values.values)
    ic_low, ic_high = ST.block_bootstrap_mean(ic_values.values)
    return {
        "quarters": rows, "n_days": int(len(values)),
        "mean": float(values.mean()), "ci": (low, high), "significant": low > 0,
        "ic": float(ic_values.mean()), "ic_ci": (ic_low, ic_high),
        "ic_significant": ic_low > 0,
        "hit_days": float((values.values > 0).mean()),
        "series": values,
    }


def universe_comparison(conn, min_days: int = 120) -> dict:
    """O mesmo protocolo, quatro universos, unica diferenca: quem esta' na lista.

    Os paineis vem todos de barras DIARIAS da mesma fonte, para a comparacao
    nao misturar granularidades. `min_days` e' baixo de proposito -- uma accao
    que so' existe parte do periodo tem de poder entrar, porque e' precisamente
    isso que caracteriza as que faltam.

    Os dois universos do meio sao o que separa as duas explicacoes possiveis:
    vies de sobrevivencia (o painel exclui perdedores que la' deviam estar) ou
    falta de generalizacao (o modelo so' funciona em grandes capitalizacoes)."""
    survivors = list(config.TREE_TICKERS)
    liquid = set(surviving_fallen(conn))
    removed = [t for t in INDEX_REMOVED if t in liquid]
    speculative = [t for t in NEVER_LARGE_CAP if t in liquid]

    universes = {
        "A. so' sobreviventes (a ronda 7)": survivors,
        "B. + caidas do indice": survivors + removed,
        "C. + especulativas": survivors + speculative,
        "D. + ambas": survivors + removed + speculative,
    }

    out = {"removed": removed, "speculative": speculative, "results": {},
           "corrupt": []}
    for name, tickers in universes.items():
        rejected = []
        frame = P.build(conn, tickers=tickers, min_days=min_days, timeframe="1d",
                        excluded=rejected)
        out["corrupt"] = sorted(set(out["corrupt"]) | set(rejected))
        if frame.empty:
            continue
        outcome = quarterly_walkforward(frame)
        if not outcome:
            continue
        outcome["n_tickers"] = int(frame["ticker"].nunique())
        out["results"][name] = outcome

    # Diferencas emparelhadas contra o universo A: os universos partilham os
    # mesmos dias, e comparar medias soltas desperdicaria essa estrutura.
    base = out["results"].get("A. so' sobreviventes (a ronda 7)")
    if base is not None:
        for name, result in out["results"].items():
            if name.startswith("A."):
                continue
            joined = pd.concat([base["series"], result["series"]], axis=1,
                               keys=["a", "b"]).dropna()
            difference = (joined["b"] - joined["a"]).values
            low, high = ST.block_bootstrap_mean(difference)
            result["vs_base"] = {"mean": float(difference.mean()), "ci": (low, high),
                                 "n_days": int(len(difference)),
                                 "significant": low > 0 or high < 0}
    return out


def universe_stability(conn, seeds=(42, 7, 101), min_days: int = 120) -> pd.DataFrame:
    """A comparacao entre universos repetida com varias sementes.

    A ronda 7 ensinou que trocar a semente do XGBoost move os resultados deste
    projecto tanto como as diferencas que se quer medir. Uma comparacao entre
    universos feita com uma unica semente nao teria valor nenhum -- reportaria
    uma amostra de uma distribuicao sem dizer que a distribuicao existe."""
    original = config.DECISION["xgb_params"]["random_state"]
    rows = []
    for seed in seeds:
        config.DECISION["xgb_params"]["random_state"] = seed
        outcome = universe_comparison(conn, min_days=min_days)
        row = {"semente": seed}
        row.update({name.split(".")[0]: result["mean"]
                    for name, result in outcome["results"].items()})
        rows.append(row)
    config.DECISION["xgb_params"]["random_state"] = original
    return pd.DataFrame(rows).set_index("semente")


def run(conn, panel_frame: pd.DataFrame = None) -> dict | None:
    frame = panel_frame if panel_frame is not None else P.build(conn)
    if frame.empty:
        return None

    train, validation, test = P.split(frame)
    fit = pd.concat([train, validation])
    columns = ST.feature_sets(frame)["transversal (percentis do dia)"]
    model = ST.fit(fit, columns)
    confidence = ST.score(model, test, columns)

    leg_series = legs(test, confidence)
    leg_rows = []
    for name, series in leg_series.items():
        low, high = ST.block_bootstrap_mean(series.values)
        leg_rows.append({"perna": name, "mean": float(series.mean()),
                         "ci": (low, high), "significant": low > 0,
                         "share": None})
    total = sum(r["mean"] for r in leg_rows)
    for row in leg_rows:
        row["share"] = row["mean"] / total if total else np.nan

    return {
        "n_tickers": int(frame["ticker"].nunique()),
        "attrition": int((frame.groupby("ticker")["ts"].max()
                          < frame["ts"].max() - pd.Timedelta(days=30)).sum()),
        "test_days": P.effective_days(test),
        "legs": leg_rows,
        "total": total,
        "crash": crash_response(test, confidence),
        "strata": fragility_strata(test, confidence),
    }


if __name__ == "__main__":
    connection = database.get_connection()
    cache = Path(config.DATA_DIR) / "decision_panel.pkl"
    cached = pd.read_pickle(cache) if cache.exists() else None
    outcome = run(connection, cached)
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print("=" * 88)
    print("EXPERIENCIA K -- o vies de sobrevivencia explica o resultado?")
    print(f"  {outcome['n_tickers']} accoes, {outcome['test_days']} dias de teste")
    print(f"  accoes que desaparecem antes do fim do painel: {outcome['attrition']}"
          f"  <- um indice real perderia 15 a 25 em tres anos")

    print()
    print("  TESTE 1 -- de que perna vem o resultado?")
    print(f"    {'perna':>12} {'retorno 5d':>12} {'IC95% (blocos)':>26} {'quota':>8}")
    for row in outcome["legs"]:
        low, high = row["ci"]
        mark = "*" if row["significant"] else " "
        print(f"    {row['perna']:>12} {row['mean']:>+12.4%}{mark} "
              f"[{low:>+8.4%}, {high:>+8.4%}] {row['share']:>8.1%}")
    print(f"    {'total':>12} {outcome['total']:>+12.4%}")

    print()
    print("  TESTE 2 -- o modelo poe as quedas violentas do lado vendido?")
    crash = outcome["crash"]
    print(f"    limiar de 'queda' (2% piores)     : {crash['crash_threshold']:+.2%}"
          f"  ({crash['n_crashes']:,} casos)")
    print(f"    posto medio de confianca nas quedas: {crash['crash_mean_rank']:.4f}"
          f"   (0.5 = indiferente)")
    print(f"    posto medio nas subidas violentas  : {crash['spike_mean_rank']:.4f}")
    if crash["crash_p_value"] is not None:
        print(f"    p-value (Wilcoxon contra 0.5)      : {crash['crash_p_value']:.2e}")

    print()
    print("  TESTE 3 -- o efeito concentra-se nas accoes seguras?")
    print(f"    {'grupo':>14} {'n':>4} {'drawdown':>10} {'IC medio':>10} "
          f"{'IC95% (blocos)':>26}")
    for _, row in outcome["strata"].iterrows():
        low, high = row["ci"]
        mark = "*" if row["significant"] else " "
        print(f"    {row['grupo']:>14} {row['n_tickers']:>4} "
              f"{row['drawdown_medio']:>10.1%} {row['ic']:>+10.4f}{mark} "
              f"[{low:>+8.4f}, {high:>+8.4f}]")

    print()
    print("  TESTE 4 -- repor as accoes que faltam (o teste directo)")
    comparison = universe_comparison(connection)
    print(f"    {len(comparison['removed'])} caidas do indice + "
          f"{len(comparison['speculative'])} especulativas repostas; "
          f"{len(DELISTED_NO_DATA)} delistadas impossiveis de recuperar")
    if comparison["corrupt"]:
        print(f"    rejeitadas por dados impossiveis (|retorno diario| > 100%): "
              f"{', '.join(comparison['corrupt'])}")
    print()
    print(f"    {'universo':>34} {'n':>5} {'ret 5d':>10} {'IC95% (blocos)':>26} "
          f"{'IC':>9}  {'vs A':>10}")
    for name, result in comparison["results"].items():
        low, high = result["ci"]
        mark = "*" if result["significant"] else " "
        against = ""
        if "vs_base" in result:
            change = "!" if result["vs_base"]["significant"] else " "
            against = f"{result['vs_base']['mean']:>+10.4%}{change}"
        print(f"    {name:>34} {result['n_tickers']:>5} {result['mean']:>+10.4%}{mark} "
              f"[{low:>+8.4%}, {high:>+8.4%}] {result['ic']:>+9.4f}  {against}")

    print()
    print("    Por trimestre:")
    names = list(comparison["results"])
    header = "".join(f"{n.split('.')[0]:>12}" for n in names)
    print(f"      {'trimestre':>10}{header}")
    quarters = comparison["results"][names[0]]["quarters"]
    for index in range(len(quarters)):
        row = ""
        for name in names:
            rows = comparison["results"][name]["quarters"]
            row += f"{rows[index]['mean']:>+12.3%}" if index < len(rows) else f"{'--':>12}"
        print(f"      {quarters[index]['quarter']:>10}{row}")

    print()
    print("    ! = a diferenca face ao universo A e' significativa")

    print()
    print("  ESTABILIDADE -- a mesma comparacao com outras sementes:")
    spread = universe_stability(connection)
    print(spread.to_string(float_format=lambda v: f"{v:+.4%}"))
    print()
    print(f"    {'universo':>10} {'media':>10} {'amplitude':>11}")
    for column in spread.columns:
        print(f"    {column:>10} {spread[column].mean():>+10.4%} "
              f"{spread[column].max() - spread[column].min():>11.4%}")
    queda = spread["A"].mean() - spread["D"].mean()
    print()
    print(f"    Queda media de A para D: {queda:+.4%} "
          f"({queda / spread['A'].mean() * 100:.0f}% do efeito original)")

    print()
    print("  * = intervalo de confianca inteiramente acima de zero")
    connection.close()
