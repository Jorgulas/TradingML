"""TABELA DE VERDADE -- todos os algoritmos a votar na mesma decisao.

A ideia. Cada algoritmo emite um veredicto (ver signals.py). Cinco veredictos
com 3, 3, 3, 3 e 2 valores dao 162 combinacoes possiveis. Para cada combinacao
mede-se o que ACONTECEU historicamente, e a tabela passa a dizer o que fazer
quando essa combinacao voltar a aparecer.

O QUE ISTO ACRESCENTA A UMA VOTACAO. Somar votos assume que os sinais sao
independentes e aditivos: dois sinais optimistas valem o dobro de um. Uma tabela
nao assume nada disso -- deixa espaco para "o padrao grafico so' funciona em
regime calmo" ou "o modelo rapido contradiz o de 5 dias e ai' e' melhor nao
fazer nada". Sao INTERACCOES, e sao invisiveis a qualquer soma de votos.

O PERIGO, que e' a razao de metade deste ficheiro. 162 celulas sobre 144 dias de
teste dao menos de um dia por celula. Enchendo a tabela com medias brutas,
algumas celulas vao ter numeros espectaculares por puro acaso, e escolher a
melhor e' garantir que se escolhe a mais sortuda. E' o problema classico de
comparacoes multiplas, agravado por os 158 tickers de um mesmo dia nao serem
158 observacoes independentes mas praticamente uma.

TRES DEFESAS, todas necessarias:

1. SUPORTE EM DIAS, nao em linhas. Uma celula com 800 linhas concentradas em 6
   dias tem 6 observacoes.

2. ENCOLHIMENTO PARA O MODELO ADITIVO. Cada celula nao usa a sua propria media
   crua, mas uma interpolacao entre ela e o que a soma dos efeitos marginais
   preveria. O peso e' n_dias/(n_dias + k): uma celula muito observada fala por
   si, uma celula rara diz o mesmo que a votacao simples diria. A tabela nunca
   fica pior do que a votacao por falta de dados -- so' se afasta dela onde ha'
   mesmo evidencia para isso.

3. A REGRA E' FIXADA ANTES DE VER O TESTE. Tabela construida em treino,
   limiar escolhido em validacao, teste medido uma vez e no fim. E cada esquema
   escolhe o SEU limiar, senao a comparacao e' viciada a favor da tabela.

A COMPARACAO QUE DECIDE nao e' "a tabela ganha dinheiro?" mas "a tabela ganha
MAIS do que a soma dos sinais?". Se nao ganhar, as interaccoes nao existem e a
resposta honesta e' que uma votacao simples chega -- que e' um resultado util,
so' menos entusiasmante.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.decision import panel as P
from src.decision import signals as S
from src.decision.short_term import STOCK_COST_BPS, block_bootstrap_mean

import numpy as np
import pandas as pd

D = config.DECISION


# ---------------------------------------------------------------------------
# Construcao da tabela
# ---------------------------------------------------------------------------

def marginals(fit: pd.DataFrame, target: str, names: list = None) -> tuple:
    """Efeito de cada sinal isolado, e a media global.

    Sao estes os efeitos que a tabela encolhe -- e sao tambem, somados, a
    votacao simples contra a qual ela tem de provar valer a pena."""
    overall = float(fit[target].mean())
    by_signal = {}
    for name in (names or S.SIGNAL_NAMES):
        grouped = fit.groupby(name, observed=True)[target].mean()
        by_signal[name] = (grouped - overall).to_dict()
    return overall, by_signal


def additive_prediction(frame: pd.DataFrame, overall: float, by_signal: dict) -> pd.Series:
    """A votacao simples: media global mais a soma dos desvios de cada sinal."""
    total = pd.Series(overall, index=frame.index, dtype=float)
    for name, effects in by_signal.items():
        total = total + frame[name].map(effects).fillna(0.0)
    return total


def build_table(fit: pd.DataFrame, target: str = "forward_return",
                group_by: list = None) -> pd.DataFrame:
    """Uma linha por combinacao observada, com a estimativa encolhida.

    `weight` e' o peso dado a' propria celula. Explicita-se como coluna porque
    e' a quantidade que diz quanto da tabela e' interaccao genuina e quanto e'
    apenas a votacao vestida de tabela."""
    group_by = group_by or S.SIGNAL_NAMES
    overall, by_signal = marginals(fit, target, names=group_by)
    k = D["backoff_k"]

    grouped = fit.groupby(group_by, observed=True)
    table = grouped.agg(
        n_rows=(target, "size"),
        cell_mean=(target, "mean"),
        cell_std=(target, "std"),
        p_up=("label", lambda s: float((s == P.UP).mean())),
    )
    table["n_days"] = grouped["ts"].nunique()
    table = table.reset_index()

    table["additive"] = additive_prediction(table, overall, by_signal)
    table["weight"] = table["n_days"] / (table["n_days"] + k)
    table["estimate"] = (table["weight"] * table["cell_mean"]
                         + (1 - table["weight"]) * table["additive"])
    table["interaction"] = table["cell_mean"] - table["additive"]
    table["overall"] = overall
    return table.sort_values("estimate", ascending=False).reset_index(drop=True)


def verdicts(table: pd.DataFrame, threshold: float, column: str = "estimate") -> pd.DataFrame:
    """COMPRAR / VENDER / ESPERAR por celula.

    ESPERAR e' a decisao por omissao e nao um resto: sem vantagem estimada
    suficiente, nao agir e' estritamente melhor do que agir, porque agir custa
    spread e nao agir custa zero."""
    table = table.copy()
    table["accao"] = np.where(table[column] > threshold, "COMPRAR",
                              np.where(table[column] < -threshold, "VENDER", "ESPERAR"))
    return table


def apply_table(table: pd.DataFrame, frame: pd.DataFrame,
                column: str = "accao", names: list = None) -> pd.Series:
    """Le a decisao de cada linha na tabela. Combinacoes nunca vistas ficam
    ESPERAR, que e' a resposta certa para "nunca vi isto acontecer"."""
    names = names or S.SIGNAL_NAMES
    lookup = table.set_index(names)[column]
    keys = pd.MultiIndex.from_frame(frame[names])
    return pd.Series(lookup.reindex(keys).values, index=frame.index).fillna("ESPERAR")


# ---------------------------------------------------------------------------
# Avaliacao
# ---------------------------------------------------------------------------

def daily_result(frame: pd.DataFrame, action: pd.Series,
                 target: str = "forward_return") -> pd.Series:
    """Retorno por dia de seguir a tabela: media dos COMPRAR menos media dos
    VENDER, com peso igual dentro de cada lado.

    Comprar e vender ao mesmo tempo mantem a carteira neutra ao mercado, o que
    e' coerente com o alvo ja' ser retorno em excesso. Um dia em que a tabela so'
    diz ESPERAR rende zero -- e conta como dia, porque nao agir tambem e' um
    resultado e omiti-lo inflacionaria a media."""
    work = frame[["ts", target]].copy()
    work["accao"] = action.values
    out = {}
    for stamp, group in work.groupby("ts"):
        longs = group.loc[group["accao"] == "COMPRAR", target]
        shorts = group.loc[group["accao"] == "VENDER", target]
        value = 0.0
        if len(longs):
            value += float(longs.mean())
        if len(shorts):
            value -= float(shorts.mean())
        out[stamp] = value
    return pd.Series(out).sort_index()


def turnover_and_cost(frame: pd.DataFrame, action: pd.Series) -> tuple:
    """Rotacao medida entre rebalanceamentos espacados pelo horizonte, e o custo
    que dela decorre.

    Medir em vez de assumir importa aqui mais do que parece: um sinal que muda
    de opiniao todos os dias paga o spread todos os dias, e uma vantagem bruta
    de meio ponto percentual desaparece depressa. Duas pontas por ciclo, sobre a
    exposicao bruta (comprada mais vendida)."""
    work = frame[["ts", "ticker"]].copy()
    work["accao"] = action.values
    # Um esquema que nunca age nao paga spread nenhum. Sem este caso, o valor
    # por omissao (rotacao total) aparecia como custo de uma carteira vazia.
    if not (work["accao"] != "ESPERAR").any():
        return 0.0, 0.0
    stamps = sorted(work["ts"].unique())[::D["horizon_days"]]

    previous, changes = None, []
    for stamp in stamps:
        held = work[(work["ts"] == stamp) & (work["accao"] != "ESPERAR")]
        names = set(zip(held["ticker"], held["accao"]))
        if previous:
            changes.append(1 - len(names & previous) / len(previous))
        previous = names or previous
    turnover = float(np.mean(changes)) if changes else 1.0
    return turnover, 2 * (STOCK_COST_BPS / 10_000) * turnover


def summarise(daily: pd.Series, frame: pd.DataFrame, action: pd.Series,
              label: str) -> dict:
    values = daily.values
    low, high = block_bootstrap_mean(values)
    acted = (action != "ESPERAR")
    turnover, cost = turnover_and_cost(frame, action)
    mean = float(values.mean()) if len(values) else float("nan")
    return {
        "name": label,
        "mean": mean,
        "ci": (low, high),
        "significant": low > 0,
        "hit_days": float((values > 0).mean()) if len(values) else float("nan"),
        "coverage": float(acted.mean()),
        "n_days": int(len(values)),
        "turnover": turnover,
        "cost": cost,
        "net_mean": mean - cost,
        "net_significant": (low - cost) > 0,
    }


def choose_threshold(estimator, validation: pd.DataFrame, candidates=None) -> tuple:
    """Limiar escolhido na validacao, com CADA esquema a escolher o seu.

    Um limiar unico partilhado seria uma comparacao viciada: os efeitos
    marginais de um sinal isolado sao da ordem de 0.1% e os de uma celula da
    tabela chegam a 3%, portanto um limiar afinado para a tabela poe todos os
    sinais isolados a ESPERAR sempre, e a tabela "ganha" contra adversarios que
    nunca chegaram a jogar. Cada esquema recebe o seu melhor limiar, medido na
    validacao.

    So' cinco candidatos, de proposito: cada um e' uma comparacao a mais, e
    varrer cinquenta limiares seria escolher o mais sortudo com outro nome."""
    candidates = candidates or [0.0, 0.001, 0.002, 0.004, 0.008]
    best, best_score = candidates[0], -np.inf
    for threshold in candidates:
        score = daily_result(validation, estimator(validation, threshold)).mean()
        if np.isfinite(score) and score > best_score:
            best, best_score = threshold, score
    return best, best_score


def single_signal_tables(fit: pd.DataFrame, target: str = "forward_return") -> dict:
    """Uma "tabela" por sinal isolado. E' a fasquia mais dura da comparacao: se
    a tabela inteira nao bater o melhor sinal sozinho, juntar os sinais nao
    serviu de nada."""
    overall = float(fit[target].mean())
    out = {}
    for name in S.SIGNAL_NAMES:
        grouped = fit.groupby(name, observed=True)
        piece = grouped.agg(n_rows=(target, "size"), cell_mean=(target, "mean")).reset_index()
        piece["n_days"] = grouped["ts"].nunique().values
        piece["estimate"] = piece["cell_mean"]
        piece["overall"] = overall
        out[name] = piece
    return out


def apply_single(piece: pd.DataFrame, name: str, frame: pd.DataFrame,
                 threshold: float) -> pd.Series:
    lookup = piece.set_index(name)["estimate"]
    estimate = frame[name].map(lookup)
    return pd.Series(np.where(estimate > threshold, "COMPRAR",
                              np.where(estimate < -threshold, "VENDER", "ESPERAR")),
                     index=frame.index).fillna("ESPERAR")


def table_estimator(source: pd.DataFrame, names: list, column: str = "estimate"):
    """Devolve (funcao_de_decisao, tabela) construidas sobre `source`.

    Empacotar assim serve para o limiar de cada esquema poder ser escolhido na
    validacao exactamente com o mecanismo que depois vai ao teste, sem risco de
    as duas metades divergirem."""
    table = build_table(source, group_by=names)

    def estimate(frame, threshold):
        return apply_table(verdicts(table, threshold, column=column), frame, names=names)

    return estimate, table


def run(conn, enriched: pd.DataFrame = None) -> dict | None:
    if enriched is None:
        frame = P.build(conn)
        _, validation_split, _ = P.split(frame)
        enriched = S.attach(conn, frame, frame["ts"] < validation_split["ts"].max())

    train, validation, test = P.split(enriched)
    if test.empty or train.empty:
        return None
    fit = pd.concat([train, validation])

    def scheme(label, names, column):
        """Constroi em treino, escolhe o limiar em validacao, reconstroi em
        treino+validacao e so' entao toca no teste. Uma unica passagem."""
        on_train, _ = table_estimator(train, names, column)
        threshold, _ = choose_threshold(on_train, validation)
        on_fit, final = table_estimator(fit, names, column)
        action = on_fit(test, threshold)
        row = summarise(daily_result(test, action), test, action, label)
        row["threshold"] = threshold
        return row, action, final

    rows = []
    full_row, full_action, final_table = scheme(
        "tabela de verdade (encolhida)", S.SIGNAL_NAMES, "estimate")
    rows.append(full_row)

    raw_row, _, _ = scheme("tabela crua (sem encolhimento)", S.SIGNAL_NAMES, "cell_mean")
    rows.append(raw_row)

    additive_row, additive_action, _ = scheme(
        "votacao simples (aditiva)", S.SIGNAL_NAMES, "additive")
    rows.append(additive_row)

    # Cada sinal sozinho, cada um com o seu limiar.
    on_fit_singles = single_signal_tables(fit)
    on_train_singles = single_signal_tables(train)
    for name in S.SIGNAL_NAMES:
        train_piece = on_train_singles[name]
        estimator = (lambda n, p: lambda f, t: apply_single(p, n, f, t))(name, train_piece)
        threshold, _ = choose_threshold(estimator, validation)
        action = apply_single(on_fit_singles[name], name, test, threshold)
        row = summarise(daily_result(test, action), test, action, f"so' o sinal '{name}'")
        row["threshold"] = threshold
        rows.append(row)

    # ABLACAO: a tabela sem cada um dos sinais. Diz quais e' que pagam a
    # complexidade que impoem -- cada sinal a mais multiplica as celulas e
    # divide os dados por elas.
    ablation = []
    for dropped in S.SIGNAL_NAMES:
        remaining = [n for n in S.SIGNAL_NAMES if n != dropped]
        row, _, _ = scheme(f"tabela sem '{dropped}'", remaining, "estimate")
        row["dropped"] = dropped
        row["delta"] = full_row["mean"] - row["mean"]
        ablation.append(row)

    # Diferenca tabela - votacao, com IC: e' a medicao directa das interaccoes.
    difference = (daily_result(test, full_action)
                  - daily_result(test, additive_action)).dropna()
    diff_low, diff_high = block_bootstrap_mean(difference.values)

    return {
        "threshold": full_row["threshold"],
        "n_cells": len(final_table),
        "n_cells_supported": int((final_table["n_days"] >= D["min_cell_days"]).sum()),
        "mean_weight": float(final_table["weight"].mean()),
        "median_days_per_cell": float(final_table["n_days"].median()),
        "train_days": P.effective_days(train),
        "validation_days": P.effective_days(validation),
        "test_days": P.effective_days(test),
        "results": rows,
        "ablation": ablation,
        "interaction_difference": float(difference.mean()),
        "interaction_ci": (diff_low, diff_high),
        "interaction_significant": diff_low > 0,
        "table": verdicts(final_table, full_row["threshold"]),
    }


def _print_cells(table: pd.DataFrame, n: int = 12) -> None:
    header = "  ".join(f"{name:>8}" for name in S.SIGNAL_NAMES)
    print(f"    {header}  {'dias':>5} {'linhas':>7} {'celula':>9} {'aditivo':>9} "
          f"{'peso':>6} {'usado':>9}  accao")
    ordered = table.sort_values("estimate", ascending=False)
    show = pd.concat([ordered.head(n // 2), ordered.tail(n // 2)])
    for _, row in show.iterrows():
        cells = "  ".join(f"{row[name]:>8}" for name in S.SIGNAL_NAMES)
        print(f"    {cells}  {row['n_days']:>5} {row['n_rows']:>7} "
              f"{row['cell_mean']:>+9.3%} {row['additive']:>+9.3%} "
              f"{row['weight']:>6.2f} {row['estimate']:>+9.3%}  {row['accao']}")


if __name__ == "__main__":
    connection = database.get_connection()
    cache = Path(config.DATA_DIR) / "decision_signals.pkl"
    cached = pd.read_pickle(cache) if cache.exists() else None
    outcome = run(connection, cached)
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print("=" * 108)
    print("TABELA DE VERDADE -- decisao combinada de todos os algoritmos")
    print(f"  {outcome['n_cells']} celulas, {outcome['n_cells_supported']} com "
          f">= {D['min_cell_days']} dias de suporte | mediana de "
          f"{outcome['median_days_per_cell']:.0f} dias por celula")
    print(f"  peso medio dado a' propria celula: {outcome['mean_weight']:.2f} "
          f"(0 = votacao pura, 1 = tabela pura)")
    print(f"  teste = {outcome['test_days']} dias, "
          f"~{outcome['test_days'] // D['horizon_days']} janelas independentes")

    print()
    print("  RESULTADO NO TESTE (retorno a 5 dias por dia de decisao):")
    print(f"    {'esquema':>34} {'limiar':>7} {'bruto':>9} {'IC95% (blocos)':>26} "
          f"{'dias +':>7} {'cobert.':>8} {'rotacao':>8} {'liquido':>9}")
    for r in outcome["results"]:
        low, high = r["ci"]
        mark = "*" if r["significant"] else " "
        net = "*" if r["net_significant"] else " "
        print(f"    {r['name']:>34} {r['threshold']:>7.2%} {r['mean']:>+9.4%}{mark} "
              f"[{low:>+8.4%}, {high:>+8.4%}] {r['hit_days']:>7.1%} "
              f"{r['coverage']:>8.1%} {r['turnover']:>8.0%} {r['net_mean']:>+8.4%}{net}")

    print()
    print("  ABLACAO -- quanto se perde ao tirar cada sinal da tabela:")
    print(f"    {'sinal retirado':>18} {'retorno sem ele':>16} {'perda':>10}")
    for r in sorted(outcome["ablation"], key=lambda x: -x["delta"]):
        print(f"    {r['dropped']:>18} {r['mean']:>+16.4%} {r['delta']:>+10.4%}")

    print()
    print("  A PERGUNTA QUE DECIDE -- a tabela bate a votacao simples?")
    low, high = outcome["interaction_ci"]
    print(f"    diferenca (tabela - votacao): {outcome['interaction_difference']:+.4%}  "
          f"IC95% [{low:+.4%}, {high:+.4%}]")
    verdict = ("as interaccoes acrescentam valor mensuravel"
               if outcome["interaction_significant"]
               else "dentro do ruido: as interaccoes NAO se demonstram")
    print(f"    -> {verdict}")

    print()
    print("  Extremos da tabela (6 celulas mais optimistas, 6 mais pessimistas):")
    _print_cells(outcome["table"])
    print()
    print("  * = intervalo de confianca inteiramente acima de zero")
    connection.close()
