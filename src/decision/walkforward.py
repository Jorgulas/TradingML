"""AVALIACAO EM WALK-FORWARD -- o unico teste que ainda pode decidir isto.

O problema que resolve, e que e' o mesmo desde a ronda 6. Todas as medicoes
deste projecto acabam em "o sinal aponta na direccao certa mas o intervalo de
confianca toca no zero". Nao e' azar nem falta de sinal: e' falta de dados.
Um teste unico de 144 dias com horizonte de 5 dias sao ~28 janelas
independentes, e com 28 pontos so' um efeito enorme seria detectavel.

O que ainda ha' para fazer sem esperar anos por historico novo: parar de usar
apenas os ultimos 20% da amostra como teste. Em walk-forward re-treina-se tudo
de trimestre a trimestre e cada trimestre e' avaliado com modelos que nunca o
viram. Todos os trimestres a partir do sexto passam a ser fora da amostra, e o
periodo de avaliacao multiplica-se por cerca de tres.

O QUE SE MANTEM INTACTO em cada passo, porque e' isso que torna o resultado
valido:

  - os modelos de sinal (curto, rapido, alarme) sao treinados so' com dados
    anteriores ao trimestre;
  - os tercis do regime tambem;
  - a tabela e' construida so' com dados anteriores;
  - o limiar e' escolhido nos ultimos 20% da janela de treino, nunca no
    trimestre a avaliar.

O CUSTO DE HONESTIDADE: os trimestres nao sao independentes uns dos outros --
sao periodos contiguos do mesmo mercado, e um ano mau contamina os quatro. O
bootstrap de blocos continua a ser aplicado sobre a serie inteira. Isto aumenta
o poder estatistico de forma real, mas nao o multiplica por tres.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.decision import panel as P
from src.decision import signals as S
from src.decision import truth_table as T
from src.decision.short_term import block_bootstrap_mean

import numpy as np
import pandas as pd

D = config.DECISION

MIN_TRAIN_QUARTERS = 6      # 18 meses antes de a primeira decisao contar
VALIDATION_TAIL = 0.2       # fatia final do treino onde se escolhe o limiar


def quarters(frame: pd.DataFrame) -> list:
    stamps = pd.to_datetime(pd.Series(frame["ts"].unique()))
    return sorted(stamps.dt.to_period("Q").astype(str).unique())


def _schemes(train: pd.DataFrame, validation: pd.DataFrame,
             evaluation: pd.DataFrame) -> dict:
    """Os tres esquemas em competicao, cada um com o seu limiar."""
    out = {}
    for label, column in (("tabela", "estimate"),
                          ("tabela crua", "cell_mean"),
                          ("votacao", "additive")):
        on_train, _ = T.table_estimator(train, S.SIGNAL_NAMES, column)
        threshold, _ = T.choose_threshold(on_train, validation)
        on_fit, _ = T.table_estimator(pd.concat([train, validation]),
                                      S.SIGNAL_NAMES, column)
        out[label] = on_fit(evaluation, threshold)

    # A melhor referencia isolada: o sinal de curto prazo sozinho.
    pieces = T.single_signal_tables(pd.concat([train, validation]))
    train_pieces = T.single_signal_tables(train)
    estimator = lambda f, t: T.apply_single(train_pieces["curto"], "curto", f, t)
    threshold, _ = T.choose_threshold(estimator, validation)
    out["so' o curto"] = T.apply_single(pieces["curto"], "curto", evaluation, threshold)
    return out


def run(conn, panel_frame: pd.DataFrame = None, verbose: bool = True,
        seed: int = None) -> dict | None:
    if seed is not None:
        D["xgb_params"]["random_state"] = seed
    frame = panel_frame if panel_frame is not None else P.build(conn)
    if frame.empty:
        return None

    frame = frame.copy()
    frame["quarter"] = pd.to_datetime(frame["ts"]).dt.to_period("Q").astype(str)
    labels = quarters(frame)
    if len(labels) <= MIN_TRAIN_QUARTERS:
        return None

    collected = {}
    per_quarter = []

    for index in range(MIN_TRAIN_QUARTERS, len(labels)):
        label = labels[index]
        history = frame[frame["quarter"] < label]
        evaluation_raw = frame[frame["quarter"] == label]
        if len(history) < 5000 or evaluation_raw.empty:
            continue

        # Embargo: o alvo de 5 dias da ultima barra do treino cai ja' dentro do
        # trimestre avaliado. Sem cortar isto, treino e avaliacao partilham
        # dias.
        cutoff = evaluation_raw["ts"].min() - pd.Timedelta(days=D["horizon_days"] + 2)
        history = history[history["ts"] <= cutoff]
        if len(history) < 5000:
            continue

        # Sinais treinados apenas com o historico deste passo.
        window = pd.concat([history, evaluation_raw])
        enriched = S.attach(conn, window, window["ts"] <= cutoff)
        past = P.labelled(enriched[enriched["ts"] <= cutoff])
        # So' linhas com futuro conhecido entram na avaliacao: as ultimas barras
        # do painel existem para poderem ser decididas hoje, mas ainda nao ha'
        # resultado com que as pontuar.
        evaluation = P.labelled(enriched[enriched["quarter"] == label])
        if evaluation.empty:
            continue

        stamps = np.sort(past["ts"].unique())
        boundary = stamps[int(len(stamps) * (1 - VALIDATION_TAIL))]
        train = past[past["ts"] < boundary]
        validation = past[past["ts"] >= boundary]
        if train.empty or validation.empty:
            continue

        actions = _schemes(train, validation, evaluation)
        row = {"quarter": label, "train_days": P.effective_days(train),
               "eval_days": P.effective_days(evaluation)}
        for name, action in actions.items():
            daily = T.daily_result(evaluation, action)
            collected.setdefault(name, []).append(daily)
            row[name] = float(daily.mean())
        per_quarter.append(row)
        if verbose:
            parts = "  ".join(f"{k}={row[k]:+.3%}" for k in actions)
            print(f"  {label}: treino {row['train_days']:>3}d, "
                  f"avaliacao {row['eval_days']:>2}d | {parts}", flush=True)

    if not collected:
        return None

    series = {name: pd.concat(pieces).sort_index() for name, pieces in collected.items()}
    results = []
    for name, values in series.items():
        array = values.values
        low, high = block_bootstrap_mean(array)
        results.append({
            "name": name, "mean": float(array.mean()), "ci": (low, high),
            "significant": low > 0, "n_days": int(len(array)),
            "hit_days": float((array > 0).mean()),
        })

    difference = (series["tabela"] - series["votacao"]).dropna()
    diff_low, diff_high = block_bootstrap_mean(difference.values)

    return {
        "quarters": per_quarter,
        "results": sorted(results, key=lambda r: -r["mean"]),
        "n_days": int(len(series["tabela"])),
        "independent_windows": int(len(series["tabela"]) / D["horizon_days"]),
        "interaction_difference": float(difference.mean()),
        "interaction_ci": (diff_low, diff_high),
        "interaction_significant": diff_low > 0,
    }


def stability(conn, panel_frame: pd.DataFrame, seeds=(42, 7, 101)) -> dict:
    """Repete o walk-forward inteiro com sementes diferentes.

    Porque isto e' obrigatorio e nao um extra. A meio deste trabalho corrigi um
    desequilibrio de um nome em sessenta nos tercis -- uma alteracao minuscula --
    e um trimestre mexeu quase um ponto percentual, com a ordem entre esquemas a
    trocar. Ou o efeito e' fragil, ou a medicao e' ruidosa; de qualquer forma
    reportar um numero unico seria reportar uma amostra de uma distribuicao sem
    dizer que ela existe.

    Trocar a semente do XGBoost muda apenas a amostragem interna das arvores.
    Nada no problema muda -- mesmos dados, mesmo alvo, mesmo protocolo. A
    dispersao que sair e' portanto um piso para a incerteza, nao o total dela."""
    original = D["xgb_params"]["random_state"]
    runs = []
    for seed in seeds:
        outcome = run(conn, panel_frame, verbose=False, seed=seed)
        if outcome:
            runs.append({"seed": seed,
                         **{r["name"]: r["mean"] for r in outcome["results"]},
                         "diferenca": outcome["interaction_difference"]})
        if verbose_stability:
            print(f"  semente {seed}: " + "  ".join(
                f"{k}={v:+.3%}" for k, v in runs[-1].items() if k != "seed"), flush=True)
    D["xgb_params"]["random_state"] = original

    if not runs:
        return {}
    frame = pd.DataFrame(runs).set_index("seed")
    winners = frame.drop(columns=["diferenca"]).idxmax(axis=1)
    return {
        "runs": runs,
        "mean": frame.mean().to_dict(),
        "spread": (frame.max() - frame.min()).to_dict(),
        "winners": winners.value_counts().to_dict(),
    }


verbose_stability = True


if __name__ == "__main__":
    connection = database.get_connection()
    cache = Path(config.DATA_DIR) / "decision_panel.pkl"
    cached = pd.read_pickle(cache) if cache.exists() else None

    print("=" * 96)
    print("WALK-FORWARD -- re-treino trimestral, tudo fora da amostra")
    print()
    outcome = run(connection, cached)
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print()
    print(f"  ACUMULADO: {outcome['n_days']} dias fora da amostra "
          f"(~{outcome['independent_windows']} janelas independentes)")
    print(f"    {'esquema':>16} {'retorno/decisao':>16} {'IC95% (blocos)':>26} {'dias +':>8}")
    for r in outcome["results"]:
        low, high = r["ci"]
        mark = "*" if r["significant"] else " "
        print(f"    {r['name']:>16} {r['mean']:>+16.4%}{mark} "
              f"[{low:>+8.4%}, {high:>+8.4%}] {r['hit_days']:>8.1%}")

    print()
    low, high = outcome["interaction_ci"]
    print(f"  Tabela menos votacao simples: {outcome['interaction_difference']:+.4%}  "
          f"IC95% [{low:+.4%}, {high:+.4%}]")
    verdict = ("as interaccoes acrescentam valor mensuravel"
               if outcome["interaction_significant"]
               else "dentro do ruido: as interaccoes NAO se demonstram")
    print(f"  -> {verdict}")
    print()
    print("  * = intervalo de confianca inteiramente acima de zero")

    print()
    print("  ESTABILIDADE -- o mesmo walk-forward com outras sementes:")
    spread = stability(connection, cached)
    if spread:
        print()
        print(f"    {'esquema':>16} {'media':>10} {'amplitude':>11} {'vezes 1o':>9}")
        for name, value in sorted(spread["mean"].items(), key=lambda kv: -kv[1]):
            if name == "diferenca":
                continue
            print(f"    {name:>16} {value:>+10.4%} {spread['spread'][name]:>11.4%} "
                  f"{spread['winners'].get(name, 0):>9}")
        print(f"    {'tabela - votacao':>16} {spread['mean']['diferenca']:>+10.4%} "
              f"{spread['spread']['diferenca']:>11.4%}")
        print()
        print("    Se a amplitude entre sementes for da ordem da diferenca entre")
        print("    esquemas, a diferenca entre esquemas nao significa nada.")
    connection.close()
