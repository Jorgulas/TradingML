"""EXPERIENCIA B -- a direccao e' previsivel em ALGUM regime?

A media de 53.5% de acerto direccional pode esconder heterogeneidade: talvez
seja 50% quase sempre e 58% num subconjunto identificavel a' partida (alta
volatilidade, pico de volume, abertura ou fecho da sessao, depois de um
movimento grande). Se existisse esse regime, negociar so' la' dentro seria
uma estrategia.

RISCO METODOLOGICO DESTA EXPERIENCIA, e e' grave: testar 20 subconjuntos e
reportar o melhor e' pescar resultados. Com 20 testes a 5%, espera-se UM falso
positivo so' por acaso. Duas defesas:

  1. correccao de Bonferroni -- o limiar de significancia e' dividido pelo
     numero de testes feitos;
  2. os regimes sao definidos em treino+validacao e so' depois medidos no
     teste, nunca escolhidos por olhar para o teste.

O resultado esperado, dado tudo o resto, e' que nenhum regime sobreviva. Isso
seria um resultado util: fecha a porta a "e se procurassemos melhor".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.trees import features as F
from src.trees import train as tree_train

import numpy as np
import pandas as pd
from scipy import stats


def define_regimes(frame: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Regimes definidos por quantis calculados no conjunto de REFERENCIA
    (treino+validacao). Aplicar quantis do proprio teste seria deixar o teste
    influenciar a definicao dos grupos."""
    def cut(column, quantile):
        return reference[column].quantile(quantile) if column in reference else None

    regimes = {}

    high_vol = cut("vol_20", 0.8)
    if high_vol is not None:
        regimes["volatilidade alta (top 20%)"] = frame["vol_20"] > high_vol
        regimes["volatilidade baixa (bottom 20%)"] = frame["vol_20"] < cut("vol_20", 0.2)

    high_volume = cut("volume_z20", 0.8)
    if high_volume is not None:
        regimes["pico de volume (top 20%)"] = frame["volume_z20"] > high_volume

    if "session_position" in frame:
        regimes["abertura da sessao (1o quinto)"] = frame["session_position"] < 0.2
        regimes["fecho da sessao (ultimo quinto)"] = frame["session_position"] > 0.8

    if "ret_1" in frame:
        big = reference["ret_1"].abs().quantile(0.9)
        regimes["depois de movimento grande (top 10%)"] = frame["ret_1"].abs() > big
        regimes["depois de subida forte"] = frame["ret_1"] > reference["ret_1"].quantile(0.9)
        regimes["depois de descida forte"] = frame["ret_1"] < reference["ret_1"].quantile(0.1)

    if "rsi14" in frame:
        regimes["RSI sobrecomprado (>70)"] = frame["rsi14"] > 70
        regimes["RSI sobrevendido (<30)"] = frame["rsi14"] < 30

    if "bb_width" in frame:
        regimes["bandas apertadas (bottom 20%)"] = frame["bb_width"] < cut("bb_width", 0.2)

    if "corr_60" in frame:
        regimes["baixa correlacao com o indice"] = frame["corr_60"] < cut("corr_60", 0.2)

    if "beta_60" in frame:
        regimes["beta alto"] = frame["beta_60"] > cut("beta_60", 0.8)

    return regimes


def directional_accuracy(subset: pd.DataFrame) -> dict | None:
    """Acerto direccional apenas nas barras que se mexeram mesmo -- prever a
    direccao de uma barra que ficou dentro da banda nao e' uma pergunta com
    resposta."""
    moved = subset[subset["label"] != F.WAIT]
    if len(moved) < 200:
        return None

    actual_up = (moved["label"] == F.UP).values
    # "Direccao correcta" aqui e' a persistencia: o proximo movimento repete o
    # sinal do anterior? E' o preditor direccional mais simples que existe, e
    # serve para saber se HA estrutura direccional no regime, sem depender de
    # nenhum modelo treinado.
    previous_up = (moved["ret_1"] > 0).values
    persistence = float((previous_up == actual_up).mean())

    # Erro-padrao sobre DIAS distintos, nao sobre linhas
    days = pd.Series(moved["ts"]).dt.normalize().nunique()
    standard_error = float(np.sqrt(0.25 / max(days, 1)))

    return {
        "n": len(moved), "days": int(days),
        "up_rate": float(actual_up.mean()),
        "persistence_accuracy": persistence,
        "standard_error": standard_error,
        "z": (persistence - 0.5) / standard_error if standard_error else 0.0,
    }


def run(conn, timeframe: str = "1h") -> dict | None:
    frame = tree_train.build_dataset(conn, timeframe, "indicators")
    if frame.empty:
        return None

    train_df, validation_df, test_df = tree_train.split_by_time(
        frame, config.TREES["horizon_bars"])
    reference = pd.concat([train_df, validation_df])
    regimes = define_regimes(test_df, reference)

    overall = directional_accuracy(test_df)
    results = []
    for name, mask in regimes.items():
        outcome = directional_accuracy(test_df[mask])
        if outcome:
            results.append({"regime": name, **outcome})

    # Bonferroni: com N testes, o limiar de |z| sobe para manter 5% global.
    n_tests = max(len(results), 1)
    bonferroni_z = float(stats.norm.ppf(1 - 0.05 / (2 * n_tests)))

    return {
        "timeframe": timeframe, "overall": overall, "regimes": results,
        "n_tests": n_tests, "bonferroni_z": bonferroni_z,
    }


if __name__ == "__main__":
    connection = database.get_connection()
    for tf in ("1h", "5m"):
        outcome = run(connection, tf)
        if outcome is None:
            print(f"{tf}: dados insuficientes")
            continue

        overall = outcome["overall"]
        print(f"\n{'='*78}")
        print(f"EXPERIENCIA B -- ha algum regime onde a direccao seja previsivel? @ {tf}")
        print(f"  global: persistencia acerta {overall['persistence_accuracy']:.1%} "
              f"(n={overall['n']:,}, {overall['days']} dias, +-{overall['standard_error']*100:.1f}pp, "
              f"z={overall['z']:+.2f})")
        print(f"  {outcome['n_tests']} regimes testados -> limiar de Bonferroni |z| > "
              f"{outcome['bonferroni_z']:.2f}")
        print()
        print(f"  {'regime':>36} {'acerto':>8} {'n':>8} {'dias':>6} {'z':>7}  veredicto")
        for r in sorted(outcome["regimes"], key=lambda x: -abs(x["z"])):
            verdict = ("SIGNIFICATIVO" if abs(r["z"]) > outcome["bonferroni_z"]
                       else ("sugestivo" if abs(r["z"]) > 1.96 else "ruido"))
            print(f"  {r['regime']:>36} {r['persistence_accuracy']:>7.1%} {r['n']:>8,} "
                  f"{r['days']:>6} {r['z']:>+7.2f}  {verdict}")

        survivors = [r for r in outcome["regimes"] if abs(r["z"]) > outcome["bonferroni_z"]]
        print()
        if survivors:
            print(f"  -> {len(survivors)} regime(s) sobrevive(m) a Bonferroni: "
                  f"{[r['regime'] for r in survivors]}")
        else:
            print("  -> NENHUM regime sobrevive a correccao de multiplos testes.")
    connection.close()
