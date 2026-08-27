"""EXPERIENCIA J -- dimensionar pela CAUDA prevista, nao pela volatilidade.

Porque vale a pena repetir a ronda 4. A experiencia I revelou uma coisa que
explica porque e' que aquela falhou:

    volatilidade recente, a prever caudas:  AUC 0.5071  (acaso puro)
    modelo de cauda + resultados:           AUC 0.6633

Na ronda 4 dimensionei posicoes pelo inverso da volatilidade recente (a
referencia gratuita) e por um modelo de magnitude que correlacionava 0.78 com
ela. NENHUM dos dois tinha poder preditivo sobre caudas -- e as caudas sao
precisamente o que produz os piores dias e os drawdowns. Estava a medir a coisa
certa (variancia, drawdown) com o preditor errado.

Aqui dimensiona-se pelo P(cauda) previsto e mede-se o que essa previsao devia
melhorar: PIOR DIA, VaR 99% e DRAWDOWN. A variancia total pode ate' nao mudar
-- reduzir caudas sem reduzir variancia geral seria na mesma um bom resultado,
e mais coerente com o que se sabe do modelo.

Referencia gratuita continua a ser o inverso da volatilidade recente, porque e'
o que qualquer mesa ja' faz. Exposicao identica em todos os esquemas.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.research.tail_probability import feature_sets, prepare

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

TRADING_DAYS = 252


def portfolio_returns(frame: pd.DataFrame, weights: pd.Series) -> pd.Series:
    clipped = weights.clip(lower=1e-9)
    normalised = clipped / clipped.groupby(frame["ts"]).transform("sum")
    return (frame["next_return"] * normalised).groupby(frame["ts"]).sum()


def summarise(returns: pd.Series, label: str) -> dict:
    values = returns.values
    cumulative = np.cumprod(1 + values)
    drawdown = cumulative / np.maximum.accumulate(cumulative) - 1
    return {
        "name": label,
        "annual_vol": float(values.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "worst_day": float(values.min()),
        "var99": float(np.percentile(values, 1)),
        "var95": float(np.percentile(values, 5)),
        "max_drawdown": float(drawdown.min()),
        "worst_5_mean": float(np.sort(values)[:5].mean()),
    }


def run(conn) -> dict | None:
    fit_df, test_df, _ = prepare(conn)
    if test_df.empty:
        return None

    # Retorno com sinal do dia seguinte -- e' o que a carteira sofre.
    for part in (fit_df, test_df):
        part.sort_values(["ticker", "ts"], inplace=True)
        part["next_return"] = part.groupby("ticker")["abs_return"].shift(-1) * np.where(
            part.groupby("ticker")["ret_1"].shift(-1) >= 0, 1.0, -1.0)

    test = test_df.dropna(subset=["next_return"]).copy()
    columns = feature_sets(fit_df)["modelo completo + resultados"]
    params = {k: v for k, v in config.TREES["xgb_params"].items()
              if k not in ("objective", "num_class")}

    model = XGBClassifier(**params, max_depth=4, objective="binary:logistic")
    model.fit(fit_df[columns], fit_df["is_tail"])
    test["tail_probability"] = model.predict_proba(test[columns])[:, 1]

    base_rate = float(fit_df["is_tail"].mean())
    schemes = {
        "pesos iguais": pd.Series(1.0, index=test.index),
        "inverso da volatilidade (gratis)": 1.0 / test["expected_move"],
        "inverso do risco de cauda previsto": 1.0 / test["tail_probability"].clip(lower=1e-3),
        # Penalizacao suave: corta so' quando a cauda esta' mesmo elevada,
        # em vez de reescalar tudo continuamente.
        "corte nas caudas previstas": pd.Series(
            np.where(test["tail_probability"] > 3 * base_rate, 0.25, 1.0), index=test.index),
        "volatilidade + corte nas caudas": (1.0 / test["expected_move"]) * pd.Series(
            np.where(test["tail_probability"] > 3 * base_rate, 0.25, 1.0), index=test.index),
    }

    series = {label: portfolio_returns(test, weights) for label, weights in schemes.items()}
    results = [summarise(values, label) for label, values in series.items()]

    free = series["inverso da volatilidade (gratis)"].values
    rng = np.random.default_rng(23)
    comparisons = {}
    for label in ("inverso do risco de cauda previsto", "corte nas caudas previstas",
                  "volatilidade + corte nas caudas"):
        candidate = series[label].values
        differences = []
        for _ in range(2000):
            idx = rng.integers(0, len(free), len(free))
            differences.append(np.percentile(candidate[idx], 1) - np.percentile(free[idx], 1))
        low, high = np.percentile(differences, [2.5, 97.5])
        comparisons[label] = {
            "var99_difference": float(np.percentile(candidate, 1) - np.percentile(free, 1)),
            "ci": (float(low), float(high)),
            "improves": low > 0,   # VaR e' negativo: subir e' melhorar
        }

    return {
        "n_days": len(free), "n_tickers": int(test["ticker"].nunique()),
        "base_rate": base_rate, "results": results, "comparisons": comparisons,
    }


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection)
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print("=" * 92)
    print("EXPERIENCIA J -- dimensionar pela cauda prevista")
    print(f"  {outcome['n_days']} dias, {outcome['n_tickers']} tickers, "
          f"taxa-base de cauda {outcome['base_rate']:.2%}, exposicao identica")
    print()
    print(f"  {'esquema':>36} {'vol anual':>10} {'pior dia':>10} {'VaR 99%':>9} "
          f"{'5 piores':>10} {'drawdown':>10}")
    for r in outcome["results"]:
        print(f"  {r['name']:>36} {r['annual_vol']:>9.2%} {r['worst_day']:>10.2%} "
              f"{r['var99']:>9.2%} {r['worst_5_mean']:>10.2%} {r['max_drawdown']:>10.2%}")

    print()
    print("  Contra a referencia GRATUITA (inverso da volatilidade), no VaR 99%:")
    for label, comparison in outcome["comparisons"].items():
        low, high = comparison["ci"]
        verdict = ("MELHORA de forma significativa" if comparison["improves"]
                   else "dentro do ruido")
        print(f"    {label:>36}: {comparison['var99_difference']:+.4%}  "
              f"IC95% [{low:+.4%}, {high:+.4%}]  -> {verdict}")
    connection.close()
