"""EXPERIENCIA G -- o alarme de risco reduz mesmo a variancia da carteira?

A ronda 3 mostrou que o modelo tem competencia genuina a detectar risco de
cauda: apanha 29.8% dos movimentos do top 1% contra 10% do acaso, um lift de
2.98x. Como sinal de compra de volatilidade falhou (o mercado de opcoes ja'
cobra esses eventos), mas dimensionar posicoes nao paga spread de opcoes nem
premio de risco -- e' a aplicacao onde a competencia que ha' e' a competencia
de que se precisa.

A COMPARACAO JUSTA NAO E' CONTRA PESOS IGUAIS.
Dimensionar pelo inverso da volatilidade recente e' pratica corrente e nao
custa nada. Se o modelo nao bater ISSO, nao acrescenta valor -- foi
exactamente o que aconteceu no teste de slippage da ronda 2, onde uma regra de
relogio ganhou ao modelo. As carteiras comparadas sao:

  1. pesos iguais                      -- a referencia ingenua
  2. inverso da volatilidade recente   -- a referencia a SERIO, gratuita
  3. inverso da volatilidade PREVISTA  -- o modelo
  4. combinacao das duas               -- o modelo so' corrige a estimativa

EXPOSICAO IGUAL EM TODAS. Os pesos somam sempre 1. Sem isto, reduzir variancia
seria trivial: bastava investir menos.

Mede-se a volatilidade realizada, mas sobretudo as METRICAS DE CAUDA (pior
dia, VaR 99%, drawdown), porque e' na cauda que esta' a competencia do modelo.
Se ele reduzir a cauda sem reduzir a variancia geral, isso e' na mesma um
resultado util -- e mais coerente com o que se sabe dele.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.research.daily_horizon import ROLLING_WINDOW, build, split

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

TRADING_DAYS = 252


def portfolio_returns(frame: pd.DataFrame, weight_column: str) -> pd.Series:
    """Retorno diario da carteira. Os pesos sao normalizados dentro de cada dia
    para somarem 1 -- exposicao identica em todos os esquemas."""
    weights = frame[weight_column].clip(lower=1e-9)
    normalised = weights / frame.groupby("ts")[weight_column].transform(
        lambda w: w.clip(lower=1e-9).sum())
    return (frame["next_return"] * normalised).groupby(frame["ts"]).sum()


def summarise(returns: pd.Series, label: str) -> dict:
    values = returns.values
    cumulative = np.cumprod(1 + values)
    drawdown = cumulative / np.maximum.accumulate(cumulative) - 1
    return {
        "name": label,
        "n_days": len(values),
        "annual_vol": float(values.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "worst_day": float(values.min()),
        "var99": float(np.percentile(values, 1)),
        "var95": float(np.percentile(values, 5)),
        "max_drawdown": float(drawdown.min()),
        "mean_return": float(values.mean()),
    }


def run(conn) -> dict | None:
    frame = build(conn)
    if frame.empty:
        return None

    # Retorno COM SINAL do dia seguinte -- e' o que a carteira sofre. O resto
    # do projecto usa |retorno|; aqui precisa-se do sinal para haver carteira.
    frame = frame.sort_values(["ticker", "ts"]).copy()
    frame["next_return"] = frame.groupby("ticker")["abs_return"].shift(-1) * np.where(
        frame.groupby("ticker")["ret_1"].shift(-1) >= 0, 1.0, -1.0)

    train_df, validation_df, test_df = split(frame)
    fit_df = pd.concat([train_df, validation_df])

    weekday_factor = (fit_df.assign(_d=pd.to_datetime(fit_df["ts"]).dt.dayofweek)
                      .groupby("_d")["abs_return"].mean())
    weekday_factor = weekday_factor / weekday_factor.mean()

    for part in (fit_df, test_df):
        weekday = pd.to_datetime(part["ts"]).dt.dayofweek
        shape = pd.Series(weekday_factor.reindex(weekday).values, index=part.index).fillna(1.0)
        level = (part.groupby("ticker")["abs_return"]
                 .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=20).mean()))
        part["expected_move"] = level * shape
        part["surprise"] = part["abs_return"] / part["expected_move"].replace(0, np.nan)

    features = [c for c in frame.columns
                if c not in ("ticker", "ts", "big_move", "magnitude_ratio",
                             "abs_return", "band", "next_return", "surprise", "expected_move")]
    fit_clean = fit_df[np.isfinite(fit_df["surprise"])]
    model = XGBRegressor(**{k: v for k, v in config.TREES["xgb_params"].items()
                            if k not in ("objective", "num_class")},
                         max_depth=4, objective="reg:squarederror")
    model.fit(fit_clean[features], fit_clean["surprise"])

    test = test_df[np.isfinite(test_df["surprise"])].dropna(subset=["next_return"]).copy()
    test["predicted_surprise"] = model.predict(test[features])
    # Risco previsto em unidades absolutas: surpresa prevista x movimento esperado
    test["predicted_risk"] = test["predicted_surprise"].clip(lower=0.1) * test["expected_move"]

    test["w_equal"] = 1.0
    test["w_inverse_trailing"] = 1.0 / test["expected_move"]
    test["w_inverse_predicted"] = 1.0 / test["predicted_risk"]
    # Combinacao: parte da estimativa gratuita e deixa o modelo corrigi-la
    test["w_blend"] = 1.0 / (0.5 * test["expected_move"] + 0.5 * test["predicted_risk"])

    schemes = [
        ("pesos iguais", "w_equal"),
        ("inverso da volatilidade recente (gratis)", "w_inverse_trailing"),
        ("inverso da volatilidade prevista (modelo)", "w_inverse_predicted"),
        ("combinacao das duas", "w_blend"),
    ]
    series = {label: portfolio_returns(test, column) for label, column in schemes}
    results = [summarise(values, label) for label, values in series.items()]

    # Significancia da diferenca entre o modelo e a referencia gratuita,
    # por bootstrap sobre dias.
    free = series["inverso da volatilidade recente (gratis)"].values
    modelled = series["inverso da volatilidade prevista (modelo)"].values
    rng = np.random.default_rng(17)
    differences = []
    for _ in range(2000):
        idx = rng.integers(0, len(free), len(free))
        differences.append(modelled[idx].std(ddof=1) - free[idx].std(ddof=1))
    low, high = np.percentile(differences, [2.5, 97.5])

    return {
        "n_days": len(free), "n_tickers": int(test["ticker"].nunique()),
        "results": results,
        "vol_difference": float(modelled.std(ddof=1) - free.std(ddof=1)),
        "vol_difference_ci": (float(low), float(high)),
        "model_helps": high < 0,
        "model_hurts": low > 0,
    }


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection)
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print("=" * 84)
    print("EXPERIENCIA G -- o alarme de risco reduz a variancia da carteira?")
    print(f"  {outcome['n_days']} dias de teste, {outcome['n_tickers']} tickers, "
          f"exposicao identica em todos os esquemas")
    print()
    print(f"  {'esquema de pesos':>42} {'vol anual':>10} {'pior dia':>10} "
          f"{'VaR 99%':>9} {'drawdown':>10}")
    for r in outcome["results"]:
        print(f"  {r['name']:>42} {r['annual_vol']:>9.2%} {r['worst_day']:>10.2%} "
              f"{r['var99']:>9.2%} {r['max_drawdown']:>10.2%}")

    baseline = outcome["results"][1]   # inverso da volatilidade recente
    model = outcome["results"][2]
    print()
    print("  O que interessa: MODELO vs a referencia GRATUITA (inverso da vol recente)")
    print(f"    volatilidade anual : {model['annual_vol']:.2%} vs {baseline['annual_vol']:.2%}  "
          f"({model['annual_vol'] - baseline['annual_vol']:+.2%})")
    print(f"    pior dia           : {model['worst_day']:.2%} vs {baseline['worst_day']:.2%}")
    print(f"    VaR 99%            : {model['var99']:.2%} vs {baseline['var99']:.2%}")
    print(f"    drawdown maximo    : {model['max_drawdown']:.2%} vs {baseline['max_drawdown']:.2%}")
    low, high = outcome["vol_difference_ci"]
    print()
    print(f"    diferenca de volatilidade diaria: {outcome['vol_difference']:+.5f}  "
          f"IC95% [{low:+.5f}, {high:+.5f}]")
    if outcome["model_helps"]:
        print("    -> O MODELO REDUZ a volatilidade de forma significativa.")
    elif outcome["model_hurts"]:
        print("    -> O MODELO AUMENTA a volatilidade. Piora face a regra gratuita.")
    else:
        print("    -> Diferenca dentro do ruido: o modelo NAO acrescenta nada")
        print("       ao que a volatilidade recente ja' dava de graca.")
    connection.close()
