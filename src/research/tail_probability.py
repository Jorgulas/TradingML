"""EXPERIENCIA I -- modelar a CAUDA em vez da magnitude.

A ideia vem de um diagnostico especifico, nao de querer um modelo mais
complicado. Na ronda 3 o decil de topo tinha:

    mediana da surpresa = 0.975   (praticamente normal)
    mas 2.98x mais eventos do top 1% que o acaso

Isto NAO e' uma mudanca de escala. Se a distribuicao inteira fosse multiplicada
por 1.3, a mediana subia 30%. Ela nao sobe -- o centro fica onde estava e sao
os EXTREMOS que ficam desproporcionalmente mais provaveis. E' uma mudanca na
ESPESSURA DA CAUDA.

E eu tenho estado a treinar com erro quadratico, que ajusta a media condicional.
Com um alvo de cauda pesada, a media e' dominada pelos outliers e nao descreve
bem nem o centro nem a cauda. Estava a modelar a quantidade errada.

TRES ALVOS, sobre exactamente as mesmas features, para separar o que e' o que:

  escala   E[|r|]                 -- a magnitude tipica (o que fiz ate' agora)
  cauda    P(|r| > 3 x esperado)  -- a probabilidade de rebentar
  quantil  Q90[|r|]               -- regressao quantilica: "quao mau pode ser"

Se o modelo previr a CAUDA melhor do que preve a ESCALA, confirma-se que a
competencia que ele tem e' de cauda -- e todas as avaliacoes anteriores, feitas
em metricas de escala, subestimaram-no.

Junta-se tambem o CALENDARIO DE RESULTADOS, que era o dado em falta: a
hipotese da ronda 3 era que os extremos sao eventos agendados. Agora da' para
verificar, e para perguntar o que importa: o modelo sabe alguma coisa ALEM de
saber que ha resultados?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.research.daily_horizon import ROLLING_WINDOW, build, split
from src.research.earnings_data import load as load_earnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

TAIL_MULTIPLE = 3.0        # |r| > 3x o esperado conta como cauda
QUANTILE = 0.90


def add_earnings_features(frame: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """Distancia ao proximo/ultimo resultado, e o historico de reaccoes.

    Tudo calculado sem olhar para o futuro do PRECO. As datas de resultados
    futuras sao conhecidas com semanas de antecedencia -- e' informacao
    legitimamente disponivel, e e' precisamente a que o mercado de opcoes tem
    e o meu movimento esperado nao tinha.

    RESSALVA HONESTA, porque nao e' isenta de risco: o calendario foi buscado
    HOJE, logo contem datas que, num dia distante do periodo de teste, ainda
    nao tinham sido anunciadas. Para as features que decidem -- `is_earnings_day`
    (±1 dia) e `earnings_window` (±3 dias) -- isso nao e' problema: as empresas
    anunciam a data 2 a 4 semanas antes, portanto a poucos dias do evento ela
    era mesmo conhecida. Ja' `days_to_earnings` com valores altos (60+ dias)
    pode conter informacao que nao estava disponivel na altura, embora os
    trimestres sejam previsiveis a ~90 dias. E' um lookahead residual, e nao
    afecta as conclusoes que dependem da janela curta."""
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["ts"]).dt.normalize()
    frame["days_to_earnings"] = 999.0
    frame["days_since_earnings"] = 999.0
    frame["is_earnings_day"] = 0.0
    frame["earnings_window"] = 0.0

    if earnings.empty:
        return frame

    by_ticker = {t: g["earnings_date"].values for t, g in earnings.groupby("ticker")}
    for ticker, group in frame.groupby("ticker"):
        dates = by_ticker.get(ticker)
        if dates is None or len(dates) == 0:
            continue
        stamps = group["date"].values.astype("datetime64[ns]")
        known = np.sort(np.asarray(dates, dtype="datetime64[ns]"))

        following = np.searchsorted(known, stamps, side="left")
        to_next = np.where(following < len(known),
                           (known[np.clip(following, 0, len(known) - 1)] - stamps)
                           / np.timedelta64(1, "D"), 999.0)
        previous = following - 1
        since = np.where(previous >= 0,
                         (stamps - known[np.clip(previous, 0, len(known) - 1)])
                         / np.timedelta64(1, "D"), 999.0)

        frame.loc[group.index, "days_to_earnings"] = np.clip(to_next, 0, 999)
        frame.loc[group.index, "days_since_earnings"] = np.clip(since, 0, 999)

    frame["is_earnings_day"] = ((frame["days_to_earnings"] <= 1) |
                               (frame["days_since_earnings"] <= 1)).astype(float)
    frame["earnings_window"] = ((frame["days_to_earnings"] <= 3) |
                                (frame["days_since_earnings"] <= 3)).astype(float)
    return frame


def prepare(conn) -> tuple:
    frame = build(conn)
    earnings = load_earnings(conn)
    frame = add_earnings_features(frame, earnings)

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
        part["is_tail"] = (part["surprise"] > TAIL_MULTIPLE).astype(float)

    return (fit_df[np.isfinite(fit_df["surprise"])],
            test_df[np.isfinite(test_df["surprise"])].copy(),
            earnings)


def feature_sets(frame: pd.DataFrame) -> dict:
    excluded = {"ticker", "ts", "date", "big_move", "magnitude_ratio", "abs_return",
                "band", "surprise", "expected_move", "is_tail"}
    earnings_columns = ["days_to_earnings", "days_since_earnings",
                        "is_earnings_day", "earnings_window"]
    full = [c for c in frame.columns if c not in excluded]
    return {
        "so volatilidade recente": ["expected_move"],
        "so calendario de resultados": earnings_columns,
        "volatilidade + resultados": ["expected_move"] + earnings_columns,
        "modelo completo (sem resultados)": [c for c in full if c not in earnings_columns],
        "modelo completo + resultados": full,
    }


def run(conn) -> dict | None:
    fit_df, test_df, earnings = prepare(conn)
    if test_df.empty:
        return None

    sets = feature_sets(fit_df)
    base_rate = float(test_df["is_tail"].mean())
    params = {k: v for k, v in config.TREES["xgb_params"].items()
              if k not in ("objective", "num_class")}

    tail_results = []
    for label, columns in sets.items():
        model = XGBClassifier(**params, max_depth=4, objective="binary:logistic")
        model.fit(fit_df[columns], fit_df["is_tail"])
        probabilities = model.predict_proba(test_df[columns])[:, 1]
        top = test_df["is_tail"].values[probabilities >= np.quantile(probabilities, 0.9)]
        tail_results.append({
            "name": label, "n_features": len(columns),
            "auc": float(roc_auc_score(test_df["is_tail"], probabilities)),
            "top_decile_rate": float(top.mean()),
            "lift": float(top.mean() / base_rate) if base_rate else None,
        })

    # Comparacao de alvos: escala vs cauda vs quantil, com as mesmas features
    full = sets["modelo completo + resultados"]
    scale = XGBRegressor(**params, max_depth=4, objective="reg:squarederror")
    scale.fit(fit_df[full], fit_df["surprise"])
    scale_predictions = scale.predict(test_df[full])

    quantile = XGBRegressor(**params, max_depth=4,
                            objective="reg:quantileerror", quantile_alpha=QUANTILE)
    quantile.fit(fit_df[full], fit_df["surprise"])
    quantile_predictions = quantile.predict(test_df[full])

    tail_model = XGBClassifier(**params, max_depth=4, objective="binary:logistic")
    tail_model.fit(fit_df[full], fit_df["is_tail"])
    tail_predictions = tail_model.predict_proba(test_df[full])[:, 1]

    targets = []
    for label, predictions in (("escala  E[|r|]", scale_predictions),
                               (f"quantil Q{int(QUANTILE*100)}[|r|]", quantile_predictions),
                               (f"cauda   P(|r|>{TAIL_MULTIPLE:.0f}x)", tail_predictions)):
        cutoff = np.quantile(predictions, 0.9)
        selected = predictions >= cutoff
        targets.append({
            "name": label,
            "auc_tail": float(roc_auc_score(test_df["is_tail"], predictions)),
            "top_decile_tail_rate": float(test_df["is_tail"].values[selected].mean()),
            "lift": float(test_df["is_tail"].values[selected].mean() / base_rate)
            if base_rate else None,
        })

    earnings_days = test_df[test_df["earnings_window"] > 0]
    normal_days = test_df[test_df["earnings_window"] == 0]

    return {
        "n_test": len(test_df), "base_rate": base_rate,
        "n_tickers_with_earnings": int(earnings["ticker"].nunique()) if not earnings.empty else 0,
        "tail_results": tail_results, "targets": targets,
        "earnings_share": float((test_df["earnings_window"] > 0).mean()),
        "tail_rate_earnings": float(earnings_days["is_tail"].mean()) if len(earnings_days) else None,
        "tail_rate_normal": float(normal_days["is_tail"].mean()) if len(normal_days) else None,
        "surprise_earnings": float(earnings_days["surprise"].mean()) if len(earnings_days) else None,
        "surprise_normal": float(normal_days["surprise"].mean()) if len(normal_days) else None,
    }


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection)
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print("=" * 80)
    print("EXPERIENCIA I -- modelar a CAUDA, e o que os resultados trimestrais explicam")
    print(f"  {outcome['n_test']:,} linhas | taxa-base de cauda "
          f"(|r| > {TAIL_MULTIPLE:.0f}x esperado): {outcome['base_rate']:.2%}")
    print(f"  {outcome['n_tickers_with_earnings']} tickers com calendario de resultados")

    print()
    print("  A HIPOTESE DA RONDA 3: os extremos sao dias de resultados?")
    print(f"    janela de resultados cobre {outcome['earnings_share']:.1%} dos dias")
    print(f"    taxa de cauda em dias de resultados : {outcome['tail_rate_earnings']:.2%}")
    print(f"    taxa de cauda em dias normais       : {outcome['tail_rate_normal']:.2%}")
    if outcome["tail_rate_normal"]:
        print(f"    -> resultados tem {outcome['tail_rate_earnings']/outcome['tail_rate_normal']:.2f}x "
              f"mais eventos de cauda")
    print(f"    surpresa media: resultados {outcome['surprise_earnings']:.3f} "
          f"vs normais {outcome['surprise_normal']:.3f}")

    print()
    print("  O MODELO SABE ALGO ALEM DO CALENDARIO?")
    print(f"    {'conjunto de features':>36} {'n':>4} {'AUC':>8} {'lift no decil':>14}")
    for r in outcome["tail_results"]:
        print(f"    {r['name']:>36} {r['n_features']:>4} {r['auc']:>8.4f} "
              f"{(r['lift'] or 0):>13.2f}x")

    print()
    print("  QUAL O ALVO CERTO? (mesmas features, avaliados na cauda)")
    print(f"    {'alvo treinado':>24} {'AUC na cauda':>14} {'lift no decil':>14}")
    for r in outcome["targets"]:
        print(f"    {r['name']:>24} {r['auc_tail']:>14.4f} {(r['lift'] or 0):>13.2f}x")
    connection.close()
