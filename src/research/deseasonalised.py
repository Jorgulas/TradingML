"""EXPERIENCIA D -- depois de tirar o relogio e a persistencia, sobra alguma coisa?

A experiencia C mostrou que 74% do poder preditivo do modelo de 40 features
vem de oito dummies de hora do dia, e 96% de hora + HAR. Ou seja: quase todo o
"acerto" e' saber que a volatilidade e' alta na abertura e no fecho, e que
quem esteve volatil tende a continuar.

Nada disso e' previsao de mercado. E' calendario e inercia.

A pergunta que sobra e' a unica interessante: existe SURPRESA previsivel? Isto
e', depois de descontar o que a hora do dia e a volatilidade recente ja'
explicam, resta estrutura que as outras features consigam apanhar?

METODO. Constroi-se um alvo residualizado:

  1. estima-se, SO' com dados de treino, a volatilidade tipica de cada faixa
     horaria para cada ticker (a forma do "U" intradiario);
  2. divide-se o movimento realizado por essa expectativa -> resta a surpresa;
  3. treina-se sobre esse residuo e mede-se se ainda ha' sinal.

Se o Spearman colapsar para perto de zero, a conclusao e' definitiva: o modelo
nunca previu nada -- soube as horas. Seria um resultado negativo forte e util,
porque fecha a questao em vez de a deixar em aberto.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.research.har_benchmark import build_frame
from src.research.har_plus_seasonality import N_SESSION_BUCKETS
from src.trees import magnitude as M
from src.trees import train as tree_train

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor


def seasonal_profile(reference: pd.DataFrame) -> pd.Series:
    """Volatilidade tipica por (ticker, faixa horaria), estimada APENAS no
    conjunto de referencia. Estimar isto no teste seria deixar o teste
    explicar-se a si proprio."""
    bucket = np.clip((reference["session_position"] * N_SESSION_BUCKETS).astype(int),
                     0, N_SESSION_BUCKETS - 1)
    return reference.assign(_bucket=bucket).groupby(["ticker", "_bucket"])["abs_return"].median()


def apply_profile(frame: pd.DataFrame, profile: pd.Series) -> pd.Series:
    bucket = np.clip((frame["session_position"] * N_SESSION_BUCKETS).astype(int),
                     0, N_SESSION_BUCKETS - 1)
    keys = pd.MultiIndex.from_arrays([frame["ticker"], bucket])
    expected = profile.reindex(keys).values
    # Tickers/faixas sem historico no treino ficam com a mediana global
    return pd.Series(np.where(np.isnan(expected), np.nanmedian(profile.values), expected),
                     index=frame.index)


def run(conn, timeframe: str = "1h") -> dict | None:
    frame = build_frame(conn, timeframe)
    if frame.empty:
        return None

    train_df, validation_df, test_df = tree_train.split_by_time(
        frame, config.TREES["horizon_bars"])
    fit_df = pd.concat([train_df, validation_df])

    profile = seasonal_profile(fit_df)
    for part in (fit_df, test_df):
        expected = apply_profile(part, profile)
        part["expected_vol"] = expected
        # Surpresa: quantas vezes o movimento excedeu o que era tipico
        # naquela accao, naquela altura do dia.
        part["surprise"] = part["abs_return"] / expected.replace(0, np.nan)

    fit_df = fit_df.dropna(subset=["surprise"])
    test_df = test_df.dropna(subset=["surprise"])

    har_columns = [c for c in frame.columns if c.startswith("har_")]
    full_features = [c for c in frame.columns
                     if c not in ("ticker", "ts", "big_move", "magnitude_ratio",
                                  "abs_return", "band")]

    def evaluate(predictions, realised, label):
        deciles = M.decile_lift(predictions, realised)
        return {
            "name": label,
            "spearman": float(stats.spearmanr(predictions, realised).statistic),
            "top_bottom": (deciles[-1]["mean_realised"] / deciles[0]["mean_realised"]
                           if deciles[0]["mean_realised"] else None),
        }

    def fit_tree(columns, target, label):
        model = XGBRegressor(**{k: v for k, v in config.TREES["xgb_params"].items()
                                if k not in ("objective", "num_class")},
                             max_depth=4, objective="reg:squarederror")
        model.fit(fit_df[columns], fit_df[target])
        return evaluate(model.predict(test_df[columns]), test_df[target].values, label)

    # Referencia: quanto valia o modelo no alvo BRUTO (com sazonalidade dentro)
    raw = fit_tree(full_features, "abs_return", "alvo bruto |retorno|")
    # A pergunta a serio: e no alvo residualizado?
    surprise = fit_tree(full_features, "surprise", "alvo SURPRESA (des-sazonalizado)")
    surprise_har = fit_tree(har_columns, "surprise", "surpresa, so' com HAR")

    # Quanto do "U" existe mesmo? Amplitude do perfil sazonal.
    by_bucket = fit_df.groupby(
        np.clip((fit_df["session_position"] * N_SESSION_BUCKETS).astype(int),
                0, N_SESSION_BUCKETS - 1))["abs_return"].median()

    return {
        "timeframe": timeframe, "n_test": len(test_df),
        "effective_days": int(pd.Series(test_df["ts"]).dt.normalize().nunique()),
        "results": [raw, surprise, surprise_har],
        "seasonal_shape": by_bucket.round(6).to_dict(),
        "seasonal_amplitude": float(by_bucket.max() / by_bucket.min()) if by_bucket.min() else None,
    }


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection, "1h")
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print(f"{'='*74}")
    print("EXPERIENCIA D -- sobra sinal depois de tirar a sazonalidade e a persistencia?")
    print(f"  @ {outcome['timeframe']}, {outcome['n_test']:,} linhas, "
          f"{outcome['effective_days']} dias efectivos")
    print()
    print("  O 'U' intradiario medido (|retorno| mediano por faixa da sessao):")
    for bucket, value in outcome["seasonal_shape"].items():
        bar = "#" * int(value * 12000)
        print(f"    faixa {bucket}/{N_SESSION_BUCKETS-1}  {value:.5f}  {bar}")
    print(f"    amplitude: a faixa mais volatil move-se "
          f"{outcome['seasonal_amplitude']:.2f}x a mais calma")
    print()
    print(f"  {'modelo / alvo':>34} {'spearman':>10} {'decis':>9}")
    for r in outcome["results"]:
        print(f"  {r['name']:>34} {r['spearman']:>10.4f} {(r['top_bottom'] or 0):>8.2f}x")

    raw, surprise = outcome["results"][0]["spearman"], outcome["results"][1]["spearman"]
    kept = surprise / raw * 100 if raw else 0
    print()
    print(f"  Ao tirar a sazonalidade, o Spearman cai de {raw:.4f} para {surprise:.4f} "
          f"({kept:.0f}% do original).")
    if surprise < 0.05:
        print("  -> NAO SOBRA NADA. O modelo sabia as horas, nao previa o mercado.")
    elif surprise < 0.15:
        print("  -> sobra pouco, mas nao zero: ha' alguma surpresa previsivel.")
    else:
        print("  -> sobra sinal a serio para la' do calendario e da inercia.")
    connection.close()
