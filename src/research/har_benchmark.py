"""EXPERIENCIA A -- o XGBoost bate o HAR-RV?

Contexto da literatura. O HAR-RV (Corsi, 2009) preve a volatilidade futura a
partir de tres medias da volatilidade passada: curta, media e longa. Sao tres
coeficientes numa regressao linear. E' o benchmark padrao da area, e varios
estudos comparativos concluem que metodos de machine learning NAO o batem de
forma consistente -- "the classic HAR-RV model consistently outperform both
machine learning approaches, especially when the information set is limited".

Isto e' um problema directo para o resultado anterior deste projecto: a
baseline que usei era a persistencia de volatilidade com UMA feature (vol_10),
o que e' muito mais fraco do que o HAR. Bater uma baseline fraca nao prova
nada. Este ficheiro poe o modelo contra o benchmark a serio.

Se o XGBoost com 40 features nao bater tres coeficientes lineares, a conclusao
honesta e' que as 40 features nao valem a complexidade -- e' um resultado, nao
uma falha do teste.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.patterns.ingest_intraday import load_bars
from src.trees import features as F
from src.trees import magnitude as M
from src.trees import train as tree_train

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import roc_auc_score

# Quantas barras fazem um dia / uma semana / um mes, por timeframe. O HAR
# original e' diario (1, 5, 22); aqui traduz-se para a grelha de cada
# timeframe mantendo as mesmas escalas temporais.
HAR_WINDOWS = {
    "1h": {"short": 1, "day": 7, "week": 35, "month": 140},
    "5m": {"short": 1, "day": 78, "week": 390, "month": 1560},
    "1m": {"short": 1, "day": 390, "week": 1950, "month": 1950},
}


def har_features(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """As tres (quatro) medias de volatilidade realizada do HAR.

    Volatilidade realizada aproximada pelo retorno absoluto -- com barras OHLC
    e sem dados tick a tick, e' o estimador disponivel."""
    windows = HAR_WINDOWS[timeframe]
    absolute_return = bars["close"].pct_change().abs()

    out = pd.DataFrame(index=bars.index)
    for name, window in windows.items():
        out[f"har_{name}"] = absolute_return.rolling(window, min_periods=window).mean()
    return out


def build_frame(conn, timeframe: str, tickers=None) -> pd.DataFrame:
    """HAR + features completas + alvo, para comparacao lado a lado sobre
    exactamente as mesmas linhas."""
    tickers = tickers or config.TREE_TICKERS
    benchmark = load_bars(conn, config.BENCHMARK["ticker"], timeframe)

    frames = []
    for ticker in tickers:
        bars = load_bars(conn, ticker, timeframe)
        if len(bars) < 400:
            continue
        matrix = (F.indicator_features(bars)
                  .join(F.market_context_features(bars, benchmark))
                  .join(har_features(bars, timeframe))
                  .join(F.label_magnitude(bars)))
        matrix["ticker"] = ticker
        matrix["ts"] = matrix.index
        matrix.index.name = None
        frames.append(matrix.dropna())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values(["ts", "ticker"]).reset_index(drop=True)


def compare(conn, timeframe: str = "1h", target: str = "abs_return") -> dict | None:
    """DOIS alvos, porque a comparacao justa depende de qual se escolhe:

      abs_return       -- o NIVEL do proximo movimento. E' o terreno proprio do
                          HAR: foi desenhado exactamente para isto.
      magnitude_ratio  -- o movimento a dividir pela volatilidade recente, ou
                          seja a SURPRESA. E' uma pergunta diferente e mais
                          dificil, e o HAR nao foi feito para ela: a
                          normalizacao divide fora justamente o que ele preve.

    A primeira versao desta experiencia so' testou o segundo alvo e concluiu
    que o XGBoost ganhava por 0.31 de Spearman. Era uma comparacao viciada a
    favor do XGBoost."""
    frame = build_frame(conn, timeframe)
    if frame.empty:
        return None

    train_df, validation_df, test_df = tree_train.split_by_time(frame, config.TREES["horizon_bars"])
    if test_df.empty:
        return None

    har_columns = [c for c in frame.columns if c.startswith("har_")]
    fit_df = pd.concat([train_df, validation_df])
    realised = test_df[target].values
    big_move = test_df["big_move"].astype(int).values

    # RidgeCV e nao OLS: as componentes do HAR sao medias moveis sobrepostas da
    # mesma serie, logo fortemente colineares. Com OLS sairam coeficientes de
    # -67 e +52 a anularem-se -- instabilidade classica de colinearidade, que
    # penaliza o benchmark por um motivo que nada tem a ver com o seu merito.
    har = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(fit_df[har_columns], fit_df[target])
    har_predictions = har.predict(test_df[har_columns])

    # --- XGBoost com todas as features ---
    # Retreinado para o alvo em causa: comparar um modelo treinado no alvo A
    # com outro treinado no alvo B nao diria nada.
    from xgboost import XGBRegressor
    bundle = joblib.load(config.MODELS_DIR / f"magnitude_indicators_{timeframe}.joblib")
    if target == "magnitude_ratio":
        tree = bundle["regressor"]
    else:
        tree = XGBRegressor(**{k: v for k, v in config.TREES["xgb_params"].items()
                               if k not in ("objective", "num_class")},
                            max_depth=bundle["max_depth"], objective="reg:squarederror")
        tree.fit(fit_df[bundle["feature_names"]], fit_df[target])
    tree_predictions = tree.predict(test_df[bundle["feature_names"]])
    tree_probabilities = bundle["classifier"].predict_proba(test_df[bundle["feature_names"]])[:, 1]

    def evaluate(predictions, label):
        deciles = M.decile_lift(predictions, realised)
        return {
            "name": label,
            "spearman": float(stats.spearmanr(predictions, realised).statistic),
            "auc": float(roc_auc_score(big_move, predictions)),
            "top_bottom": (deciles[-1]["mean_realised"] / deciles[0]["mean_realised"]
                           if deciles[0]["mean_realised"] else None),
            "deciles": deciles,
        }

    results = {
        "har": evaluate(har_predictions, "HAR-RV (3 coeficientes)"),
        "tree": evaluate(tree_predictions, "XGBoost (40 features)"),
        "tree_classifier_auc": float(roc_auc_score(big_move, tree_probabilities)),
        "har_coefficients": dict(zip(har_columns, har.coef_.round(4).tolist())),
        "target": target,
        "n_test": len(test_df),
        "effective_days": int(pd.Series(test_df["ts"]).dt.normalize().nunique()),
        "timeframe": timeframe,
    }

    # Combinar os dois: o HAR captura a persistencia de longo prazo, a arvore
    # captura nao-linearidades. Se a combinacao bater ambos, sao complementares.
    combined = np.column_stack([har_predictions, tree_predictions])
    blend = LinearRegression().fit(
        np.column_stack([
            har.predict(fit_df[har_columns]),
            tree.predict(fit_df[bundle["feature_names"]]),
        ]),
        fit_df[target],
    )
    results["combined"] = evaluate(blend.predict(combined), "HAR + XGBoost")
    results["blend_weights"] = blend.coef_.round(4).tolist()
    return results


if __name__ == "__main__":
    connection = database.get_connection()
    for tf in ("1h", "5m"):
        outcome = compare(connection, tf)
        if outcome is None:
            print(f"{tf}: dados insuficientes")
            continue

        print(f"\n{'='*66}")
        print(f"EXPERIENCIA A -- HAR-RV vs XGBoost @ {tf}")
        print(f"  teste: {outcome['n_test']:,} linhas, {outcome['effective_days']} dias efectivos")
        print(f"  coeficientes HAR: {outcome['har_coefficients']}")
        print()
        print(f"  {'modelo':>26} {'spearman':>10} {'AUC':>8} {'decis':>8}")
        for key in ("har", "tree", "combined"):
            r = outcome[key]
            print(f"  {r['name']:>26} {r['spearman']:>10.4f} {r['auc']:>8.4f} "
                  f"{(r['top_bottom'] or 0):>7.2f}x")
        print()
        gap = outcome["tree"]["spearman"] - outcome["har"]["spearman"]
        print(f"  XGBoost - HAR = {gap:+.4f} de Spearman -> ", end="")
        print("XGBoost ganha" if gap > 0.02 else
              ("empate tecnico" if abs(gap) <= 0.02 else "HAR GANHA (as 40 features nao compensam)"))
        print(f"  peso do blend [HAR, XGBoost] = {outcome['blend_weights']}")
    connection.close()
