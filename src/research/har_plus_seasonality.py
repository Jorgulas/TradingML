"""EXPERIENCIA C -- o XGBoost ganha por ser esperto ou por o HAR nao ter relogio?

Na experiencia A o XGBoost bateu o HAR-RV mesmo no terreno do HAR (Spearman
0.44 vs 0.22). Isso CONTRADIZ varios estudos comparativos, e quando um
resultado meu contradiz literatura publicada a hipotese mais provavel e' que
haja uma vantagem injusta escondida.

A suspeita concreta: a feature numero 1 do XGBoost e' `session_position`. O
HAR-RV foi desenhado para dados DIARIOS, onde nao existe hora do dia. Aplicado
a barras de 1h, ele nao tem como capturar a sazonalidade intradiaria -- o "U"
de volatilidade, alta na abertura e no fecho e baixa a meio da sessao. Esse
efeito e' grande, deterministico e conhecido ha' decadas.

Se bastar dar um relogio ao HAR para o fosso fechar, entao a conclusao correcta
nao e' "o machine learning bate a econometria" mas sim "a sazonalidade
intradiaria importa muito, e o HAR nao a modela nesta frequencia". Sao
afirmacoes muito diferentes.

Modelos comparados, todos sobre exactamente as mesmas linhas:
  HAR              -- as medias de volatilidade, so'
  HAR + hora       -- o mesmo, mais dummies de posicao na sessao
  XGBoost(HAR+hora)-- arvore com APENAS essas features (isola a nao-linearidade)
  XGBoost completo -- as 40 features
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.research.har_benchmark import build_frame
from src.trees import magnitude as M
from src.trees import train as tree_train

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

N_SESSION_BUCKETS = 8


def add_session_dummies(frame: pd.DataFrame) -> list:
    """Divide a sessao em faixas e cria uma dummy por faixa. E' a forma mais
    simples de dar ao modelo linear a informacao "onde estamos no dia"."""
    buckets = np.clip((frame["session_position"] * N_SESSION_BUCKETS).astype(int),
                      0, N_SESSION_BUCKETS - 1)
    names = []
    for i in range(N_SESSION_BUCKETS):
        name = f"session_bucket_{i}"
        frame[name] = (buckets == i).astype(float)
        names.append(name)
    return names


def run(conn, timeframe: str = "1h", target: str = "abs_return") -> dict | None:
    frame = build_frame(conn, timeframe)
    if frame.empty:
        return None

    session_columns = add_session_dummies(frame)
    har_columns = [c for c in frame.columns if c.startswith("har_")]

    train_df, validation_df, test_df = tree_train.split_by_time(
        frame, config.TREES["horizon_bars"])
    fit_df = pd.concat([train_df, validation_df])
    realised = test_df[target].values

    full_features = [c for c in frame.columns
                     if c not in ("ticker", "ts", "big_move", "magnitude_ratio",
                                  "abs_return", "band") and not c.startswith("session_bucket_")]

    def evaluate(predictions, label):
        deciles = M.decile_lift(predictions, realised)
        return {
            "name": label,
            "spearman": float(stats.spearmanr(predictions, realised).statistic),
            "top_bottom": (deciles[-1]["mean_realised"] / deciles[0]["mean_realised"]
                           if deciles[0]["mean_realised"] else None),
        }

    def ridge(columns, label):
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(fit_df[columns], fit_df[target])
        return evaluate(model.predict(test_df[columns]), label)

    def tree(columns, label):
        model = XGBRegressor(**{k: v for k, v in config.TREES["xgb_params"].items()
                                if k not in ("objective", "num_class")},
                             max_depth=4, objective="reg:squarederror")
        model.fit(fit_df[columns], fit_df[target])
        return evaluate(model.predict(test_df[columns]), label)

    results = [
        ridge(har_columns, "HAR (so volatilidade)"),
        ridge(har_columns + session_columns, "HAR + hora da sessao"),
        tree(har_columns + session_columns, "XGBoost (so HAR + hora)"),
        tree(full_features, "XGBoost (40 features)"),
    ]

    return {"timeframe": timeframe, "target": target, "results": results,
            "n_test": len(test_df),
            "effective_days": int(pd.Series(test_df["ts"]).dt.normalize().nunique())}


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection, "1h", "abs_return")
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print(f"{'='*72}")
    print(f"EXPERIENCIA C -- de onde vem a vantagem do XGBoost sobre o HAR?")
    print(f"  @ {outcome['timeframe']}, alvo {outcome['target']}, "
          f"{outcome['n_test']:,} linhas de teste, {outcome['effective_days']} dias")
    print()
    print(f"  {'modelo':>28} {'spearman':>10} {'decis':>9}")
    baseline = outcome["results"][0]["spearman"]
    best = outcome["results"][-1]["spearman"]
    for r in outcome["results"]:
        share = ((r["spearman"] - baseline) / (best - baseline) * 100
                 if best != baseline else 0.0)
        print(f"  {r['name']:>28} {r['spearman']:>10.4f} {(r['top_bottom'] or 0):>8.2f}x"
              f"   {share:>5.0f}% do fosso fechado")

    with_clock = outcome["results"][1]["spearman"]
    closed = (with_clock - baseline) / (best - baseline) * 100 if best != baseline else 0
    print()
    print(f"  So' dar um RELOGIO ao HAR fecha {closed:.0f}% do fosso.")
    if closed > 60:
        print("  -> a vantagem do XGBoost e' sobretudo SAZONALIDADE INTRADIARIA,")
        print("     que o HAR nao modela nesta frequencia. Nao e' 'ML bate econometria'.")
    elif closed > 25:
        print("  -> parte substancial da vantagem e' sazonalidade; o resto e' outra coisa.")
    else:
        print("  -> a sazonalidade explica pouco; a vantagem vem mesmo das outras features.")
    connection.close()
