"""EXPERIENCIA F -- o sinal sobrevive ao horizonte DIARIO, onde ha opcoes?

E' o teste de implementabilidade. A ronda 2 mostrou um efeito real (surpresa
1.241 no decil de topo, IC95% inteiramente acima do equilibrio) sem instrumento
para o explorar: nao existem opcoes a uma hora. A prazo diario existem.

DUAS RAZOES PARA ESPERAR UM RESULTADO PIOR, e ambas sao boas razoes:

1. Desaparece o relogio. A 1h, 74% do poder preditivo vinha das dummies de
   hora da sessao. Em barras diarias cada dia e' UMA barra -- nao ha
   sazonalidade intradiaria para explorar. O que sobrar e' sinal a serio.

2. O HAR-RV joga em casa. Foi desenhado para dados diarios; a desvantagem
   estrutural que tinha a 1h (nao ter relogio) desaparece aqui.

Se o XGBoost continuar a bater o HAR a prazo diario, e' um resultado bem mais
forte do que o de 1h. Se nao bater, a conclusao e' que a vantagem toda era
sazonalidade intradiaria -- e' um resultado igualmente util, so' menos
agradavel.

Os dados diarios sao construidos agregando as barras de 1h dos 160 tickers,
o que da' muito mais historico do que a tabela `prices` original (8 tickers).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.patterns.ingest_intraday import load_bars
from src.research.har_benchmark import HAR_WINDOWS
from src.trees import features as F

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

# Escalas do HAR original de Corsi: diaria, semanal, mensal.
DAILY_HAR_WINDOWS = {"short": 1, "week": 5, "month": 22, "quarter": 66}
ROLLING_WINDOW = 60          # dias para estimar o nivel corrente de volatilidade
VOLATILITY_RISK_PREMIUM = 0.11
OPTION_COST_RATIOS = (0.01, 0.03, 0.05)


def to_daily(bars: pd.DataFrame) -> pd.DataFrame:
    """Agrega barras intradiarias em barras diarias."""
    grouped = bars.groupby(bars.index.normalize())
    daily = pd.DataFrame({
        "open": grouped["open"].first(),
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close": grouped["close"].last(),
        "volume": grouped["volume"].sum(),
    })
    return daily.dropna()


def daily_har_features(bars: pd.DataFrame) -> pd.DataFrame:
    absolute_return = bars["close"].pct_change().abs()
    out = pd.DataFrame(index=bars.index)
    for name, window in DAILY_HAR_WINDOWS.items():
        out[f"har_{name}"] = absolute_return.rolling(window, min_periods=window).mean()
    return out


def weekday_dummies(bars: pd.DataFrame) -> pd.DataFrame:
    """O analogo diario do relogio intradiario. Se houver efeito de calendario
    a este prazo, e' aqui que aparece."""
    out = pd.DataFrame(index=bars.index)
    weekday = bars.index.dayofweek
    for day in range(5):
        out[f"weekday_{day}"] = (weekday == day).astype(float)
    return out


def build(conn, tickers=None) -> pd.DataFrame:
    tickers = tickers or config.TREE_TICKERS
    benchmark_daily = to_daily(load_bars(conn, config.BENCHMARK["ticker"], "1h"))

    frames = []
    for ticker in tickers:
        intraday = load_bars(conn, ticker, "1h")
        if intraday.empty:
            continue
        bars = to_daily(intraday)
        if len(bars) < 250:
            continue

        matrix = (F.indicator_features(bars)
                  .join(F.market_context_features(bars, benchmark_daily))
                  .join(daily_har_features(bars))
                  .join(weekday_dummies(bars))
                  .join(F.label_magnitude(bars)))
        # session_position e' constante em barras diarias (um bar por dia).
        matrix = matrix.drop(columns=["session_position"], errors="ignore")
        matrix["ticker"] = ticker
        matrix["ts"] = matrix.index
        matrix.index.name = None
        frames.append(matrix.dropna())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values(["ts", "ticker"]).reset_index(drop=True)


def split(frame: pd.DataFrame, horizon: int = 1):
    stamps = np.sort(frame["ts"].unique())
    first = stamps[int(len(stamps) * 0.6)]
    second = stamps[int(len(stamps) * 0.8)]
    step = pd.Series(stamps).diff().median()
    embargo = step * horizon if pd.notna(step) else pd.Timedelta(0)
    return (frame[frame["ts"] < first - embargo],
            frame[(frame["ts"] >= first) & (frame["ts"] < second - embargo)],
            frame[frame["ts"] >= second])


def run(conn) -> dict | None:
    frame = build(conn)
    if frame.empty:
        return None

    train_df, validation_df, test_df = split(frame)
    if test_df.empty:
        return None
    fit_df = pd.concat([train_df, validation_df])

    har_columns = [c for c in frame.columns if c.startswith("har_")]
    weekday_columns = [c for c in frame.columns if c.startswith("weekday_")]
    full_features = [c for c in frame.columns
                     if c not in ("ticker", "ts", "big_move", "magnitude_ratio",
                                  "abs_return", "band")]

    realised = test_df["abs_return"].values

    def spearman_of(predictions):
        return float(stats.spearmanr(predictions, realised).statistic)

    def ridge(columns):
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(fit_df[columns], fit_df["abs_return"])
        return spearman_of(model.predict(test_df[columns]))

    def tree(columns):
        model = XGBRegressor(**{k: v for k, v in config.TREES["xgb_params"].items()
                                if k not in ("objective", "num_class")},
                             max_depth=4, objective="reg:squarederror")
        model.fit(fit_df[columns], fit_df["abs_return"])
        return spearman_of(model.predict(test_df[columns]))

    comparison = [
        ("HAR-RV (4 medias)", ridge(har_columns)),
        ("HAR + dia da semana", ridge(har_columns + weekday_columns)),
        ("XGBoost (todas as features)", tree(full_features)),
    ]

    # ---- analise de surpresa e custos, igual a' da ronda 2 ----
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

    fit_clean = fit_df[np.isfinite(fit_df["surprise"])]
    test_clean = test_df[np.isfinite(test_df["surprise"])].copy()

    surprise_model = XGBRegressor(**{k: v for k, v in config.TREES["xgb_params"].items()
                                     if k not in ("objective", "num_class")},
                                  max_depth=4, objective="reg:squarederror")
    surprise_model.fit(fit_clean[full_features], fit_clean["surprise"])
    test_clean["predicted"] = surprise_model.predict(test_clean[full_features])
    test_clean["decile"] = pd.qcut(test_clean["predicted"], 10, labels=False, duplicates="drop")

    by_decile = test_clean.groupby("decile").agg(
        predicted=("predicted", "mean"), realised=("surprise", "mean"),
        n=("surprise", "size")).reset_index()
    top = test_clean[test_clean["decile"] == by_decile["decile"].max()]

    # Bootstrap por dia, com codigos inteiros (comparar datetime64 directamente
    # produzia grupos vazios e IC a NaN em versoes anteriores destas experiencias)
    codes = pd.factorize(pd.to_datetime(top["ts"]).dt.normalize())[0]
    values = top["surprise"].values
    groups = [np.flatnonzero(codes == c) for c in range(codes.max() + 1)]
    groups = [g for g in groups if len(g)]
    rng = np.random.default_rng(13)
    boot = [values[np.concatenate([groups[i] for i in
                                   rng.integers(0, len(groups), len(groups))])].mean()
            for _ in range(500)]
    low, high = np.percentile(boot, [2.5, 97.5])

    break_even = 1 + VOLATILITY_RISK_PREMIUM
    return {
        "n_test": len(test_clean),
        "effective_days": int(pd.Series(test_clean["ts"]).dt.normalize().nunique()),
        "n_tickers": int(test_clean["ticker"].nunique()),
        "comparison": comparison,
        "surprise_spearman": float(stats.spearmanr(
            test_clean["predicted"], test_clean["surprise"]).statistic),
        "mean_surprise": float(test_clean["surprise"].mean()),
        "deciles": by_decile.to_dict("records"),
        "top_mean": float(values.mean()), "ci": (float(low), float(high)),
        "break_even": break_even,
        "beats_break_even": low > break_even,
        "scenarios": [{"cost": c, "net": float(values.mean() - break_even - c),
                       "profitable": values.mean() - break_even - c > 0}
                      for c in OPTION_COST_RATIOS],
    }


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection)
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print("=" * 76)
    print("EXPERIENCIA F -- horizonte DIARIO (onde existem opcoes)")
    print(f"  {outcome['n_test']:,} linhas | {outcome['n_tickers']} tickers | "
          f"{outcome['effective_days']} dias efectivos no teste")
    print()
    print("  Previsao do NIVEL do movimento -- agora sem relogio intradiario:")
    print(f"    {'modelo':>30} {'spearman':>10}")
    for name, value in outcome["comparison"]:
        print(f"    {name:>30} {value:>10.4f}")
    har = outcome["comparison"][0][1]
    tree_score = outcome["comparison"][2][1]
    print(f"    -> XGBoost - HAR = {tree_score - har:+.4f}", end="  ")
    print("(XGBoost ganha)" if tree_score - har > 0.02 else
          ("empate tecnico" if abs(tree_score - har) <= 0.02 else "HAR GANHA"))

    print()
    print(f"  Sinal de SURPRESA a prazo diario: Spearman {outcome['surprise_spearman']:.4f}")
    print(f"  Calibracao: surpresa media = {outcome['mean_surprise']:.3f} (tem de ser ~1.00)")
    print(f"    {'decil':>6} {'previsto':>10} {'realizado':>11} {'n':>8}")
    for d in outcome["deciles"]:
        print(f"    {int(d['decile']) + 1:>6} {d['predicted']:>10.3f} "
              f"{d['realised']:>11.3f} {int(d['n']):>8,}")

    print()
    print(f"  Decil de topo: surpresa {outcome['top_mean']:.4f}  "
          f"IC95% [{outcome['ci'][0]:.4f}, {outcome['ci'][1]:.4f}]")
    print(f"  Equilibrio (1 + premio de risco) = {outcome['break_even']:.4f}")
    print(f"  -> {'BATE o equilibrio' if outcome['beats_break_even'] else 'NAO BATE'}")
    print()
    print("  Cenarios de custo em opcoes:")
    for s in outcome["scenarios"]:
        print(f"    custo {s['cost']:>5.0%}  liquido {s['net']:>+8.4f}  "
              f"{'LUCRATIVO' if s['profitable'] else 'perde'}")
    connection.close()
