"""Previsao de DIRECCAO depois de um padrao -- alvo binario.

Pergunta diferente da do sequence.py. Em vez de "que padrao vem a seguir"
(20 classes, sem vantagem mensuravel sobre saber as frequencias), pergunta-se
"o preco sobe ou desce nas H barras a seguir a este padrao". Duas classes,
quase equilibradas, e os mesmos ~3400 exemplos deixam de se diluir por 20
classes.

TRES COISAS QUE FAZEM A DIFERENCA ENTRE ISTO E UM BACKTEST QUE SE ENGANA:

1. O preco de referencia e' o fecho em CONFIRMED_TS, nao no fim do padrao.
   O padrao acaba em end_ts, mas so' se SABE que existe pivot_window barras
   depois, quando o ultimo pivot fica confirmado. Usar o preco do fim do
   padrao e' lookahead e inflaciona tudo.

2. Amostra efectiva conta-se em DIAS, nao em padroes. Os 43 tickers movem-se
   com o mercado no mesmo dia, portanto as observacoes do mesmo dia nao sao
   independentes. Todos os erros-padrao aqui usam o numero de dias distintos.

3. Nao basta bater a classe maioritaria. Mede-se tambem contra instantes
   ALEATORIOS que nao sao fim de padrao nenhum -- se um momento-padrao nao for
   diferente de um momento qualquer, o padrao nao esta' a dizer nada.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.patterns import context
from src.patterns.ingest_intraday import load_bars

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

D = config.PATTERN_DIRECTION


# --------------------------------------------------------------------------
# Rotulos
# --------------------------------------------------------------------------

def _benchmark_returns(conn, timeframe: str):
    """Serie de fechos do indice, indexada por timestamp, para o rotulo
    neutro ao mercado."""
    bars = load_bars(conn, config.BENCHMARK["ticker"], timeframe)
    if bars.empty:
        return None, None
    return bars["close"].values, {ts: i for i, ts in enumerate(bars.index)}


def build_outcomes(conn, timeframe: str, horizons=None, tickers=None) -> int:
    horizons = horizons or D["horizons_bars"]
    tickers = tickers or config.PATTERN_TICKERS
    benchmark_closes, benchmark_pos = _benchmark_returns(conn, timeframe)
    now = datetime.now(timezone.utc).isoformat()

    rows = []
    for ticker in tickers:
        bars = load_bars(conn, ticker, timeframe)
        if bars.empty:
            continue
        closes = bars["close"].values
        position_of = {ts: i for i, ts in enumerate(bars.index)}

        patterns = conn.execute(
            """SELECT pattern_type, confirmed_ts FROM detected_patterns
               WHERE ticker = ? AND timeframe = ? ORDER BY confirmed_ts""",
            (ticker, timeframe),
        ).fetchall()

        for pattern in patterns:
            timestamp = pd.Timestamp(pattern["confirmed_ts"])
            entry = position_of.get(timestamp)
            if entry is None:
                continue
            for horizon in horizons:
                exit_index = entry + horizon
                if exit_index >= len(closes):
                    continue
                ref_close, target_close = float(closes[entry]), float(closes[exit_index])
                forward_return = target_close / ref_close - 1

                benchmark_return = excess_return = excess_direction = None
                if benchmark_pos is not None:
                    b_entry = benchmark_pos.get(timestamp)
                    if b_entry is not None and b_entry + horizon < len(benchmark_closes):
                        benchmark_return = float(
                            benchmark_closes[b_entry + horizon] / benchmark_closes[b_entry] - 1
                        )
                        excess_return = forward_return - benchmark_return
                        excess_direction = 1 if excess_return > 0 else 0

                rows.append((
                    ticker, timeframe, pattern["confirmed_ts"], horizon, pattern["pattern_type"],
                    ref_close, target_close, forward_return, benchmark_return, excess_return,
                    1 if forward_return > 0 else 0, excess_direction, now,
                ))

    conn.execute("DELETE FROM pattern_outcomes WHERE timeframe = ?", (timeframe,))
    conn.executemany(
        """INSERT INTO pattern_outcomes
             (ticker, timeframe, confirmed_ts, horizon_bars, pattern_type, ref_close,
              target_close, forward_return, benchmark_return, excess_return,
              direction, excess_direction, computed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


# --------------------------------------------------------------------------
# Controlo: instantes aleatorios que NAO sao fim de padrao
# --------------------------------------------------------------------------

def random_control(conn, timeframe: str, horizon: int, seed: int = 42) -> dict:
    """Mesma medicao em momentos escolhidos ao acaso. Se um momento-padrao nao
    for diferente de um momento qualquer, o padrao nao esta' a dizer nada."""
    rng = np.random.default_rng(seed)
    benchmark_closes, benchmark_pos = _benchmark_returns(conn, timeframe)
    per_ticker = max(1, D["n_random_controls"] // len(config.PATTERN_TICKERS))

    returns, excess = [], []
    for ticker in config.PATTERN_TICKERS:
        bars = load_bars(conn, ticker, timeframe)
        if bars.empty or len(bars) < horizon + 10:
            continue
        closes = bars["close"].values
        occupied = set()
        for row in conn.execute(
            "SELECT confirmed_ts FROM detected_patterns WHERE ticker = ? AND timeframe = ?",
            (ticker, timeframe),
        ).fetchall():
            index = {ts: i for i, ts in enumerate(bars.index)}.get(pd.Timestamp(row["confirmed_ts"]))
            if index is not None:
                occupied.update(range(max(0, index - 2), index + 3))

        candidates = [i for i in range(len(closes) - horizon) if i not in occupied]
        if not candidates:
            continue
        for index in rng.choice(candidates, size=min(per_ticker, len(candidates)), replace=False):
            index = int(index)
            forward = closes[index + horizon] / closes[index] - 1
            returns.append(forward)
            if benchmark_pos is not None:
                b = benchmark_pos.get(bars.index[index])
                if b is not None and b + horizon < len(benchmark_closes):
                    excess.append(forward - (benchmark_closes[b + horizon] / benchmark_closes[b] - 1))

    returns = np.array(returns)
    excess = np.array(excess)
    return {
        "n": len(returns),
        "up_rate": float((returns > 0).mean()) if len(returns) else None,
        "mean_return": float(returns.mean()) if len(returns) else None,
        "excess_up_rate": float((excess > 0).mean()) if len(excess) else None,
        "excess_mean_return": float(excess.mean()) if len(excess) else None,
    }


# --------------------------------------------------------------------------
# Dataset + split com embargo
# --------------------------------------------------------------------------

def build_dataset(conn, timeframe: str, horizon: int, market_neutral: bool) -> pd.DataFrame:
    label_column = "excess_direction" if market_neutral else "direction"
    return_column = "excess_return" if market_neutral else "forward_return"

    records = []
    for ticker in config.PATTERN_TICKERS:
        bars = load_bars(conn, ticker, timeframe)
        if bars.empty:
            continue
        position_of = {ts: i for i, ts in enumerate(bars.index)}

        rows = conn.execute(
            f"""SELECT o.confirmed_ts, o.{label_column} AS label, o.{return_column} AS ret,
                       d.pattern_type, d.start_ts, d.end_ts, d.quality, d.meta_json
                FROM pattern_outcomes o
                JOIN detected_patterns d
                  ON d.ticker = o.ticker AND d.timeframe = o.timeframe
                 AND d.confirmed_ts = o.confirmed_ts AND d.pattern_type = o.pattern_type
                WHERE o.ticker = ? AND o.timeframe = ? AND o.horizon_bars = ?
                  AND o.{label_column} IS NOT NULL
                ORDER BY o.confirmed_ts""",
            (ticker, timeframe, horizon),
        ).fetchall()

        for row in rows:
            start_idx = position_of.get(pd.Timestamp(row["start_ts"]))
            end_idx = position_of.get(pd.Timestamp(row["end_ts"]))
            confirm_idx = position_of.get(pd.Timestamp(row["confirmed_ts"]))
            if start_idx is None or end_idx is None or confirm_idx is None:
                continue
            meta = json.loads(row["meta_json"] or "{}")
            features = context.pattern_context_features(
                bars, start_idx, end_idx, row["pattern_type"], row["quality"], meta.get("n_pivots", 3)
            )
            features.update({
                "ticker": ticker, "confirmed_ts": row["confirmed_ts"],
                "confirm_idx": confirm_idx, "label": int(row["label"]),
                "forward_return": float(row["ret"]),
                "day": pd.Timestamp(row["confirmed_ts"]).normalize(),
            })
            records.append(features)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["confirmed_ts", "ticker"]).reset_index(drop=True)


def split_with_embargo(frame: pd.DataFrame, horizon: int):
    """Split cronologico por ticker, descartando as observacoes cuja janela
    futura atravessa a fronteira para a particao seguinte.

    Sem este embargo, um padrao no fim do treino tem o seu resultado a
    acontecer ja' dentro do periodo de validacao -- treino e teste passam a
    partilhar as mesmas barras futuras e a avaliacao fica optimista."""
    train, validation, test = [], [], []
    for _, group in frame.groupby("ticker", sort=False):
        group = group.sort_values("confirm_idx")
        n = len(group)
        first = int(n * D["train_fraction"])
        second = int(n * (D["train_fraction"] + D["validation_fraction"]))
        if first == 0 or second <= first or second >= n:
            continue

        train_part, validation_part, test_part = group.iloc[:first], group.iloc[first:second], group.iloc[second:]
        validation_start = validation_part["confirm_idx"].min()
        test_start = test_part["confirm_idx"].min()

        train.append(train_part[train_part["confirm_idx"] + horizon < validation_start])
        validation.append(validation_part[validation_part["confirm_idx"] + horizon < test_start])
        test.append(test_part)

    def combine(parts):
        return pd.concat(parts).sort_values("confirmed_ts").reset_index(drop=True) if parts else pd.DataFrame()

    return combine(train), combine(validation), combine(test)


# --------------------------------------------------------------------------
# Treino e avaliacao
# --------------------------------------------------------------------------

def _candidates() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            min_samples_leaf=40, l2_regularization=1.0, random_state=42,
        ),
    }


def evaluate_horizon(conn, timeframe: str, horizon: int, market_neutral: bool) -> dict | None:
    frame = build_dataset(conn, timeframe, horizon, market_neutral)
    if frame.empty:
        return None

    train_df, validation_df, test_df = split_with_embargo(frame, horizon)
    if len(train_df) < D["min_train_rows"] or validation_df.empty or test_df.empty:
        return None
    if train_df["label"].nunique() < 2:
        return None

    features = context.FEATURE_NAMES
    X_train, y_train = train_df[features], train_df["label"]

    scored = {}
    for name, estimator in _candidates().items():
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(validation_df[features])[:, 1]
        scored[name] = roc_auc_score(validation_df["label"], probabilities) \
            if validation_df["label"].nunique() > 1 else 0.5

    best_name = max(scored, key=scored.get)
    model = _candidates()[best_name]
    model.fit(X_train, y_train)

    y_test = test_df["label"].values
    probabilities = model.predict_proba(test_df[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = float((predictions == y_test).mean())
    majority = int(y_train.mode().iloc[0])
    baseline = float((y_test == majority).mean())

    # Erro-padrao sobre DIAS distintos, nao sobre padroes: os 43 tickers
    # movem-se juntos, observacoes do mesmo dia nao sao independentes.
    effective_n = max(test_df["day"].nunique(), 1)
    standard_error = float(np.sqrt(0.25 / effective_n))

    returns = test_df["forward_return"].values
    return {
        "timeframe": timeframe, "horizon_bars": horizon, "market_neutral": market_neutral,
        "algorithm": best_name, "validation_auc": scored[best_name],
        "n_train": len(train_df), "n_validation": len(validation_df), "n_test": len(test_df),
        "n_effective_days": effective_n,
        "accuracy": accuracy, "baseline_majority": baseline,
        "auc": float(roc_auc_score(y_test, probabilities)) if len(set(y_test)) > 1 else None,
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "mean_return_when_up": float(returns[predictions == 1].mean()) if (predictions == 1).any() else None,
        "mean_return_when_down": float(returns[predictions == 0].mean()) if (predictions == 0).any() else None,
        "standard_error": standard_error,
        "significant": abs(accuracy - baseline) > 2 * standard_error,
    }


def run(conn, timeframe: str = "1h") -> list:
    results = []
    for market_neutral in (False, True):
        for horizon in D["horizons_bars"]:
            outcome = evaluate_horizon(conn, timeframe, horizon, market_neutral)
            if outcome:
                outcome["random_control"] = random_control(conn, timeframe, horizon)
                results.append(outcome)
    return results


def store(conn, results: list) -> int:
    now = datetime.now(timezone.utc).isoformat()
    # O horizonte "usado" escolhe-se pela AUC de VALIDACAO, nunca pelo teste.
    best = max(results, key=lambda r: r["validation_auc"]) if results else None
    rows = []
    for r in results:
        control = r.get("random_control", {})
        control_rate = control.get("excess_up_rate" if r["market_neutral"] else "up_rate")
        rows.append((
            f"dir_{r['timeframe']}_h{r['horizon_bars']}_{'neutral' if r['market_neutral'] else 'raw'}"
            f"_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
            r["timeframe"], r["horizon_bars"], int(r["market_neutral"]), r["algorithm"], now,
            r["n_train"], r["n_validation"], r["n_test"], r["n_effective_days"],
            r["accuracy"], r["auc"], r["log_loss"], r["baseline_majority"], control_rate,
            r["mean_return_when_up"], r["mean_return_when_down"],
            r["standard_error"], int(r["significant"]),
            int(r is best), f"validation_auc={r['validation_auc']:.4f}",
        ))
    conn.executemany(
        """INSERT INTO pattern_direction_models
             (version, timeframe, horizon_bars, market_neutral, algorithm, trained_at,
              n_train, n_validation, n_test, n_effective_days, accuracy, auc, log_loss,
              baseline_majority, random_control_accuracy, mean_return_when_up,
              mean_return_when_down, standard_error, significant, selected, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


if __name__ == "__main__":
    connection = database.get_connection()
    for tf in D["timeframes"]:
        n = build_outcomes(connection, tf)
        print(f"{tf}: {n} rotulos de direccao gravados\n")

        results = run(connection, tf)
        store(connection, results)

        for neutral in (False, True):
            subset = [r for r in results if r["market_neutral"] is neutral]
            if not subset:
                continue
            print(f"--- {'NEUTRO AO MERCADO (vs SPY)' if neutral else 'RETORNO BRUTO'} ---")
            print(f"    {'H':>3s} {'acerto':>7s} {'baseline':>9s} {'AUC':>6s} {'controlo':>9s} "
                  f"{'n_dias':>7s} {'+-':>6s}  veredicto")
            for r in sorted(subset, key=lambda x: x["horizon_bars"]):
                control = r.get("random_control", {})
                control_rate = control.get("excess_up_rate" if neutral else "up_rate") or 0
                verdict = "SIGNIFICATIVO" if r["significant"] else "dentro do ruido"
                print(f"    {r['horizon_bars']:>3d} {r['accuracy']:>6.1%} {r['baseline_majority']:>9.1%} "
                      f"{(r['auc'] or 0):>6.3f} {control_rate:>8.1%} {r['n_effective_days']:>7d} "
                      f"{r['standard_error']*100:>5.1f}pp  {verdict}")
            print()
    connection.close()
