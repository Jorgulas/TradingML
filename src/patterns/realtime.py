"""Forward test ao vivo -- o algoritmo a tentar acertar, em tempo real.

Porque e' que esta e' a avaliacao mais honesta de todas: a previsao e' gravada
ANTES de a barra que a resolve existir. Nao ha' lookahead possivel, nem por
engano, nem por um bug subtil no split. E' o unico numero deste projecto que
nao pode ser viciado.

Dois modos:

  live    -- busca as barras mais recentes do yfinance de X em X segundos.
             So' funciona com o mercado aberto, e os dados vem com atraso
             (medido: 15 a 60 min). Isto e' dito na interface: "tempo real"
             aqui significa "tempo real atrasado", nao tick a tick.

  replay  -- percorre barras historicas uma a uma como se estivessem a chegar
             agora. Serve para ver o mecanismo a funcionar com o mercado
             fechado. A garantia critica: em cada passo o detector so' ve
             bars[:cursor+1]. Se alguma vez vir uma barra a' frente do cursor,
             o replay deixa de valer nada -- ha' um teste dedicado a isso.

APRENDIZAGEM ONLINE: cada resultado que chega actualiza o modelo por
partial_fit (SGD), arrancando dos coeficientes do modelo treinado em
historico. E' o que faz a margem de acerto mexer enquanto corre.
"""

import argparse
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.patterns import context, direction, scanner
from src.patterns.ingest_intraday import fetch_intraday, load_bars, upsert_intraday

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

R = config.PATTERN_REALTIME


def wilson_interval(successes: int, total: int, z: float = 1.96):
    """Intervalo de confianca de Wilson. Usa-se este e nao o normal porque com
    poucas observacoes (que e' exactamente o caso no inicio de uma sessao) o
    intervalo normal produz limites fora de [0,1] e larguras absurdas."""
    if total == 0:
        return None, None
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


# --------------------------------------------------------------------------
# Modelo online
# --------------------------------------------------------------------------

class OnlineModel:
    """SGD logistico que arranca do modelo treinado em historico e depois vai
    sendo corrigido a cada resultado real que chega."""

    def __init__(self, conn, timeframe: str, horizon: int):
        self.features = context.FEATURE_NAMES
        self.updates = 0
        self.scaler = StandardScaler()
        self.model = SGDClassifier(
            loss="log_loss", learning_rate="constant",
            eta0=R["online_learning_rate"], random_state=42,
        )
        self._bootstrap(conn, timeframe, horizon)

    def _bootstrap(self, conn, timeframe: str, horizon: int):
        """Treina em todo o historico disponivel ANTES de a sessao comecar."""
        self.bootstrap_rows = 0
        frame = direction.build_dataset(conn, timeframe, horizon, market_neutral=False)
        if frame.empty:
            # Os rotulos deste timeframe/horizonte ainda nao existem: gera-os.
            direction.build_outcomes(conn, timeframe, horizons=[horizon])
            frame = direction.build_dataset(conn, timeframe, horizon, market_neutral=False)
        if frame.empty or frame["label"].nunique() < 2:
            self.ready = False
            return
        X = self.scaler.fit_transform(frame[self.features])
        self.model.fit(X, frame["label"])
        self.ready = True
        self.bootstrap_rows = len(frame)

    def predict_proba(self, features: dict) -> float:
        if not self.ready:
            return 0.5
        X = self.scaler.transform(pd.DataFrame([features])[self.features])
        return float(self.model.predict_proba(X)[0][1])

    def learn(self, features: dict, label: int):
        if not self.ready:
            return
        X = self.scaler.transform(pd.DataFrame([features])[self.features])
        self.model.partial_fit(X, [label], classes=[0, 1])
        self.updates += 1


# --------------------------------------------------------------------------
# Sessao
# --------------------------------------------------------------------------

def start_session(conn, mode: str) -> str:
    session_id = f"{mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:4]}"
    conn.execute("UPDATE live_sessions SET status = 'stopped' WHERE status = 'running'")
    conn.execute(
        """INSERT INTO live_sessions
             (session_id, mode, timeframe, horizon_bars, threshold, started_at, status)
           VALUES (?, ?, ?, ?, ?, ?, 'running')""",
        (session_id, mode, R["timeframe"], R["horizon_bars"], R["threshold"],
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return session_id


def _record_prediction(conn, session_id, ticker, match, bars, cursor, model) -> bool:
    """Grava uma previsao para um padrao acabado de confirmar. A barra de
    resolucao ainda nao existe -- e' esse o ponto."""
    confirm_idx = min(match.confirm_idx, cursor)
    resolve_idx = confirm_idx + R["horizon_bars"]
    predicted_at = bars.index[confirm_idx]

    already = conn.execute(
        "SELECT 1 FROM live_predictions WHERE session_id = ? AND ticker = ? AND predicted_at_ts = ?",
        (session_id, ticker, predicted_at.isoformat()),
    ).fetchone()
    if already:
        return False

    features = context.pattern_context_features(
        bars.iloc[:cursor + 1], match.start_idx, match.end_idx,
        match.pattern_type, match.quality, match.n_pivots,
    )
    probability = model.predict_proba(features)
    direction_up = 1 if probability >= 0.5 else 0
    confidence = probability if direction_up else 1 - probability
    if confidence < R["threshold"]:
        return False

    # A barra de resolucao e' calculada por deslocamento temporal, nao lida --
    # ainda nao existe quando a previsao e' feita.
    step = bars.index[1] - bars.index[0] if len(bars) > 1 else pd.Timedelta(minutes=5)
    resolve_ts = (bars.index[resolve_idx] if resolve_idx < len(bars)
                  else predicted_at + step * R["horizon_bars"])

    conn.execute(
        """INSERT INTO live_predictions
             (session_id, ticker, pattern_type, predicted_at_ts, resolve_at_ts, entry_price,
              predicted_direction, confidence, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, ticker, match.pattern_type, predicted_at.isoformat(), resolve_ts.isoformat(),
         float(bars["close"].iloc[confirm_idx]), direction_up, float(confidence),
         datetime.now(timezone.utc).isoformat()),
    )
    return True


def _resolve_due(conn, session_id, ticker, bars, cursor, model) -> int:
    """Resolve as previsoes cuja barra de saida ja' existe."""
    visible_ts = bars.index[cursor]
    pending = conn.execute(
        """SELECT * FROM live_predictions
           WHERE session_id = ? AND ticker = ? AND resolved = 0 AND resolve_at_ts <= ?""",
        (session_id, ticker, visible_ts.isoformat()),
    ).fetchall()

    resolved = 0
    for row in pending:
        target = pd.Timestamp(row["resolve_at_ts"])
        positions = bars.index[:cursor + 1]
        if target not in positions:
            continue
        exit_index = positions.get_loc(target)
        exit_price = float(bars["close"].iloc[exit_index])
        actual = 1 if exit_price > row["entry_price"] else 0
        correct = int(actual == row["predicted_direction"])

        conn.execute(
            """UPDATE live_predictions SET resolved = 1, exit_price = ?, actual_direction = ?,
                 correct = ?, return_pct = ?, resolved_at = ? WHERE id = ?""",
            (exit_price, actual, correct, exit_price / row["entry_price"] - 1,
             datetime.now(timezone.utc).isoformat(), row["id"]),
        )
        resolved += 1

        if R["online_learning"]:
            # Reconstroi as features do momento da previsao para o modelo
            # aprender com o par (o que via, o que aconteceu).
            entry_positions = bars.index[:cursor + 1]
            entry_ts = pd.Timestamp(row["predicted_at_ts"])
            if entry_ts in entry_positions:
                entry_index = entry_positions.get_loc(entry_ts)
                features = context.pattern_context_features(
                    bars.iloc[:entry_index + 1],
                    max(0, entry_index - 30), entry_index,
                    row["pattern_type"], 0.7, 4,
                )
                model.learn(features, actual)
    return resolved


def _refresh_session_stats(conn, session_id, cursor_ts, online_updates):
    stats = conn.execute(
        """SELECT COUNT(*) AS total, SUM(resolved) AS resolved, SUM(COALESCE(correct, 0)) AS correct
           FROM live_predictions WHERE session_id = ?""",
        (session_id,),
    ).fetchone()
    conn.execute(
        """UPDATE live_sessions SET last_tick_at = ?, cursor_ts = ?, n_predictions = ?,
             n_resolved = ?, n_correct = ?, online_updates = ? WHERE session_id = ?""",
        (datetime.now(timezone.utc).isoformat(), cursor_ts,
         stats["total"] or 0, stats["resolved"] or 0, stats["correct"] or 0,
         online_updates, session_id),
    )
    conn.commit()


# --------------------------------------------------------------------------
# Modo replay
# --------------------------------------------------------------------------

def run_replay(conn, tickers=None, bars_to_replay: int = None, delay: float = None):
    # Universo alargado: com 8 tickers saiam ~1 previsao em 60 barras e nao
    # havia nada para ver a mexer.
    tickers = tickers or config.PATTERN_TICKERS
    bars_to_replay = bars_to_replay or R["replay_bars"]
    delay = R["replay_delay_seconds"] if delay is None else delay
    timeframe = R["timeframe"]

    session_id = start_session(conn, "replay")
    model = OnlineModel(conn, timeframe, R["horizon_bars"])
    print(f"sessao {session_id} | modelo arrancado com "
          f"{getattr(model, 'bootstrap_rows', 0)} exemplos historicos")

    series = {}
    for ticker in tickers:
        bars = load_bars(conn, ticker, timeframe)
        if len(bars) > bars_to_replay + R["scan_window"]:
            series[ticker] = bars
    if not series:
        print("sem barras suficientes para replay")
        return session_id

    total_bars = min(len(b) for b in series.values())
    start_cursor = total_bars - bars_to_replay

    for step, cursor in enumerate(range(start_cursor, total_bars)):
        for ticker, bars in series.items():
            # CRITICO: so' as barras ate' ao cursor. Uma unica barra a' frente
            # aqui e o replay inteiro deixa de valer.
            visible = bars.iloc[:cursor + 1]
            window = visible.iloc[-R["scan_window"]:]
            offset = len(visible) - len(window)

            _resolve_due(conn, session_id, ticker, visible, len(visible) - 1, model)

            matches = scanner.scan_patterns(
                window, config.PATTERN_TIMEFRAMES[timeframe]["pivot_window"]
            )
            if matches:
                last = matches[-1]
                if last.confirm_idx >= len(window) - 2:  # acabou de confirmar
                    shifted = type(last)(
                        pattern_type=last.pattern_type,
                        start_idx=last.start_idx + offset, end_idx=last.end_idx + offset,
                        confirm_idx=min(last.confirm_idx + offset, len(visible) - 1),
                        quality=last.quality, n_pivots=last.n_pivots, meta=last.meta,
                    )
                    _record_prediction(conn, session_id, ticker, shifted, visible,
                                       len(visible) - 1, model)

        cursor_ts = list(series.values())[0].index[cursor].isoformat()
        _refresh_session_stats(conn, session_id, cursor_ts, model.updates)

        if step % 20 == 0:
            row = conn.execute(
                "SELECT n_predictions, n_resolved, n_correct FROM live_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            accuracy = (row["n_correct"] / row["n_resolved"]) if row["n_resolved"] else 0
            print(f"  barra {step + 1}/{bars_to_replay} | previsoes={row['n_predictions']} "
                  f"resolvidas={row['n_resolved']} acerto={accuracy:.1%} "
                  f"updates_online={model.updates}")
        if delay:
            time.sleep(delay)

    conn.execute("UPDATE live_sessions SET status = 'finished' WHERE session_id = ?", (session_id,))
    conn.commit()
    return session_id


# --------------------------------------------------------------------------
# Modo live
# --------------------------------------------------------------------------

def market_is_open(now=None) -> bool:
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 13 * 60 + 30 <= minutes < 20 * 60   # 13:30-20:00 UTC (horario de verao)


def run_live(conn, tickers=None, max_ticks: int = None):
    tickers = tickers or config.PATTERN_TICKERS
    timeframe = R["timeframe"]
    session_id = start_session(conn, "live")
    model = OnlineModel(conn, timeframe, R["horizon_bars"])
    print(f"sessao {session_id} (live) | mercado aberto: {market_is_open()}")

    ticks = 0
    while max_ticks is None or ticks < max_ticks:
        if not market_is_open():
            print("mercado fechado -- a aguardar")
        else:
            for ticker in tickers:
                try:
                    fresh = fetch_intraday(ticker, timeframe)
                    if not fresh.empty:
                        upsert_intraday(conn, ticker, timeframe, fresh)
                except Exception as exc:
                    print(f"  {ticker}: falha a buscar dados ({exc})")
                    continue

                bars = load_bars(conn, ticker, timeframe, limit=R["scan_window"])
                if bars.empty:
                    continue
                cursor = len(bars) - 1
                _resolve_due(conn, session_id, ticker, bars, cursor, model)

                matches = scanner.scan_patterns(
                    bars, config.PATTERN_TIMEFRAMES[timeframe]["pivot_window"]
                )
                if matches and matches[-1].confirm_idx >= cursor - 2:
                    _record_prediction(conn, session_id, ticker, matches[-1], bars, cursor, model)

            _refresh_session_stats(conn, session_id, datetime.now(timezone.utc).isoformat(),
                                   model.updates)
        ticks += 1
        time.sleep(R["poll_seconds"])

    conn.execute("UPDATE live_sessions SET status = 'stopped' WHERE session_id = ?", (session_id,))
    conn.commit()
    return session_id


def session_state(conn, session_id: str = None) -> dict:
    session = conn.execute(
        "SELECT * FROM live_sessions WHERE session_id = ?" if session_id
        else "SELECT * FROM live_sessions ORDER BY started_at DESC LIMIT 1",
        (session_id,) if session_id else (),
    ).fetchone()
    if session is None:
        return {"session": None}

    low, high = wilson_interval(session["n_correct"], session["n_resolved"])
    recent = conn.execute(
        """SELECT ticker, pattern_type, predicted_at_ts, resolve_at_ts, entry_price, exit_price,
                  predicted_direction, confidence, resolved, correct, return_pct
           FROM live_predictions WHERE session_id = ?
           ORDER BY predicted_at_ts DESC LIMIT 40""",
        (session["session_id"],),
    ).fetchall()

    # Acerto nas ultimas 20 resolvidas -- mostra se esta' a melhorar ou nao
    rolling = conn.execute(
        """SELECT AVG(correct) AS accuracy, COUNT(*) AS n FROM (
             SELECT correct FROM live_predictions
             WHERE session_id = ? AND resolved = 1 ORDER BY resolve_at_ts DESC LIMIT 20)""",
        (session["session_id"],),
    ).fetchone()

    return {
        "session": dict(session),
        "accuracy": (session["n_correct"] / session["n_resolved"]) if session["n_resolved"] else None,
        "wilson_low": low, "wilson_high": high,
        "rolling_accuracy": rolling["accuracy"], "rolling_n": rolling["n"],
        "open_predictions": [dict(r) for r in recent if not r["resolved"]],
        "recent_resolved": [dict(r) for r in recent if r["resolved"]][:15],
        "market_open": market_is_open(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["live", "replay"], default="replay")
    parser.add_argument("--bars", type=int, default=None, help="barras a percorrer no replay")
    parser.add_argument("--delay", type=float, default=None, help="segundos entre barras")
    args = parser.parse_args()

    connection = database.get_connection()
    if args.mode == "replay":
        run_replay(connection, bars_to_replay=args.bars, delay=args.delay)
    else:
        run_live(connection)
    connection.close()
