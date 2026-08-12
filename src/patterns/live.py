"""Deteccao + previsao em tempo real, para ser chamada a cada actualizacao
do grafico. Desenhada para responder em milissegundos:

  - a matriz de transicoes e' carregada uma vez e fica em cache no processo
    (so' muda quando o backfill volta a correr);
  - a varredura de padroes so' olha para as ultimas SCAN_BARS barras, nao
    para o historico todo.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.patterns import scanner, sequence
from src.patterns.ingest_intraday import load_bars

SCAN_BARS = 600  # janela suficiente para o maior padrao (150 barras) e rapida

_TRANSITION_CACHE = {}


def get_transitions(conn, timeframe: str, refresh: bool = False) -> dict:
    if refresh or timeframe not in _TRANSITION_CACHE:
        _TRANSITION_CACHE[timeframe] = sequence.load_transitions(conn, timeframe)
    return _TRANSITION_CACHE[timeframe]


def timeframe_minutes(timeframe: str) -> int:
    return {"1h": 60, "5m": 5, "15m": 15, "1m": 1}.get(timeframe, 5)


def analyse(conn, ticker: str, timeframe: str = None, steps: int = None) -> dict:
    timeframe = timeframe or config.PATTERN_LIVE_TIMEFRAME
    pivot_window = config.PATTERN_TIMEFRAMES[timeframe]["pivot_window"]

    bars = load_bars(conn, ticker, timeframe, limit=SCAN_BARS)
    if bars.empty:
        return {"ticker": ticker, "timeframe": timeframe, "error": "sem barras para este ticker/timeframe"}

    matches = scanner.scan_patterns(bars, pivot_window)
    last_ts = bars.index[-1]
    minutes = timeframe_minutes(timeframe)

    recent = [
        {
            "pattern_type": m.pattern_type,
            "bias": config.PATTERN_BIAS.get(m.pattern_type),
            "quality": round(m.quality, 3),
            "start_ts": bars.index[m.start_idx].isoformat(),
            "end_ts": bars.index[m.end_idx].isoformat(),
            "start_price": float(bars["close"].iloc[m.start_idx]),
            "end_price": float(bars["close"].iloc[m.end_idx]),
        }
        for m in matches[-6:]
    ]

    if not matches:
        return {
            "ticker": ticker, "timeframe": timeframe,
            "as_of_ts": last_ts.isoformat(),
            "last_price": float(bars["close"].iloc[-1]),
            "current_pattern": None,
            "recent_patterns": [],
            "forecast": [],
            "alternatives": [],
            "note": "nenhum padrao reconhecivel na janela recente -- sem estado de partida para a cadeia",
        }

    current = matches[-1]
    transitions = get_transitions(conn, timeframe)

    # Passo 1 pelo classificador contextual, se estiver ligado. Por omissao
    # nao esta': medido, nao bate a cadeia de Markov (ver a nota em
    # config.PATTERN_SEQUENCE["use_classifier"]).
    step1_distribution, step1_source = None, "markov"
    if config.PATTERN_SEQUENCE.get("use_classifier"):
        from src.patterns import classifier, context as pattern_context

        features = pattern_context.pattern_context_features(
            bars, current.start_idx, current.end_idx,
            current.pattern_type, current.quality, current.n_pivots,
        )
        step1_distribution = classifier.predict_distribution(timeframe, features)
        if step1_distribution:
            step1_source = "classifier"

    best, alternatives = sequence.forecast_chain(
        transitions, current.pattern_type, steps=steps, timeframe=timeframe,
        step1_distribution=step1_distribution,
    )

    def serialise(chain):
        return [
            {
                "step": s.step,
                "pattern_type": s.pattern_type,
                "bias": config.PATTERN_BIAS.get(s.pattern_type),
                "step_confidence": round(s.step_confidence, 4),
                "cumulative_confidence": round(s.cumulative_confidence, 5),
                "expected_bars": round(s.expected_bars, 1),
                "expected_minutes": round(s.expected_bars * minutes, 1),
                "support": s.support,
                "low_support": s.support < config.PATTERN_SEQUENCE["min_support"],
                "alternatives": s.alternatives,
            }
            for s in chain
        ]

    bars_since = len(bars) - 1 - current.end_idx
    return {
        "ticker": ticker,
        "timeframe": timeframe,
        "as_of_ts": last_ts.isoformat(),
        "last_price": float(bars["close"].iloc[-1]),
        "current_pattern": {
            "pattern_type": current.pattern_type,
            "bias": config.PATTERN_BIAS.get(current.pattern_type),
            "quality": round(current.quality, 3),
            "start_ts": bars.index[current.start_idx].isoformat(),
            "end_ts": bars.index[current.end_idx].isoformat(),
            "bars_since_completion": bars_since,
        },
        "recent_patterns": recent,
        "forecast": serialise(best),
        "alternatives": [serialise(alt) for alt in alternatives[:3]],
        "step1_source": step1_source,
    }


def store_forecast(conn, result: dict) -> int:
    """Grava um snapshot da cadeia prevista, para mais tarde se poder ver se
    as previsoes que o sistema fez em tempo real se confirmaram."""
    if not result.get("forecast") or not result.get("current_pattern"):
        return 0
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """INSERT INTO pattern_forecasts
             (ticker, timeframe, as_of_ts, from_pattern, step, pattern_type,
              step_confidence, cumulative_confidence, expected_bars, support, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(ticker, timeframe, as_of_ts, step) DO UPDATE SET
             pattern_type = excluded.pattern_type,
             step_confidence = excluded.step_confidence,
             cumulative_confidence = excluded.cumulative_confidence,
             expected_bars = excluded.expected_bars,
             support = excluded.support""",
        [
            (result["ticker"], result["timeframe"], result["as_of_ts"],
             result["current_pattern"]["pattern_type"], s["step"], s["pattern_type"],
             s["step_confidence"], s["cumulative_confidence"], s["expected_bars"],
             s["support"], now)
            for s in result["forecast"]
        ],
    )
    conn.commit()
    return len(result["forecast"])


if __name__ == "__main__":
    import time

    connection = database.get_connection()
    for tk in config.PATTERN_TICKERS[:3]:
        t0 = time.time()
        result = analyse(connection, tk)
        elapsed = (time.time() - t0) * 1000
        current = result.get("current_pattern")
        label = f"{current['pattern_type']} (qualidade {current['quality']})" if current else "nenhum"
        print(f"\n{tk} ({result['timeframe']}, {elapsed:.0f}ms) -- padrao actual: {label}")
        for step in result.get("forecast", []):
            flag = "  [suporte baixo]" if step["low_support"] else ""
            print(f"   passo {step['step']}: {step['pattern_type']:28s} "
                  f"cond={step['step_confidence']:.1%}  acum={step['cumulative_confidence']:.2%} "
                  f"~{step['expected_minutes']:.0f}min  n={step['support']}{flag}")
            alts = ", ".join(
                f"{a['pattern_type']} {a['probability']:.1%}(n={a['support']})" for a in step["alternatives"]
            )
            print(f"            alternativas: {alts}")
    connection.close()
