"""Varre um historico de barras e devolve a sequencia cronologica de padroes.

Estrategia: gerar todos os candidatos (varias janelas de pivots + varredura
de tacas), depois escolher gulosamente por qualidade os que nao se sobrepoem.
O resultado e' uma sequencia limpa e nao-sobreposta -- que e' exactamente o
que a cadeia de Markov a jusante precisa para contar transicoes.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.patterns import detectors
from src.patterns.pivots import find_pivots

MIN_PIVOTS, MAX_PIVOTS = 3, 7
PIVOT_BONUS = 0.15  # ver comentario em scan_patterns()


def _generate_candidates(bars, pivot_window: int) -> list:
    highs = bars["high"].values
    lows = bars["low"].values
    pivots = find_pivots(highs, lows, pivot_window)

    candidates = []
    for end in range(len(pivots)):
        for size in range(MIN_PIVOTS, MAX_PIVOTS + 1):
            start = end - size + 1
            if start < 0:
                break
            window = pivots[start:end + 1]
            for detector in detectors.PIVOT_DETECTORS:
                try:
                    match = detector(window, bars)
                except Exception:
                    continue
                if match and match.quality >= config.PATTERN_DETECTION["min_quality"]:
                    candidates.append(match)

    # Tacas: varredura por barras (a forma arredondada nao se apanha por pivots)
    step = max(pivot_window, 3)
    for length in range(30, min(config.PATTERN_DETECTION["max_pattern_bars"], len(bars)), 15):
        for start in range(0, len(bars) - length - 1, step * 2):
            match = detectors.detect_cup(bars, start, length, pivot_window)
            if match and match.quality >= config.PATTERN_DETECTION["min_quality"]:
                candidates.append(match)
    return candidates


def scan_patterns(bars, pivot_window: int) -> list:
    """Sequencia cronologica de padroes nao-sobrepostos."""
    if len(bars) < config.PATTERN_DETECTION["min_pattern_bars"] * 2:
        return []

    # Desempate entre leituras sobrepostas: quality ponderada pelo numero de
    # pontos de viragem que sustentam cada deteccao.
    #
    # Um head-and-shoulders contem sempre, geometricamente, um duplo topo e
    # varios triangulos dentro de si -- a ambiguidade e' real, nao um defeito.
    # Testaram-se quatro criterios contra a metrica que interessa (a cadeia
    # de Markov bater o baseline de frequencia, ver sequence.evaluate):
    #   - so' quality           -> ganha sempre a leitura pequena: tres pivots
    #                              perfeitos batem oito quase perfeitos
    #   - rank fixo por tipo    -> o inverso, e mal calibrado: dava rank 3 ao
    #                              duplo topo (3 pivots) e rank 1 ao triangulo
    #                              (ate' 8 pivots), quando o triangulo e' que e'
    #                              a leitura mais rica
    #   - span primeiro         -> colapsa nas formacoes maiores (chavenas a
    #                              46% das deteccoes a 1h) e a cadeia passou a
    #                              PERDER do baseline
    #   - n_pivots da deteccao  -> escolhido: mede a especificidade REAL de
    #                              cada deteccao em vez de a assumir pelo tipo
    candidates = _generate_candidates(bars, pivot_window)
    candidates.sort(key=lambda m: (-(m.quality * (1 + PIVOT_BONUS * m.n_pivots)), m.start_idx))

    accepted, occupied = [], []
    for match in candidates:
        if any(match.start_idx <= end and start <= match.end_idx for start, end in occupied):
            continue
        accepted.append(match)
        occupied.append((match.start_idx, match.end_idx))

    accepted.sort(key=lambda m: m.confirm_idx)
    return accepted


def store_patterns(conn, ticker: str, timeframe: str, bars, matches: list) -> int:
    now = datetime.now(timezone.utc).isoformat()
    index = bars.index
    rows = []
    for match in matches:
        confirm_pos = min(match.confirm_idx, len(index) - 1)
        rows.append((
            ticker, timeframe, match.pattern_type,
            index[match.start_idx].isoformat(),
            index[match.end_idx].isoformat(),
            index[confirm_pos].isoformat(),
            match.start_idx, match.end_idx, match.quality,
            json.dumps(match.meta, default=float), now,
        ))
    conn.executemany(
        """INSERT INTO detected_patterns
             (ticker, timeframe, pattern_type, start_ts, end_ts, confirmed_ts,
              start_idx, end_idx, quality, meta_json, detected_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(ticker, timeframe, pattern_type, start_ts, end_ts) DO UPDATE SET
             quality = excluded.quality, meta_json = excluded.meta_json,
             detected_at = excluded.detected_at""",
        rows,
    )
    conn.commit()
    return len(rows)
