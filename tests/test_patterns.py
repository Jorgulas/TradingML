"""Testes dos detectores contra formas sinteticas COM a geometria conhecida.

Constroi-se a serie de precos de proposito com a forma de cada padrao e
verifica-se que o detector certo dispara -- se um detector estiver com a
geometria trocada (ex.: ascendente vs descendente), estes testes apanham.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.patterns import scanner
from src.patterns.pivots import find_pivots


def _bars(closes):
    """Envolve uma serie de fechos em OHLCV plausivel (high/low ligeiramente
    afastados do fecho, para os fractais terem onde encaixar)."""
    closes = np.asarray(closes, dtype=float)
    index = pd.date_range("2026-01-01", periods=len(closes), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": np.full(len(closes), 1_000_000),
        },
        index=index,
    )


def _zigzag(levels, leg_bars=8):
    """Serie que oscila linearmente entre os niveis dados, um leg de cada vez."""
    out = []
    for start, end in zip(levels, levels[1:]):
        out.extend(np.linspace(start, end, leg_bars, endpoint=False))
    out.append(levels[-1])
    return out


def _detect_types(closes, pivot_window=3):
    matches = scanner.scan_patterns(_bars(closes), pivot_window)
    return [m.pattern_type for m in matches]


# --------------------------------------------------------------------------
# Pivots
# --------------------------------------------------------------------------

def test_pivots_alternate_high_low():
    closes = _zigzag([100, 110, 100, 110, 100, 110])
    bars = _bars(closes)
    pivots = find_pivots(bars["high"].values, bars["low"].values, 3)
    kinds = [p.kind for p in pivots]
    assert len(pivots) >= 3
    assert all(a != b for a, b in zip(kinds, kinds[1:])), f"pivots nao alternam: {kinds}"


def test_pivot_confirmation_lag_is_recorded():
    """Um pivot so' e' conhecivel `window` barras depois de acontecer -- se
    isto se perder, tudo a jusante passa a ver o futuro."""
    closes = _zigzag([100, 112, 100, 112, 100])
    bars = _bars(closes)
    window = 4
    pivots = find_pivots(bars["high"].values, bars["low"].values, window)
    assert pivots
    assert all(p.confirm_idx == p.idx + window for p in pivots)


# --------------------------------------------------------------------------
# Padroes especificos
# --------------------------------------------------------------------------

def test_detects_double_bottom():
    # fundo, topo, fundo ao mesmo nivel -> duplo fundo
    closes = _zigzag([120, 100, 118, 100.5, 120], leg_bars=10)
    assert "DOUBLE_BOTTOM" in _detect_types(closes)


def test_detects_double_top():
    closes = _zigzag([100, 120, 102, 119.5, 100], leg_bars=10)
    assert "DOUBLE_TOP" in _detect_types(closes)


def test_detects_head_and_shoulders_top():
    # ombro, vale, CABECA mais alta, vale, ombro ao nivel do primeiro
    closes = _zigzag([100, 115, 104, 128, 104, 115.5, 100], leg_bars=9)
    assert "HEAD_AND_SHOULDERS_TOP" in _detect_types(closes)


def test_detects_inverse_head_and_shoulders():
    closes = _zigzag([128, 113, 124, 100, 124, 112.5, 128], leg_bars=9)
    assert "INVERSE_HEAD_AND_SHOULDERS" in _detect_types(closes)


def test_detects_ascending_triangle():
    # topos horizontais, fundos a subir
    closes = _zigzag([100, 120, 106, 120, 112, 120, 116, 120], leg_bars=8)
    assert "ASCENDING_TRIANGLE" in _detect_types(closes)


def test_detects_descending_triangle():
    # fundos horizontais, topos a descer
    closes = _zigzag([120, 100, 114, 100, 108, 100, 104, 100], leg_bars=8)
    assert "DESCENDING_TRIANGLE" in _detect_types(closes)


def test_ascending_and_descending_are_not_confused():
    ascending = _detect_types(_zigzag([100, 120, 106, 120, 112, 120, 116, 120], leg_bars=8))
    descending = _detect_types(_zigzag([120, 100, 114, 100, 108, 100, 104, 100], leg_bars=8))
    assert "DESCENDING_TRIANGLE" not in ascending
    assert "ASCENDING_TRIANGLE" not in descending


def test_detects_rectangle():
    # Topos e fundos ambos horizontais. Amplitude deliberadamente pequena
    # (2.5%): com a amplitude grande dos outros testes, a subida inicial ate'
    # ao primeiro topo conta como mastro e a leitura correcta passa a ser
    # BULL_FLAG (canal paralelo depois de um movimento forte) -- que e' mesmo
    # a diferenca entre um rectangulo e uma bandeira.
    closes = _zigzag([100, 102.5, 100, 102.5, 100, 102.5, 100], leg_bars=9)
    detected = _detect_types(closes)
    assert any(t.startswith("RECTANGLE") for t in detected), detected


def test_same_channel_becomes_a_flag_when_preceded_by_a_pole():
    """Mesma geometria de canal, mas com um movimento forte imediatamente
    antes: deixa de ser rectangulo e passa a bandeira."""
    closes = _zigzag([100, 115, 100, 115, 100, 115, 100], leg_bars=9)
    detected = _detect_types(closes)
    assert "BULL_FLAG" in detected, detected


def test_flat_noise_produces_no_pattern():
    rng = np.random.default_rng(7)
    closes = 100 + rng.normal(0, 0.02, 400).cumsum() * 0.01
    detected = _detect_types(closes)
    assert len(detected) <= 2, f"ruido quase plano gerou padroes a mais: {detected}"


def test_every_detected_type_is_a_known_pattern():
    closes = _zigzag([100, 115, 102, 128, 103, 116, 100, 118, 104], leg_bars=9)
    for pattern_type in _detect_types(closes):
        assert pattern_type in config.PATTERN_TYPES


def test_detected_patterns_do_not_overlap():
    closes = _zigzag([100, 118, 102, 130, 104, 119, 101, 120, 105, 122], leg_bars=8)
    matches = scanner.scan_patterns(_bars(closes), 3)
    spans = sorted((m.start_idx, m.end_idx) for m in matches)
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start > end, f"padroes sobrepostos: {spans}"


def test_confirm_idx_never_precedes_pattern_end():
    """Nenhum padrao pode ser dado como conhecido antes de terminar."""
    closes = _zigzag([100, 118, 102, 130, 104, 119, 100], leg_bars=9)
    for match in scanner.scan_patterns(_bars(closes), 3):
        assert match.confirm_idx >= match.end_idx
