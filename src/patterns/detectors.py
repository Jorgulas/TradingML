"""Detectores geometricos dos padroes classicos de analise tecnica.

Cada detector recebe uma janela de pivots consecutivos (e as barras, para
contexto de tendencia previa) e devolve um PatternMatch ou None. Todos
devolvem tambem um `quality` em [0,1] -- o quao bem a geometria real encaixa
no ideal do padrao -- para o scan a jusante poder escolher a melhor
interpretacao quando varios padroes encaixam na mesma zona.

Declives sao sempre normalizados ao preco medio (fraccao por barra), para os
limiares fazerem sentido igual numa accao a $30 e noutra a $900.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.patterns.pivots import split_by_kind

import numpy as np

P = config.PATTERN_DETECTION


# Equivalente de pivots atribuido a' chavena, que e' detectada por ajuste de
# parabola a's barras e nao por pivots. Mantem-na competitiva com um triangulo
# tipico sem a deixar dominar (dar-lhe o span como peso fazia as chavenas
# saltarem para 46% de todas as deteccoes a 1h).
CUP_PIVOT_EQUIVALENT = 4


@dataclass
class PatternMatch:
    pattern_type: str
    start_idx: int
    end_idx: int
    confirm_idx: int
    quality: float
    # Quantos pontos de viragem independentes sustentam ESTA deteccao. E' a
    # medida de especificidade usada para desempatar leituras sobrepostas, e
    # tem de ser por deteccao e nao por tipo de padrao: um duplo topo sao
    # sempre 3 pivots, mas um triangulo ascendente tanto pode ter 4 como 8, e
    # a versao de 8 e' uma leitura muito mais forte do grafico do que o duplo
    # topo que vive la' dentro.
    n_pivots: int = 3
    meta: dict = field(default_factory=dict)


def _fit_line(xs, ys):
    """Devolve (slope_normalizado, intercept, r2). slope em fraccao/barra."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mean_price = float(np.mean(ys))
    if len(xs) < 2 or mean_price == 0:
        return 0.0, float(np.mean(ys)) if len(ys) else 0.0, 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    if len(xs) == 2:
        # Uma recta passa sempre exactamente por dois pontos: R2=1 aqui nao e'
        # evidencia nenhuma. Devolve-se um valor neutro para nao inflacionar a
        # quality de padroes assentes em pouquissimos pivots.
        r2 = 0.75
    else:
        # Pontos praticamente ao MESMO nivel: uma recta horizontal ajusta-os na
        # perfeicao, logo R2=1. O teste tem de ser por tolerancia RELATIVA e
        # nao `ss_tot > 0`: com precos como 115.11 repetidos, o ruido de
        # virgula flutuante deixa ss_tot em ~1e-28 em vez de 0 exacto, a
        # divisao explode e sai um R2 negativo (-4.67 observado) que rejeitava
        # silenciosamente todo e qualquer rectangulo ou triangulo com um lado
        # plano -- justamente os padroes que tem um lado plano por definicao.
        if (float(np.max(ys)) - float(np.min(ys))) / abs(mean_price) < 1e-9:
            r2 = 1.0
        else:
            predicted = slope * xs + intercept
            ss_res = float(np.sum((ys - predicted) ** 2))
            ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
            r2 = 1.0 - ss_res / ss_tot
    return float(slope) / mean_price, float(intercept), float(r2)


def _evidence(*groups) -> float:
    """Peso em [0,1] pela quantidade de pivots que sustentam a deteccao."""
    total = sum(len(g) for g in groups)
    return min(total, 6) / 6


def _raw_fit(xs, ys):
    """Ajuste em unidades de preco (nao normalizado), para calcular larguras
    entre as duas fronteiras. `_fit_line` normaliza o declive pelo preco medio
    do SEU grupo, e topos e fundos tem medias diferentes -- misturar os dois
    ao calcular larguras da resultados sem sentido dimensional."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return 0.0, float(ys[0]) if len(ys) else 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _boundary_widths(highs, lows, x_start: float, x_end: float):
    """Distancia entre a trendline de cima e a de baixo, no inicio e no fim.
    A convergir -> triangulo/cunha. A divergir -> broadening."""
    slope_h, intercept_h = _raw_fit([p.idx for p in highs], [p.price for p in highs])
    slope_l, intercept_l = _raw_fit([p.idx for p in lows], [p.price for p in lows])
    width_start = (slope_h * x_start + intercept_h) - (slope_l * x_start + intercept_l)
    width_end = (slope_h * x_end + intercept_h) - (slope_l * x_end + intercept_l)
    return width_start, width_end


def _spread(values) -> float:
    """Dispersao relativa de um conjunto de niveis (0 = perfeitamente iguais)."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if mean == 0:
        return 1.0
    return float(np.max(values) - np.min(values)) / abs(mean)


def _prior_trend(bars, start_idx: int, lookback: int = 30) -> float:
    """Retorno fraccionario nas barras imediatamente ANTES do padrao."""
    begin = max(0, start_idx - lookback)
    if start_idx <= begin:
        return 0.0
    closes = bars["close"].values
    first, last = closes[begin], closes[max(begin, start_idx - 1)]
    return float(last / first - 1) if first else 0.0


def _is_flat(slope: float) -> bool:
    return abs(slope) <= P["flat_slope_max"]


def _is_rising(slope: float) -> bool:
    return slope >= P["min_slope"]


def _is_falling(slope: float) -> bool:
    return slope <= -P["min_slope"]


def _span_ok(pivots) -> bool:
    span = pivots[-1].idx - pivots[0].idx
    return P["min_pattern_bars"] <= span <= P["max_pattern_bars"]


def _height_ok(pivots) -> bool:
    prices = [p.price for p in pivots]
    return _spread(prices) >= P["min_pattern_height"]


def _base(pivots, pattern_type, quality, meta=None):
    return PatternMatch(
        pattern_type=pattern_type,
        start_idx=pivots[0].idx,
        end_idx=pivots[-1].idx,
        confirm_idx=max(p.confirm_idx for p in pivots),
        quality=float(np.clip(quality, 0.0, 1.0)),
        n_pivots=len(pivots),
        meta=meta or {},
    )


# --------------------------------------------------------------------------
# Triangulos e rectangulos: duas trendlines ajustadas aos topos e aos fundos
# --------------------------------------------------------------------------

def detect_triangle_or_rectangle(pivots, bars):
    highs, lows = split_by_kind(pivots)
    if len(highs) < 2 or len(lows) < 2 or not _span_ok(pivots) or not _height_ok(pivots):
        return None

    slope_h, int_h, r2_h = _fit_line([p.idx for p in highs], [p.price for p in highs])
    slope_l, int_l, r2_l = _fit_line([p.idx for p in lows], [p.price for p in lows])
    if min(r2_h, r2_l) < P["min_r2"]:
        return None

    evidence = _evidence(highs, lows)
    fit_quality = ((r2_h + r2_l) / 2) * evidence
    meta = {"slope_high": slope_h, "slope_low": slope_l, "r2_high": r2_h,
            "r2_low": r2_l, "n_pivots": len(pivots)}

    # Rectangulo: ambas as fronteiras horizontais
    if _is_flat(slope_h) and _is_flat(slope_l):
        flatness = 1 - (abs(slope_h) + abs(slope_l)) / (2 * P["flat_slope_max"])
        quality = 0.5 * fit_quality + 0.5 * flatness * evidence
        trend = _prior_trend(bars, pivots[0].idx)
        kind = "RECTANGLE_TOP" if trend > 0 else "RECTANGLE_BOTTOM"
        return _base(pivots, kind, quality, {**meta, "prior_trend": trend})

    # Triangulos: fronteiras tem de convergir (largura a fechar)
    width_start, width_end = _boundary_widths(highs, lows, pivots[0].idx, pivots[-1].idx)
    converging = width_end < width_start

    if _is_flat(slope_h) and _is_rising(slope_l) and converging:
        return _base(pivots, "ASCENDING_TRIANGLE", 0.6 * fit_quality + 0.4 * evidence, meta)
    if _is_falling(slope_h) and _is_flat(slope_l) and converging:
        return _base(pivots, "DESCENDING_TRIANGLE", 0.6 * fit_quality + 0.4 * evidence, meta)
    if _is_falling(slope_h) and _is_rising(slope_l):
        symmetry = 1 - abs(abs(slope_h) - abs(slope_l)) / max(abs(slope_h) + abs(slope_l), 1e-9)
        return _base(pivots, "SYMMETRICAL_TRIANGLE", 0.5 * fit_quality + 0.5 * symmetry * evidence, meta)
    return None


# --------------------------------------------------------------------------
# Cunhas: as DUAS fronteiras inclinadas no MESMO sentido, a convergir.
# E' o que as distingue dos triangulos (uma fronteira plana ou sentidos
# opostos) e das bandeiras (fronteiras paralelas, sem convergencia).
# --------------------------------------------------------------------------

MIN_CONVERGENCE = 0.25  # a largura tem de fechar >=25% para contar como cunha


def detect_wedge(pivots, bars):
    highs, lows = split_by_kind(pivots)
    if len(highs) < 2 or len(lows) < 2 or not _span_ok(pivots):
        return None

    slope_h, _, r2_h = _fit_line([p.idx for p in highs], [p.price for p in highs])
    slope_l, _, r2_l = _fit_line([p.idx for p in lows], [p.price for p in lows])
    if min(r2_h, r2_l) < P["min_r2"]:
        return None

    if _is_rising(slope_h) and _is_rising(slope_l):
        kind = "RISING_WEDGE"        # bearish: procura a esgotar-se na subida
    elif _is_falling(slope_h) and _is_falling(slope_l):
        kind = "FALLING_WEDGE"       # bullish: venda a esgotar-se na descida
    else:
        return None

    width_start, width_end = _boundary_widths(highs, lows, pivots[0].idx, pivots[-1].idx)
    if width_start <= 0:
        return None
    convergence = 1 - (width_end / width_start)
    if convergence < MIN_CONVERGENCE:
        return None  # inclinacao no mesmo sentido mas sem fechar: e' um canal

    evidence = _evidence(highs, lows)
    fit_quality = ((r2_h + r2_l) / 2) * evidence
    quality = 0.5 * fit_quality + 0.5 * min(convergence / 0.6, 1.0) * evidence
    meta = {"slope_high": slope_h, "slope_low": slope_l, "convergence": convergence,
            "n_pivots": len(pivots)}
    return _base(pivots, kind, quality, meta)


# --------------------------------------------------------------------------
# Broadening (megafone): o oposto do triangulo -- fronteiras a divergir.
# --------------------------------------------------------------------------

MIN_DIVERGENCE = 0.35


def detect_broadening(pivots, bars):
    highs, lows = split_by_kind(pivots)
    if len(highs) < 2 or len(lows) < 2 or not _span_ok(pivots) or not _height_ok(pivots):
        return None

    slope_h, _, r2_h = _fit_line([p.idx for p in highs], [p.price for p in highs])
    slope_l, _, r2_l = _fit_line([p.idx for p in lows], [p.price for p in lows])
    if min(r2_h, r2_l) < P["min_r2"]:
        return None
    if not (_is_rising(slope_h) and _is_falling(slope_l)):
        return None  # topos mais altos E fundos mais baixos: a abrir dos dois lados

    width_start, width_end = _boundary_widths(highs, lows, pivots[0].idx, pivots[-1].idx)
    if width_start <= 0:
        return None
    divergence = (width_end / width_start) - 1
    if divergence < MIN_DIVERGENCE:
        return None

    evidence = _evidence(highs, lows)
    quality = 0.5 * ((r2_h + r2_l) / 2) * evidence + 0.5 * min(divergence / 1.0, 1.0) * evidence
    trend = _prior_trend(bars, pivots[0].idx)
    kind = "BROADENING_TOP" if trend > 0 else "BROADENING_BOTTOM"
    meta = {"slope_high": slope_h, "slope_low": slope_l, "divergence": divergence,
            "prior_trend": trend, "n_pivots": len(pivots)}
    return _base(pivots, kind, quality, meta)


# --------------------------------------------------------------------------
# Flags e pennants: exigem um "mastro" (movimento forte e rapido) antes
# --------------------------------------------------------------------------

def _find_flagpole(bars, start_idx: int):
    """Procura, imediatamente antes do padrao, um movimento >= flagpole_min_move
    em <= flagpole_max_bars. Devolve (direccao, magnitude) ou None."""
    closes = bars["close"].values
    best = None
    for length in range(5, P["flagpole_max_bars"] + 1):
        begin = start_idx - length
        if begin < 0:
            break
        move = float(closes[start_idx] / closes[begin] - 1)
        if abs(move) >= P["flagpole_min_move"] and (best is None or abs(move) > abs(best[1])):
            best = (1 if move > 0 else -1, move)
    return best


def detect_flag_or_pennant(pivots, bars):
    highs, lows = split_by_kind(pivots)
    if len(highs) < 2 or len(lows) < 2:
        return None
    span = pivots[-1].idx - pivots[0].idx
    if span < P["min_pattern_bars"] or span > P["flag_max_bars"]:
        return None

    pole = _find_flagpole(bars, pivots[0].idx)
    if pole is None:
        return None
    direction, magnitude = pole

    slope_h, _, r2_h = _fit_line([p.idx for p in highs], [p.price for p in highs])
    slope_l, _, r2_l = _fit_line([p.idx for p in lows], [p.price for p in lows])
    if min(r2_h, r2_l) < P["min_r2"]:
        return None

    fit_quality = (r2_h + r2_l) / 2
    meta = {"pole_direction": direction, "pole_move": magnitude,
            "slope_high": slope_h, "slope_low": slope_l}

    # Pennant: pequeno triangulo simetrico logo a seguir ao mastro
    if _is_falling(slope_h) and _is_rising(slope_l):
        return _base(pivots, "PENNANT", 0.5 * fit_quality + 0.5, meta)

    # Flag: canal PARALELO inclinado contra o mastro.
    #
    # A verificacao de paralelismo pelos declives sozinha nao chegava: uma
    # cunha descendente tem os dois declives negativos e proximos um do outro,
    # passava neste teste e era classificada como BULL_FLAG -- e como as
    # bandeiras sao a deteccao mais comum, isso contaminava o maior estado da
    # matriz com uma formacao de significado oposto. Exige-se agora tambem que
    # a largura do canal se mantenha: um canal a fechar e' cunha, nao bandeira.
    parallel = abs(slope_h - slope_l) <= P["min_slope"]
    if not parallel:
        return None
    width_start, width_end = _boundary_widths(highs, lows, pivots[0].idx, pivots[-1].idx)
    if width_start > 0 and (1 - width_end / width_start) >= MIN_CONVERGENCE:
        return None
    channel_slope = (slope_h + slope_l) / 2
    if direction > 0 and channel_slope <= P["flat_slope_max"]:
        return _base(pivots, "BULL_FLAG", 0.5 * fit_quality + 0.5, meta)
    if direction < 0 and channel_slope >= -P["flat_slope_max"]:
        return _base(pivots, "BEAR_FLAG", 0.5 * fit_quality + 0.5, meta)
    return None


# --------------------------------------------------------------------------
# Duplos topos/fundos
# --------------------------------------------------------------------------

def detect_double(pivots, bars):
    if len(pivots) != 3 or not _span_ok(pivots):
        return None
    a, mid, b = pivots
    if a.kind != b.kind or mid.kind == a.kind:
        return None

    level_diff = abs(a.price - b.price) / max(abs((a.price + b.price) / 2), 1e-9)
    if level_diff > P["level_tolerance"]:
        return None
    excursion = abs(mid.price - (a.price + b.price) / 2) / max(abs(mid.price), 1e-9)
    if excursion < P["min_pattern_height"]:
        return None

    quality = (1 - level_diff / P["level_tolerance"]) * 0.6 + min(excursion / 0.05, 1.0) * 0.4
    meta = {"level_diff": level_diff, "excursion": excursion}
    kind = "DOUBLE_BOTTOM" if a.kind == "L" else "DOUBLE_TOP"
    return _base(pivots, kind, quality, meta)


# --------------------------------------------------------------------------
# Head and shoulders (normal e invertido)
# --------------------------------------------------------------------------

def detect_head_and_shoulders(pivots, bars):
    if len(pivots) != 5 or not _span_ok(pivots):
        return None
    p1, p2, p3, p4, p5 = pivots
    if not (p1.kind == p3.kind == p5.kind) or p2.kind == p1.kind or p4.kind != p2.kind:
        return None

    shoulders = [p1.price, p5.price]
    head, neckline = p3.price, [p2.price, p4.price]

    shoulder_diff = _spread(shoulders)
    neckline_diff = _spread(neckline)
    if shoulder_diff > P["level_tolerance"] or neckline_diff > P["level_tolerance"] * 1.5:
        return None

    avg_shoulder = float(np.mean(shoulders))
    head_margin = (head - avg_shoulder) / max(abs(avg_shoulder), 1e-9)
    if p1.kind == "H":
        if head_margin < P["hs_head_margin"]:
            return None
        kind = "HEAD_AND_SHOULDERS_TOP"
    else:
        if -head_margin < P["hs_head_margin"]:
            return None
        kind = "INVERSE_HEAD_AND_SHOULDERS"

    quality = (1 - shoulder_diff / P["level_tolerance"]) * 0.5 + \
              min(abs(head_margin) / 0.05, 1.0) * 0.3 + \
              (1 - neckline_diff / (P["level_tolerance"] * 1.5)) * 0.2
    meta = {"shoulder_diff": shoulder_diff, "head_margin": head_margin, "neckline_diff": neckline_diff}
    return _base(pivots, kind, quality, meta)


# --------------------------------------------------------------------------
# Diamante: primeiro alarga (broadening), depois estreita (converging)
# --------------------------------------------------------------------------

def detect_diamond(pivots, bars):
    if len(pivots) < 5 or not _span_ok(pivots) or not _height_ok(pivots):
        return None

    middle = len(pivots) // 2
    first, second = pivots[:middle + 1], pivots[middle:]
    fh, fl = split_by_kind(first)
    sh, sl = split_by_kind(second)
    if min(len(fh), len(fl), len(sh), len(sl)) < 2:
        return None

    slope_fh, _, _ = _fit_line([p.idx for p in fh], [p.price for p in fh])
    slope_fl, _, _ = _fit_line([p.idx for p in fl], [p.price for p in fl])
    slope_sh, _, _ = _fit_line([p.idx for p in sh], [p.price for p in sh])
    slope_sl, _, _ = _fit_line([p.idx for p in sl], [p.price for p in sl])

    broadening = _is_rising(slope_fh) and _is_falling(slope_fl)
    narrowing = _is_falling(slope_sh) and _is_rising(slope_sl)
    if not (broadening and narrowing):
        return None

    strength = min((abs(slope_fh) + abs(slope_fl) + abs(slope_sh) + abs(slope_sl)) / (4 * P["min_slope"]), 2.0) / 2
    trend = _prior_trend(bars, pivots[0].idx)
    kind = "DIAMOND_TOP" if trend > 0 else "DIAMOND_BOTTOM"
    meta = {"prior_trend": trend, "slopes": [slope_fh, slope_fl, slope_sh, slope_sl]}
    return _base(pivots, kind, 0.5 + 0.5 * strength, meta)


PIVOT_DETECTORS = [
    detect_double,
    detect_head_and_shoulders,
    detect_diamond,
    detect_broadening,
    detect_wedge,
    detect_flag_or_pennant,
    detect_triangle_or_rectangle,
]


# --------------------------------------------------------------------------
# Cup with handle: forma arredondada, nao se apanha bem por pivots.
# Detecta-se ajustando uma parabola as barras e exigindo vertice ao meio.
# --------------------------------------------------------------------------

def detect_cup(bars, start_idx: int, length: int, pivot_window: int):
    end_idx = start_idx + length
    if end_idx >= len(bars):
        return None
    closes = bars["close"].values[start_idx:end_idx]
    xs = np.arange(len(closes), dtype=float)
    mean_price = float(np.mean(closes))
    if mean_price <= 0:
        return None

    coeffs = np.polyfit(xs, closes, 2)
    a, b, _ = coeffs
    if a == 0:
        return None
    vertex = -b / (2 * a)
    if not (0.3 * length <= vertex <= 0.7 * length):
        return None  # o extremo tem de estar a meio, senao nao e' uma taca

    predicted = np.polyval(coeffs, xs)
    ss_res = float(np.sum((closes - predicted) ** 2))
    ss_tot = float(np.sum((closes - np.mean(closes)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if r2 < P["cup_min_r2"]:
        return None

    left_rim, right_rim = float(closes[0]), float(closes[-1])
    rim_diff = abs(left_rim - right_rim) / max(abs((left_rim + right_rim) / 2), 1e-9)
    if rim_diff > P["level_tolerance"] * 2:
        return None

    extreme = float(np.min(closes)) if a > 0 else float(np.max(closes))
    depth = abs(extreme - (left_rim + right_rim) / 2) / max(abs(mean_price), 1e-9)
    if not (P["cup_min_depth"] <= depth <= P["cup_max_depth"]):
        return None

    # Pega: pequeno recuo depois do rim direito, sem desfazer metade da taca
    handle_max = min(length // 2, 30)
    handle_end, handle_size = end_idx, 0
    for handle_len in range(pivot_window, handle_max + 1):
        stop = end_idx + handle_len
        if stop >= len(bars):
            break
        segment = bars["close"].values[end_idx:stop]
        retrace = abs(float(segment.min() if a > 0 else segment.max()) - right_rim) / max(depth * mean_price, 1e-9)
        if retrace <= P["cup_handle_max_retrace"]:
            handle_end, handle_size = stop, handle_len
    if handle_size < pivot_window:
        return None

    kind = "CUP_WITH_HANDLE" if a > 0 else "INVERSE_CUP_WITH_HANDLE"
    quality = 0.5 * r2 + 0.3 * (1 - rim_diff / (P["level_tolerance"] * 2)) + 0.2 * min(depth / 0.1, 1.0)
    return PatternMatch(
        pattern_type=kind,
        start_idx=start_idx,
        end_idx=handle_end,
        confirm_idx=handle_end + pivot_window,
        quality=float(np.clip(quality, 0.0, 1.0)),
        n_pivots=CUP_PIVOT_EQUIVALENT,
        meta={"r2": r2, "depth": depth, "rim_diff": rim_diff, "handle_bars": handle_size},
    )
