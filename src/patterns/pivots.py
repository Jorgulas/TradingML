"""Deteccao de pivots (swing highs/lows) pelo metodo dos fractais.

Um pivot alto no indice i e' uma barra cujo high e' o maximo da janela
[i-w, i+w]. Consequencia importante e inevitavel: um pivot em i so' fica
CONFIRMADO w barras depois (em i+w), porque ate' la' ainda pode aparecer uma
barra mais alta a' direita que o invalide. Todo o codigo a jusante trata
`confirm_idx` como o momento em que a informacao passou a existir -- e' isto
que impede o sistema de "ver o futuro" ao avaliar padroes historicamente.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pivot:
    idx: int          # indice da barra onde o extremo acontece
    price: float
    kind: str         # 'H' (swing high) ou 'L' (swing low)
    confirm_idx: int  # indice a partir do qual este pivot e' conhecivel


def find_pivots(highs, lows, window: int) -> list:
    """Devolve pivots alternados (H, L, H, L, ...) por ordem cronologica.

    Fractais crus podem dar dois topos seguidos sem fundo pelo meio; nesse
    caso fica so' o mais extremo, para a geometria a jusante poder assumir
    sempre alternancia."""
    n = len(highs)
    raw = []
    for i in range(window, n - window):
        left_h, right_h = highs[i - window:i], highs[i + 1:i + window + 1]
        if highs[i] >= max(left_h) and highs[i] >= max(right_h):
            raw.append(Pivot(i, float(highs[i]), "H", i + window))
        left_l, right_l = lows[i - window:i], lows[i + 1:i + window + 1]
        if lows[i] <= min(left_l) and lows[i] <= min(right_l):
            raw.append(Pivot(i, float(lows[i]), "L", i + window))

    raw.sort(key=lambda p: (p.idx, p.kind))

    alternating = []
    for pivot in raw:
        if not alternating:
            alternating.append(pivot)
            continue
        last = alternating[-1]
        if pivot.kind != last.kind:
            alternating.append(pivot)
        elif (pivot.kind == "H" and pivot.price > last.price) or \
             (pivot.kind == "L" and pivot.price < last.price):
            alternating[-1] = pivot  # mesmo tipo seguido: fica o mais extremo
    return alternating


def split_by_kind(pivots: list):
    highs = [p for p in pivots if p.kind == "H"]
    lows = [p for p in pivots if p.kind == "L"]
    return highs, lows
