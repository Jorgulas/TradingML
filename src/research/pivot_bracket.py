"""EXPERIENCIA T -- o bracket de pivots: 25 pontos de risco para ganhar 5.

A ALEGACAO, tal como chegou: USTEC ou MNQ, barras de 1 minuto, qualquer pivot
com 10 barras a esquerda e 10 a direita e' um nivel de suporte/resistencia,
stop a 25 pontos e alvo a 5 pontos. Testado num mes, 300+ negocios, taxa de
acerto de 89-91% contra um ponto morto de 83%, com uma sequencia maxima de 5
perdas.

TRES COISAS SAO DECIDIVEIS NO PAPEL, ANTES DE QUALQUER DADO:

1. O PONTO MORTO E' A TAXA DE ACERTO DE UMA MOEDA AO AR. Com stop S e alvo A,
   o ponto morto e' S/(S+A) = 25/30 = 83.33%. E numa caminhada aleatoria sem
   deriva, a probabilidade de tocar +A antes de -S e' -- pelo teorema da paragem
   opcional -- exactamente S/(S+A) = 83.33%. O MESMO NUMERO. Nao e'
   coincidencia: e' a definicao de jogo justo. Um bracket assimetrico nao cria
   nem destroi valor sozinho, so' redistribui a mesma esperanca por uma
   distribuicao diferente. Portanto "83% de ponto morto" nao e' uma fasquia que
   a estrategia tenha de bater: e' o que ela ja' teria sem estrategia nenhuma.
   Tudo o que a alegacao afirma e' que os pivots acrescentam 6 a 8 pontos
   percentuais a uma moeda ao ar. Essa e' a hipotese inteira.

2. OS CUSTOS MOVEM O PONTO MORTO, E MOVEM-NO MUITO. Com um custo c por
   ida-e-volta, o ponto morto passa a (S+c)/(S+A). Como A e' so' 5 pontos, cada
   ponto de custo vale 3.3 pontos percentuais de taxa de acerto. Num CFD de
   USTEC, onde o spread tipico e' de 1 a 2.4 pontos, o ponto morto sobe acima
   dos 90% -- ou seja, ACIMA da taxa de acerto alegada. A mesma regra pode ser
   lucrativa em MNQ e perdedora em USTEC sem nada mudar na regra.

3. A SEQUENCIA DE 5 PERDAS CONTRADIZ A TAXA DE 90%. Com perdas independentes a
   10%, a probabilidade de haver cinco seguidas algures em 300 negocios e'
   ~1-exp(-300*0.9*0.1^5) = 0.27%. Uma sequencia maxima de 5 e' o que se espera
   de uma taxa de acerto perto dos 65%, nao dos 90%. Ou a taxa e' mais baixa do
   que a reportada, ou as perdas vem em cachos -- e cachos sao a pior das duas
   hipoteses, porque destroem a independencia de que depende toda a aritmetica
   acima. Ver `streak_consistency`.

O QUE SO' OS DADOS PODEM DECIDIR, e que esta experiencia mede:

  A. O VIES DE CONFIRMACAO DO PIVOT. Um pivot com 10 barras a direita so'
     EXISTE 10 barras depois de se formar. Um backtest que use o nivel a partir
     da barra do pivot esta' a ver 10 minutos do futuro. E' o erro mais comum
     nesta familia de estrategias e aqui mede-se quanto e' que ele vale, em vez
     de so' se evitar.

  B. O PREENCHIMENTO HONESTO CONTRA O OPTIMISTA. Quando uma barra de 1 minuto
     contem o alvo E o stop, qual foi tocado primeiro? A barra nao sabe. O
     backtest optimista arruma sempre o alvo; o honesto arruma sempre o stop. A
     diferenca entre os dois e' 30 pontos POR NEGOCIO AMBIGUO, e 2.4% das barras
     de MNQ tem amplitude >= 30 pontos. Dois porcento de ambiguidade ja' valem
     0.6 pontos por negocio -- a ordem de grandeza do custo inteiro. Nao e' um
     detalhe de implementacao: e' potencialmente o resultado todo.

  C. A LINHA DE BASE ALEATORIA. A unica comparacao que interessa. Mesmas
     sessoes, mesmo bracket, mesmo modelo de preenchimento, entradas ao acaso.
     Duas versoes: uniforme, e EMPARELHADA NA HORA DO DIA -- esta ultima mantem
     a distribuicao horaria das entradas reais e destroi so' o alinhamento com
     os pivots, que e' o placebo exigente (os pivots concentram-se nas horas
     volateis, e uma linha de base uniforme dar-lhes-ia uma vantagem de horario
     disfarcada de vantagem de sinal).

  D. O VARREDOR DE SORTE. A grelha de (esquerda, direita, stop, alvo, sentido,
     atraso) tem mais de mil celulas. A melhor celula de mil e' boa por
     construcao. Por isso mede-se tambem a melhor celula da MESMA grelha sobre
     entradas aleatorias: se a melhor real couber dentro da distribuicao da
     melhor de ruido, a grelha nao encontrou nada.
"""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database

import numpy as np
import pandas as pd

TIMEFRAME = "1m_fut"
SYMBOL = "MNQ=F"

POINT_VALUE = 2.0        # dolares por ponto de indice, num contrato MNQ
TICK = 0.25              # pontos

# Escada de custos por ida-e-volta, em PONTOS de indice. Nao se escolhe um
# numero: reporta-se a escada, porque o custo e' a variavel que decide e
# depende do intermediario e do instrumento.
COSTS = {
    "zero": 0.0,                 # o backtest ingenuo
    "mnq_optimista": 0.55,       # so' taxas de bolsa/clearing/NFA, sem spread
    "mnq_realista": 1.10,        # taxas + 1 tick de spread + 1 tick no stop
    "mnq_mau_dia": 1.75,         # stops a derrapar em movimento rapido
    "cfd_ustec": 2.50,           # spread tipico de CFD de retalho
}

SESSION_SHIFT_HOURS = 6   # 18:00 em NY passa a ser o inicio do dia seguinte (CME)


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def fetch(symbol: str = SYMBOL, weeks: int = 5) -> pd.DataFrame:
    """1 minuto em janelas de 7 dias. O yfinance recusa 1m fora dos ultimos 30
    dias, portanto isto da' o mes -- que e' exactamente a janela da alegacao."""
    import yfinance as yf

    end = date.today() + timedelta(days=1)
    frames = []
    for step in range(weeks):
        begin = end - timedelta(days=7 * (step + 1))
        stop = end - timedelta(days=7 * step)
        try:
            part = yf.Ticker(symbol).history(
                start=begin.isoformat(), end=stop.isoformat(), interval="1m")
        except Exception:
            continue
        if part is not None and not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()

    bars = pd.concat(frames).sort_index()
    bars = bars[~bars.index.duplicated(keep="first")]
    bars = bars.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
    return bars[["open", "high", "low", "close", "volume"]].dropna()


def store(conn, symbol: str, bars: pd.DataFrame) -> int:
    if bars.empty:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO watchlist (ticker, name, sector, active, added_date) "
        "VALUES (?, ?, ?, 0, ?)",
        (symbol, symbol, "Futures", datetime.now(timezone.utc).date().isoformat()))
    rows = [(symbol, TIMEFRAME,
             (stamp.tz_convert("UTC") if stamp.tzinfo else stamp.tz_localize("UTC")).isoformat(),
             float(row["open"]), float(row["high"]), float(row["low"]),
             float(row["close"]), int(row["volume"]), now)
            for stamp, row in bars.iterrows()]
    conn.executemany(
        """INSERT INTO intraday_prices
             (ticker, timeframe, ts, open, high, low, close, volume, ingested_at)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(ticker, timeframe, ts) DO UPDATE SET
             open=excluded.open, high=excluded.high, low=excluded.low,
             close=excluded.close, volume=excluded.volume,
             ingested_at=excluded.ingested_at""", rows)
    conn.commit()
    return len(rows)


def load(conn, symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> pd.DataFrame:
    frame = pd.read_sql_query(
        "SELECT ts, open, high, low, close, volume FROM intraday_prices "
        "WHERE ticker = ? AND timeframe = ? ORDER BY ts",
        conn, params=(symbol, timeframe))
    if frame.empty:
        return frame
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return add_session(frame)


def add_session(frame: pd.DataFrame) -> pd.DataFrame:
    """Sessao = dia de negociacao CME. Nada atravessa a fronteira de sessao --
    a mesma disciplina da ronda 10, pela mesma razao: um stop nao e' tocado
    durante uma pausa e um nivel de ontem nao e' o mesmo nivel de hoje."""
    frame = frame.copy()
    local = frame["ts"].dt.tz_convert("America/New_York")
    frame["session"] = (local + pd.Timedelta(hours=SESSION_SHIFT_HOURS)).dt.date
    frame["minute_of_session"] = frame.groupby("session").cumcount()
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# O motor de pivots
# ---------------------------------------------------------------------------

def pivot_flags(values: np.ndarray, left: int, right: int,
                high: bool = True) -> np.ndarray:
    """True na barra que e' o extremo estrito da janela [i-left, i+right].

    Estrito dos dois lados. Com igualdade admitida, um patamar plano de dez
    barras iguais produz dez "pivots" no mesmo preco e a contagem de niveis
    passa a medir barras paradas em vez de estrutura."""
    n = len(values)
    flags = np.ones(n, dtype=bool)
    flags[:left] = False
    flags[n - right:] = False
    for shift in range(1, left + 1):
        rolled = np.full(n, np.nan)
        rolled[shift:] = values[:-shift]
        flags &= (values > rolled) if high else (values < rolled)
    for shift in range(1, right + 1):
        rolled = np.full(n, np.nan)
        rolled[:-shift] = values[shift:]
        flags &= (values > rolled) if high else (values < rolled)
    return np.nan_to_num(flags, nan=False).astype(bool)


def levels(frame: pd.DataFrame, left: int, right: int,
           lookahead: bool = False) -> pd.DataFrame:
    """Niveis com a barra a partir da qual sao NEGOCIAVEIS.

    `lookahead=False` -- o correcto -- so' liberta o nivel em `i + right`,
    porque so' ai' e' que as barras a direita existem e o pivot se confirma.
    `lookahead=True` liberta-o na propria barra `i`, que e' o erro classico
    desta familia. Existe para ser MEDIDO, nao para ser usado.

    Os pivots nao atravessam sessoes: um pivot calculado sobre a fronteira
    misturaria o fecho de ontem com a abertura de hoje."""
    out = []
    for session, part in frame.groupby("session", sort=True):
        if len(part) < left + right + 1:
            continue
        base = part.index.to_numpy()
        for kind, column, is_high in (("resistencia", "high", True),
                                      ("suporte", "low", False)):
            flags = pivot_flags(part[column].to_numpy(float), left, right, is_high)
            where = np.flatnonzero(flags)
            if where.size == 0:
                continue
            ready = where if lookahead else where + right
            out.append(pd.DataFrame({
                "pivot_index": base[where],
                "ready_index": base[np.minimum(ready, len(base) - 1)],
                "level": part[column].to_numpy(float)[where],
                "kind": kind,
                "session": session,
            }))
    if not out:
        return pd.DataFrame(columns=["pivot_index", "ready_index", "level",
                                     "kind", "session"])
    return pd.concat(out).sort_values("ready_index").reset_index(drop=True)


def entries(frame: pd.DataFrame, table: pd.DataFrame, rule: str = "fade",
            lag: int = 0, max_wait: int = 240) -> pd.DataFrame:
    """Primeiro toque em cada nivel, depois de ele estar confirmado.

    `fade`      -- resistencia e' tecto: vende-se la'. Suporte e' chao:
                   compra-se la'. E' a leitura que um alvo de 5 pontos sugere.
    `rompimento`-- o contrario: atravessar a resistencia e' forca, compra-se.

    A descricao original ("depending on the direction of the candles around
    it") admite as duas leituras, por isso mede-se as duas. Escolher so' a que
    funciona seria escolher o resultado.

    Um nivel gera no maximo UM negocio. Sem isto, um nivel tocado vinte vezes
    numa consolidacao gera vinte entradas identicas e a contagem de negocios
    deixa de medir oportunidades."""
    if table.empty:
        return pd.DataFrame(columns=["entry_index", "entry_price", "side",
                                     "kind", "session", "wait"])
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    session = frame["session"].to_numpy()

    rows = []
    for row in table.itertuples(index=False):
        start = int(row.ready_index) + 1 + lag
        stop = min(start + max_wait, len(frame))
        if start >= stop:
            continue
        window = slice(start, stop)
        same = session[window] == row.session
        touched = (low[window] <= row.level) & (high[window] >= row.level) & same
        if not touched.any():
            continue
        offset = int(np.argmax(touched))
        if rule == "fade":
            side = -1 if row.kind == "resistencia" else 1
        else:
            side = 1 if row.kind == "resistencia" else -1
        rows.append((start + offset, float(row.level), side, row.kind,
                     row.session, offset))

    out = pd.DataFrame(rows, columns=["entry_index", "entry_price", "side",
                                      "kind", "session", "wait"])
    return out.sort_values("entry_index").reset_index(drop=True)


# ---------------------------------------------------------------------------
# A resolucao do bracket -- onde a honestidade do backtest se decide
# ---------------------------------------------------------------------------

def resolve(frame: pd.DataFrame, entry_index: np.ndarray, entry_price: np.ndarray,
            side: np.ndarray, stop: float, target: float, max_bars: int = 120,
            model: str = "honesta", start_offset: int = 1) -> pd.DataFrame:
    """Caminha o bracket barra a barra e arruma cada negocio.

    MODELOS, para a barra que contem o alvo E o stop:
      optimista -- arruma o alvo. E' o que um backtest ingenuo faz.
      honesta   -- arruma o stop. E' o pior caso.
      direccao  -- usa o sentido da propria barra: numa barra que fecha acima
                   da abertura assume-se o percurso abertura->minimo->maximo,
                   portanto o minimo veio primeiro. E' a heuristica habitual e
                   fica entre as outras duas.

    O que importa nao e' qual deles esta' certo -- nenhum esta', porque a
    informacao nao existe numa barra de 1 minuto. O que importa e' a DISTANCIA
    entre eles, que e' a incerteza honesta do resultado.

    `start_offset` E' A DECISAO MAIS CONSEQUENTE DESTA FUNCAO, e descobri-o a
    medir: com o bracket de 25/5, QUATRO EM CADA CINCO negocios resolvem-se na
    propria barra de entrada. Logo o resultado inteiro depende de uma escolha
    que parece tecnica.

      start_offset=0 deixa o maximo e o minimo da barra de ENTRADA arrumarem o
      negocio. Para uma entrada ao fecho dessa barra isto e' ver o passado: os
      extremos ja' aconteceram antes do preco a que se entrou. Para uma ordem
      limitada pousada no nivel e' defensavel -- a ordem estava la' e o resto
      do minuto e' mesmo negociavel -- mas continua a usar a amplitude da barra
      para disparar a entrada E para pagar o alvo.

      start_offset=1 so' deixa o mercado decidir a partir do minuto SEGUINTE.
      Deita fora o que restava do minuto de entrada, o que e' conservador, e e'
      a unica versao em que uma entrada ao fecho significa alguma coisa.

    Reporta-se as duas. A distancia entre elas e' o valor de um minuto de
    antevisao."""
    n = len(entry_index)
    empty = pd.DataFrame(columns=["entry_index", "side", "outcome", "gross",
                                  "bars_held", "ambiguous"])
    if n == 0:
        return empty

    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    open_ = frame["open"].to_numpy(float)
    close = frame["close"].to_numpy(float)
    session = frame["session"].to_numpy()
    total = len(frame)

    offsets = np.arange(start_offset, start_offset + max_bars)
    index = entry_index[:, None] + offsets[None, :]
    inside = index < total
    index = np.minimum(index, total - 1)
    valid = inside & (session[index] == session[entry_index][:, None])

    target_level = entry_price + side * target
    stop_level = entry_price - side * stop
    long_ = (side > 0)[:, None]

    reach_target = np.where(long_, high[index] >= target_level[:, None],
                            low[index] <= target_level[:, None]) & valid
    reach_stop = np.where(long_, low[index] <= stop_level[:, None],
                          high[index] >= stop_level[:, None]) & valid

    first_target = np.where(reach_target.any(1), reach_target.argmax(1), max_bars)
    first_stop = np.where(reach_stop.any(1), reach_stop.argmax(1), max_bars)

    ambiguous = (first_target == first_stop) & (first_target < max_bars)
    won = first_target < first_stop

    if ambiguous.any():
        if model == "optimista":
            won = won | ambiguous
        elif model == "honesta":
            pass                                  # ambiguo fica perda
        elif model == "direccao":
            bar = np.where(ambiguous, first_target, 0)
            rows = np.arange(n)
            at = index[rows, bar]
            low_first = close[at] > open_[at]
            won = np.where(ambiguous, (side > 0) != low_first, won)
        else:
            raise ValueError(f"modelo desconhecido: {model}")

    resolved = (first_target < max_bars) | (first_stop < max_bars)
    held = np.where(resolved, np.minimum(first_target, first_stop), max_bars - 1)

    # Quem nao resolveu sai ao fecho da ultima barra valida DA SUA SESSAO.
    #
    # O caso de fronteira custou um teste: quando a entrada cai na ultima barra
    # da sessao nao sobra barra valida nenhuma, `valid.sum(1) - 1` da' -1, e um
    # indice -1 em numpy aponta para o fim do vector. A versao anterior usava
    # `maximum(..., 0)` e ia buscar o preco a PRIMEIRA barra da janela -- que e'
    # ja' a sessao seguinte. Um negocio sem barras para correr arrecadava o
    # salto overnight inteiro, que num indice sao dezenas de pontos, e sempre
    # em negocios do fim do dia. Agora fica explicito e a zero.
    available = valid.sum(1)
    last = np.maximum(available - 1, 0)
    exit_price = np.where(available > 0, close[index[np.arange(n), last]],
                          entry_price)
    timeout_gross = side * (exit_price - entry_price)

    gross = np.where(resolved, np.where(won, target, -stop), timeout_gross)
    outcome = np.where(resolved, np.where(won, "alvo", "stop"),
                       np.where(available > 0, "tempo", "sem_barras"))
    held = np.where(resolved, held, last) + start_offset

    return pd.DataFrame({
        "entry_index": entry_index, "side": side, "outcome": outcome,
        "gross": gross.astype(float), "bars_held": held.astype(int),
        "ambiguous": ambiguous,
    })


def trades(frame: pd.DataFrame, left: int = 10, right: int = 10,
           stop: float = 25.0, target: float = 5.0, rule: str = "fade",
           lag: int = 0, max_bars: int = 120, model: str = "honesta",
           lookahead: bool = False, start_offset: int = 1,
           entry_at: str = "fecho") -> pd.DataFrame:
    """`entry_at` fecha o ultimo buraco da comparacao, e so' dei por ele depois
    de medir que a barra que toca um nivel tem amplitude mediana de 13.5 pontos
    contra 9.0 numa barra qualquer.

      "nivel" -- entra ao preco do proprio nivel, como uma ordem limitada
      pousada la'. Combinado com `start_offset=1` e' INJUSTO contra a regra:
      deixa o mercado afastar-se do nivel ate' ao fecho do minuto sem permitir
      que esse mesmo minuto resolva o negocio. Como a barra do toque e' 50%
      mais larga do que a media, o negocio comeca tipicamente varios pontos
      submerso. Combinado com `start_offset=0` e' injusto ao contrario.

      "fecho" -- entra ao FECHO da barra do toque e resolve a partir da
      seguinte. Nao tem ambiguidade nenhuma de percurso: a barra fecha, ve-se
      que o nivel foi tocado, envia-se uma ordem a mercado. E' implementavel
      exactamente como esta' escrito, e e' o MESMO tratamento que a linha de
      base aleatoria recebe -- que e' o que torna a comparacao legitima.

    E' o `entry_at="fecho"` que esta experiencia reporta como principal."""
    table = levels(frame, left, right, lookahead=lookahead)
    found = entries(frame, table, rule=rule, lag=lag)
    if found.empty:
        return pd.DataFrame(columns=["entry_index", "side", "outcome", "gross",
                                     "bars_held", "ambiguous", "session"])
    if entry_at == "fecho":
        found["entry_price"] = frame["close"].to_numpy(float)[
            found["entry_index"].to_numpy()]
    elif entry_at != "nivel":
        raise ValueError(f"entry_at desconhecido: {entry_at}")
    out = resolve(frame, found["entry_index"].to_numpy(),
                  found["entry_price"].to_numpy(float),
                  found["side"].to_numpy(int), stop, target, max_bars, model,
                  start_offset)
    out["session"] = found["session"].to_numpy()
    out["kind"] = found["kind"].to_numpy()
    out["minute_of_session"] = frame["minute_of_session"].to_numpy()[found["entry_index"].to_numpy()]
    return out


# ---------------------------------------------------------------------------
# Aritmetica da alegacao
# ---------------------------------------------------------------------------

def breakeven(stop: float, target: float, cost: float = 0.0) -> float:
    """Taxa de acerto que poe a esperanca a zero: (S+c)/(S+A).

    Com c=0 coincide com a probabilidade de uma caminhada sem deriva tocar +A
    antes de -S. E' por isso que 83% nao e' uma fasquia: e' a moeda ao ar."""
    return (stop + cost) / (stop + target)


def expectancy(win_rate: float, stop: float, target: float,
               cost: float = 0.0) -> float:
    return win_rate * target - (1.0 - win_rate) * stop - cost


def breakeven_ladder(stop: float = 25.0, target: float = 5.0,
                     claimed: float = 0.90) -> pd.DataFrame:
    rows = []
    for name, cost in COSTS.items():
        point = breakeven(stop, target, cost)
        edge = expectancy(claimed, stop, target, cost)
        rows.append({"custo": name, "pontos": cost,
                     "ponto_morto": point,
                     "margem_pp": (claimed - point) * 100,
                     "esperanca_pontos": edge,
                     "esperanca_usd": edge * POINT_VALUE})
    return pd.DataFrame(rows)


def max_loss_streak(wins: np.ndarray) -> int:
    losses = (~np.asarray(wins, dtype=bool)).astype(int)
    best = run = 0
    for value in losses:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def streak_consistency(n_trades: int, win_rate: float, observed: int,
                       draws: int = 20000, seed: int = 17) -> dict:
    """A sequencia maxima de perdas reportada e' compativel com a taxa de
    acerto reportada?

    E' um teste de consistencia INTERNA: nao precisa de dados nenhuns, so' dos
    dois numeros da propria alegacao. Se forem incompativeis, pelo menos um
    deles esta' errado -- ou as perdas vem em cachos, caso em que a aritmetica
    de independencia que sustenta o resto da alegacao tambem cai."""
    rng = np.random.default_rng(seed)
    sample = rng.random((draws, n_trades)) < win_rate
    streaks = np.array([max_loss_streak(row) for row in sample])
    implied = _implied_win_rate(n_trades, observed, rng)
    return {
        "n_trades": n_trades, "win_rate": win_rate, "observed_streak": observed,
        "expected_streak": float(np.median(streaks)),
        "p_at_least": float((streaks >= observed).mean()),
        "implied_win_rate": implied,
    }


def _implied_win_rate(n_trades: int, observed: int, rng, draws: int = 3000) -> float:
    """Que taxa de acerto torna a sequencia observada a MEDIANA."""
    best, gap = np.nan, np.inf
    for rate in np.arange(0.50, 0.99, 0.01):
        sample = rng.random((draws, n_trades)) < rate
        median = np.median([max_loss_streak(row) for row in sample])
        if abs(median - observed) < gap:
            best, gap = float(rate), abs(median - observed)
    return best


# ---------------------------------------------------------------------------
# Linhas de base aleatorias
# ---------------------------------------------------------------------------

def random_entries(frame: pd.DataFrame, n: int, rng, max_bars: int = 120,
                   like: pd.DataFrame = None) -> pd.DataFrame:
    """Entradas sem sinal nenhum, nas mesmas sessoes e com espaco para correr.

    `like` emparelha a HORA DO DIA com as entradas reais e troca so' o dia. Os
    pivots concentram-se nas horas volateis; uma linha de base uniforme
    dar-lhes-ia uma vantagem de horario que se leria como vantagem de sinal.
    E' o mesmo raciocinio do placebo de rotacao circular da ronda 14."""
    minute = frame["minute_of_session"].to_numpy()
    session = frame["session"].to_numpy()
    length = frame.groupby("session")["close"].transform("size").to_numpy()
    eligible = np.flatnonzero(minute < (length - max_bars))
    if eligible.size == 0:
        return pd.DataFrame(columns=["entry_index", "entry_price", "side"])

    if like is None or like.empty:
        picked = rng.choice(eligible, size=min(n, eligible.size), replace=False)
    else:
        by_minute = {}
        for index in eligible:
            by_minute.setdefault(minute[index], []).append(index)
        picked = []
        for want in like["minute_of_session"].to_numpy():
            pool = by_minute.get(int(want))
            if pool:
                picked.append(rng.choice(pool))
        picked = np.array(picked, dtype=int)
        if picked.size == 0:
            return pd.DataFrame(columns=["entry_index", "entry_price", "side"])

    picked = np.sort(picked)
    return pd.DataFrame({
        "entry_index": picked,
        "entry_price": frame["close"].to_numpy(float)[picked],
        "side": rng.choice([-1, 1], size=picked.size),
        "session": session[picked],
    })


def random_baseline(frame: pd.DataFrame, n: int, stop: float, target: float,
                    repeats: int = 200, max_bars: int = 120,
                    model: str = "honesta", seed: int = 17,
                    like: pd.DataFrame = None, start_offset: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(repeats):
        picks = random_entries(frame, n, rng, max_bars, like=like)
        if picks.empty:
            continue
        out = resolve(frame, picks["entry_index"].to_numpy(),
                      picks["entry_price"].to_numpy(float),
                      picks["side"].to_numpy(int), stop, target, max_bars, model,
                      start_offset)
        rows.append({"win_rate": float((out["outcome"] == "alvo").mean()),
                     "gross": float(out["gross"].mean()),
                     "n": len(out)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap por sessao
# ---------------------------------------------------------------------------

def session_bootstrap(result: pd.DataFrame, column: str = "gross",
                      draws: int = 4000, seed: int = 17) -> tuple:
    """Reamostra SESSOES, nao negocios.

    Dentro de um dia os negocios partilham regime, volatilidade e muitas vezes
    o mesmo movimento -- 300 negocios num mes nao sao 300 observacoes
    independentes, sao ~24. Reamostrar negocios estreitaria o intervalo por um
    factor de varias vezes, que e' o erro que a ronda 7 cometeu com janelas
    sobrepostas."""
    if result.empty:
        return (np.nan, np.nan, np.nan)
    groups = [part[column].to_numpy(float)
              for _, part in result.groupby("session")]
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for draw in range(draws):
        picked = rng.integers(0, len(groups), len(groups))
        means[draw] = np.concatenate([groups[i] for i in picked]).mean()
    return (float(result[column].mean()),
            float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


# ---------------------------------------------------------------------------
# A grelha e o varredor de sorte
# ---------------------------------------------------------------------------

GRID = {
    "left_right": ((5, 5), (10, 10), (15, 15), (20, 20)),
    "stop_target": ((15, 3), (20, 5), (25, 5), (25, 3), (25, 8),
                    (30, 5), (30, 10), (40, 5)),
    "rule": ("fade", "rompimento"),
    "lag": (0, 1),
}


def grid(frame: pd.DataFrame, cost: float = 1.10, model: str = "honesta",
         max_bars: int = 120, min_trades: int = 40,
         start_offset: int = 1) -> pd.DataFrame:
    """Varre a grelha inteira. Nao se escolhe celula -- reporta-se tudo.

    As entradas so' dependem de (esquerda, direita, regra, atraso), por isso
    calculam-se uma vez por combinacao e reaproveitam-se para todos os
    brackets. Sem isso a grelha demoraria minutos em vez de segundos."""
    rows = []
    for left, right in GRID["left_right"]:
        table = levels(frame, left, right, lookahead=False)
        for rule in GRID["rule"]:
            for lag in GRID["lag"]:
                found = entries(frame, table, rule=rule, lag=lag)
                if len(found) < min_trades:
                    continue
                index = found["entry_index"].to_numpy()
                price = frame["close"].to_numpy(float)[index]
                side = found["side"].to_numpy(int)
                for stop, target in GRID["stop_target"]:
                    out = resolve(frame, index, price, side, stop, target,
                                  max_bars, model, start_offset)
                    out["session"] = found["session"].to_numpy()
                    net = out["gross"] - cost
                    rows.append({
                        "left": left, "right": right, "rule": rule, "lag": lag,
                        "stop": stop, "target": target, "n": len(out),
                        "win_rate": float((out["outcome"] == "alvo").mean()),
                        "breakeven": breakeven(stop, target, cost),
                        "ambiguous": float(out["ambiguous"].mean()),
                        "gross": float(out["gross"].mean()),
                        "net": float(net.mean()),
                        "net_usd_total": float(net.sum() * POINT_VALUE),
                    })
    return pd.DataFrame(rows)


def noise_grid(frame: pd.DataFrame, n_trades: int, cost: float = 1.10,
               cells: int = 64, repeats: int = 200, model: str = "honesta",
               max_bars: int = 120, seed: int = 17,
               start_offset: int = 1) -> pd.DataFrame:
    """O varredor de sorte. Qual e' o MELHOR de `cells` conjuntos de entradas
    aleatorias, repetido `repeats` vezes?

    Da' a distribuicao de "a melhor celula de uma grelha deste tamanho, sem
    sinal nenhum". Se a melhor celula real couber la' dentro, a grelha
    encontrou sorte e nao estrutura. E' o mesmo que a correccao de Bonferroni
    tenta fazer, mas sem assumir independencia entre celulas -- que aqui seria
    falso, porque celulas vizinhas partilham quase todas as entradas."""
    rng = np.random.default_rng(seed)
    best = []
    for _ in range(repeats):
        scores = []
        for _ in range(cells):
            picks = random_entries(frame, n_trades, rng, max_bars)
            if picks.empty:
                continue
            stop, target = GRID["stop_target"][rng.integers(len(GRID["stop_target"]))]
            out = resolve(frame, picks["entry_index"].to_numpy(),
                          picks["entry_price"].to_numpy(float),
                          picks["side"].to_numpy(int), stop, target,
                          max_bars, model, start_offset)
            scores.append(float((out["gross"] - cost).mean()))
        if scores:
            best.append(max(scores))
    return pd.DataFrame({"melhor_net": best})


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------

def regular_hours(frame: pd.DataFrame) -> pd.DataFrame:
    """So' 09:30-16:00 em Nova Iorque.

    Importa porque a sessao de futuros corre quase 24 horas e a madrugada tem
    uma fraccao da liquidez: os spreads alargam, os stops derrapam mais, e uma
    estrategia que faca o dinheiro as 3 da manha nao e' negociavel aos custos
    que se lhe atribuem de dia."""
    local = frame["ts"].dt.tz_convert("America/New_York")
    minutes = local.dt.hour * 60 + local.dt.minute
    keep = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    return add_session(frame[keep].reset_index(drop=True))


def _line(label: str, result: pd.DataFrame, cost: float) -> dict:
    net = result["gross"] - cost
    return {"caso": label, "n": len(result),
            "acerto": float((result["outcome"] == "alvo").mean()),
            "ambiguo": float(result["ambiguous"].mean()),
            "bruto": float(result["gross"].mean()),
            "liquido": float(net.mean()),
            "usd_total": float(net.sum() * POINT_VALUE)}


def run(stop: float = 25.0, target: float = 5.0, cost: float = 1.10,
        repeats: int = 200) -> dict:
    conn = database.get_connection()
    frame = load(conn)
    if frame.empty:
        raise SystemExit("sem barras: correr fetch/store primeiro")
    frame = frame[frame.groupby("session")["close"].transform("size") > 200]
    frame = add_session(frame.reset_index(drop=True))

    out = {"barras": len(frame), "sessoes": int(frame["session"].nunique())}
    print(f"MNQ 1 minuto: {len(frame)} barras, {out['sessoes']} sessoes, "
          f"{frame['ts'].min().date()} a {frame['ts'].max().date()}")

    # -- 1. aritmetica, antes dos dados ------------------------------------
    print("\n1. O PONTO MORTO E' A MOEDA AO AR")
    ladder = breakeven_ladder(stop, target, claimed=0.90)
    print(ladder.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out["escada_custos"] = ladder

    print("\n2. A SEQUENCIA DE 5 PERDAS CONTRA A TAXA DE 90%")
    streak = streak_consistency(300, 0.90, 5, draws=8000)
    print(f"   esperada {streak['expected_streak']:.0f}, observada "
          f"{streak['observed_streak']}, p={streak['p_at_least']:.4f}; "
          f"taxa implicada pela sequencia: {streak['implied_win_rate']:.2f} "
          f"(esperanca {expectancy(streak['implied_win_rate'], stop, target, cost):+.2f} pontos)")
    out["sequencia"] = streak

    # -- 2. a regra contra o acaso ------------------------------------------
    print("\n3. A REGRA CONTRA O ACASO (barra de entrada incluida vs excluida)")
    rows = []
    for offset, label in ((0, "inclui barra de entrada"), (1, "exclui barra de entrada")):
        for model in ("optimista", "direccao", "honesta"):
            result = trades(frame, stop=stop, target=target, model=model,
                            start_offset=offset, entry_at="fecho")
            rows.append(_line(f"pivot {model} / {label}", result, cost))
        reference = trades(frame, stop=stop, target=target, model="honesta",
                           start_offset=offset, entry_at="fecho")
        for like, tag in ((None, "uniforme"), (reference, "hora emparelhada")):
            base = random_baseline(frame, len(reference), stop, target,
                                   repeats=repeats, model="honesta",
                                   start_offset=offset, like=like)
            rows.append({"caso": f"ACASO {tag} / {label}", "n": int(base["n"].mean()),
                         "acerto": float(base["win_rate"].mean()),
                         "ambiguo": np.nan,
                         "bruto": float(base["gross"].mean()),
                         "liquido": float(base["gross"].mean() - cost),
                         "usd_total": np.nan,
                         "acerto_lo": float(base["win_rate"].quantile(0.025)),
                         "acerto_hi": float(base["win_rate"].quantile(0.975))})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out["principal"] = table

    # -- 3. sentido, atraso, antevisao, horario -----------------------------
    print("\n4. VARIANTES (todas honestas, sem a barra de entrada)")
    rows = []
    for rule in ("fade", "rompimento"):
        for lag in (0, 1):
            result = trades(frame, stop=stop, target=target, rule=rule,
                            lag=lag, model="honesta", start_offset=1,
                            entry_at="fecho")
            rows.append({**_line(f"{rule} atraso={lag}", result, cost)})
    for lookahead in (False, True):
        result = trades(frame, stop=stop, target=target, model="honesta",
                        start_offset=1, lookahead=lookahead, entry_at="fecho")
        rows.append(_line(f"antevisao do pivot={lookahead}", result, cost))
    for entry_at in ("fecho", "nivel"):
        result = trades(frame, stop=stop, target=target, model="honesta",
                        start_offset=1, entry_at=entry_at)
        rows.append(_line(f"entrada no {entry_at}", result, cost))
    hours = regular_hours(frame)
    result = trades(hours, stop=stop, target=target, model="honesta",
                    start_offset=1, entry_at="fecho")
    rows.append(_line("so' 09:30-16:00 NY", result, cost))
    variants = pd.DataFrame(rows)
    print(variants.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out["variantes"] = variants

    # -- 4. intervalo por sessao --------------------------------------------
    print("\n5. INTERVALO, REAMOSTRANDO SESSOES E NAO NEGOCIOS")
    result = trades(frame, stop=stop, target=target, model="honesta",
                    start_offset=1, entry_at="fecho")
    result["net"] = result["gross"] - cost
    mean, low, high = session_bootstrap(result, "net")
    print(f"   liquido {mean:+.3f} pontos por negocio, IC95% [{low:+.3f}, {high:+.3f}] "
          f"= [{low * POINT_VALUE:+.2f}, {high * POINT_VALUE:+.2f}] USD")
    out["ic_liquido"] = (mean, low, high)

    by_day = result.groupby("session")["net"].mean()
    print(f"   sessoes positivas: {int((by_day > 0).sum())} de {len(by_day)}")
    out["sessoes_positivas"] = (int((by_day > 0).sum()), len(by_day))

    # -- 5. grelha e varredor de sorte --------------------------------------
    print("\n6. A GRELHA INTEIRA E O VARREDOR DE SORTE")
    cells = grid(frame, cost=cost, model="honesta", start_offset=1)
    cells = cells.sort_values("net", ascending=False)
    print(f"   {len(cells)} celulas; positivas em liquido: "
          f"{int((cells['net'] > 0).sum())}")
    print(cells.head(5).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out["grelha"] = cells

    noise = noise_grid(frame, int(cells["n"].median()), cost=cost,
                       cells=len(cells), repeats=60, start_offset=1)
    best_real = float(cells["net"].max())
    print(f"   melhor celula real: {best_real:+.4f} pontos")
    print(f"   melhor celula de RUIDO: mediana {noise['melhor_net'].median():+.4f}, "
          f"p95 {noise['melhor_net'].quantile(0.95):+.4f}")
    print(f"   p(ruido >= real) = {float((noise['melhor_net'] >= best_real).mean()):.4f}")
    out["ruido"] = noise
    return out


if __name__ == "__main__":
    run()
