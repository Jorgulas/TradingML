"""EXPERIENCIA U -- o RSI(2) de Connors, com o protocolo da ronda 17.

A ESTRATEGIA, do livro de Larry Connors e Cesar Alvarez (2008). Compra-se
quando as tres condicoes se verificam no fecho: o preco esta' acima da media
de 200 dias (tendencia principal a subir), abaixo da media de 5 dias (recuo de
curto prazo) e o RSI de 2 periodos esta' abaixo de 10. Vende-se quando o fecho
volta acima da media de 5 dias. Connors recomenda explicitamente NAO usar stop.

AS ALEGACOES publicas andam nos 71-75% de acerto. O que raramente aparece ao
lado e' o tamanho dos ganhos: um dos backtests publicos reporta ganho medio de
1,22% contra perda media de 2,10%. Isso poe o ponto morto em
2,10/(2,10+1,22) = 63,2%, pela mesma aritmetica da ronda 17 -- e um acerto de
71% deixa entao uma margem de 8pp, nao de 71%.

E' a MESMA estrutura da ronda 17: taxa de acerto alta comprada com uma
assimetria de pagamento que ninguem poe ao lado dela. A diferenca e' que aqui o
bracket nao e' fixo, e' a propria regra de saida que o cria -- sair quando o
preco recupera trunca os ganhos cedo e deixa as perdas correr. O acerto alto e'
uma CONSEQUENCIA MECANICA da regra de saida, nao uma medida de acerto.

O PROTOCOLO E' O DA RONDA 17, com uma peca nova obrigatoria:

  1. LINHA DE BASE ALEATORIA EMPARELHADA. Nao basta "entrar ao acaso": a regra
     tem tres condicoes e cada uma pode estar a fazer o trabalho todo. Por isso
     monta-se uma ESCADA que isola o contributo de cada uma --
       (a) comprar e ficar quieto na mesma accao;
       (b) entrada ao acaso, MESMA regra de saida;
       (c) entrada ao acaso so' em dias acima da media de 200;
       (d) entrada ao acaso em dias acima da media de 200 E abaixo da de 5
           (isto e', qualquer recuo numa tendencia, sem RSI nenhum);
       (e) a regra completa.
     A diferenca entre (d) e (e) e' tudo o que o RSI acrescenta. Se for zero, a
     estrategia e' "compra recuos" com um indicador decorativo por cima.

  2. O VIES DE SOBREVIVENCIA MORDE AQUI MAIS DO QUE EM QUALQUER OUTRO SITIO
     DESTE PROJECTO, e nenhum dos backtests publicos o testa. Comprar quedas e'
     a estrategia mais contaminada que existe: num universo de sobreviventes,
     TODAS as quedas recuperaram -- e' isso que significa ter sobrevivido. A
     ronda 8 deixou 62 accoes que cairam do indice com historico completo, e
     elas sao exactamente a parte da distribuicao onde comprar a queda nao
     funcionou. Corre-se a regra nos dois universos.

  3. A ANTEVISAO DE UM FECHO. O RSI(2) so' se calcula DEPOIS do fecho, e a
     regra compra a esse mesmo fecho. A saida tem o mesmo problema: a media de
     5 dias inclui o fecho do proprio dia. Sao duas antevisoes de uma barra,
     irmas da barra de entrada da ronda 17. Mede-se a versao que entra e sai na
     ABERTURA SEGUINTE, que e' a unica executavel como esta' escrita.

  4. REPLICA POR DECADA. Sao 65 anos. A fasquia da ronda 15 aplica-se: um
     efeito real tem de aparecer em decadas com microestrutura, regulacao e
     custos completamente diferentes.

  5. E OS CUSTOS DE 1970 NAO SAO OS DE HOJE. Ate' 1975 as comissoes eram fixas
     e enormes, e os spreads em fraccoes de oitavos valiam dezenas de pontos
     base. Testar 65 anos com custos de hoje atribui a estrategia um lucro que
     nenhum participante da epoca poderia ter tido.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.research.fallen_universe import FALLEN

import numpy as np
import pandas as pd

TIMEFRAME = "1dfull"

# Custos de ida-e-volta em pontos base. A escada existe porque o custo e' a
# variavel que decide e mudou por mais de uma ordem de grandeza ao longo da
# amostra.
COSTS_BPS = {
    "zero": 0.0,
    "hoje_liquido": 5.0,      # spread de large-cap americana, ida-e-volta
    "hoje_realista": 10.0,    # spread + impacto numa ordem a mercado
    "anos_90": 40.0,          # spreads em oitavos, comissoes de retalho
    "pre_1975": 150.0,        # comissoes fixas
}

MAX_HOLD = 20                 # dias; sem isto um negocio fica aberto anos


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------

def wilder_rsi(close: np.ndarray, period: int = 2) -> np.ndarray:
    """RSI com a suavizacao de Wilder, que e' a que as plataformas usam.

    Com period=2 o RSI e' extremamente nervoso -- e' precisamente o que a
    estrategia quer, um detector de dois dias a cair."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    if len(close) <= period:
        return np.full(len(close), np.nan)

    avg_gain[period] = gain[1:period + 1].mean()
    avg_loss[period] = loss[1:period + 1].mean()
    alpha = 1.0 / period

    # A suavizacao de Wilder E' exactamente um `ewm` com alpha=1/n e
    # adjust=False. Escrita como ciclo em Python custava 69 milhoes de
    # iteracoes a varrer a grelha inteira sobre 160 accoes com 65 anos cada --
    # a experiencia nao terminava. O ciclo fica em `_wilder_loop` como
    # referencia e um teste exige que as duas versoes coincidam.
    for name, seeded, series in (("gain", avg_gain, gain), ("loss", avg_loss, loss)):
        tail = pd.Series(series[period + 1:], dtype=float)
        seed = pd.Series([seeded[period]], dtype=float)
        smoothed = pd.concat([seed, tail], ignore_index=True).ewm(
            alpha=alpha, adjust=False).mean().to_numpy()
        seeded[period:] = smoothed

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
    # Sem perdas nenhumas o RSI e' 100 por definicao, nao indefinido.
    rsi = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, rsi)
    rsi = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, rsi)
    return rsi


def _wilder_loop(close: np.ndarray, period: int = 2) -> np.ndarray:
    """A mesma coisa escrita a mao. So' existe para o teste confirmar que a
    versao rapida nao mudou a aritmetica."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    if len(close) <= period:
        return np.full(len(close), np.nan)
    avg_gain[period] = gain[1:period + 1].mean()
    avg_loss[period] = loss[1:period + 1].mean()
    alpha = 1.0 / period
    for i in range(period + 1, len(close)):
        avg_gain[i] = avg_gain[i - 1] * (1 - alpha) + gain[i] * alpha
        avg_loss[i] = avg_loss[i - 1] * (1 - alpha) + loss[i] * alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    rsi = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, rsi)
    return np.where((avg_loss == 0) & (avg_gain == 0), 50.0, rsi)


def indicators(bars: pd.DataFrame, rsi_period: int = 2, fast: int = 5,
               slow: int = 200) -> pd.DataFrame:
    out = bars.copy()
    close = out["close"].to_numpy(float)
    out["ma_fast"] = out["close"].rolling(fast).mean()
    out["ma_slow"] = out["close"].rolling(slow).mean()
    out["rsi"] = wilder_rsi(close, rsi_period)
    return out


# ---------------------------------------------------------------------------
# A regra, e as condicoes isoladas
# ---------------------------------------------------------------------------

def entry_mask(bars: pd.DataFrame, threshold: float = 10.0,
               use_slow: bool = True, use_fast: bool = True,
               use_rsi: bool = True) -> np.ndarray:
    """Cada condicao pode ser desligada, que e' como se mede o que cada uma
    vale. A regra completa e' as tres ligadas."""
    mask = np.ones(len(bars), dtype=bool)
    if use_slow:
        mask &= (bars["close"] > bars["ma_slow"]).to_numpy()
    if use_fast:
        mask &= (bars["close"] < bars["ma_fast"]).to_numpy()
    if use_rsi:
        mask &= (bars["rsi"] < threshold).to_numpy()
    return mask & bars[["ma_slow", "ma_fast", "rsi"]].notna().all(axis=1).to_numpy()


def next_exit_index(exit_ok: np.ndarray) -> np.ndarray:
    """Para cada dia, o primeiro dia POSTERIOR em que a condicao de saida se
    verifica. Vectorizado: um acumulado de minimos de tras para a frente.

    O `>` em vez de `>=` nao e' detalhe -- no dia da entrada a condicao de
    saida e' falsa por construcao (entrou-se por estar ABAIXO da media rapida),
    mas numa variante sem esse filtro podia ser verdadeira, e sair no proprio
    dia da entrada seria um negocio de duracao zero."""
    n = len(exit_ok)
    marker = np.where(exit_ok, np.arange(n), n)
    earliest = np.minimum.accumulate(marker[::-1])[::-1]
    return np.append(earliest[1:], n)


def build_trades(bars: pd.DataFrame, enter: np.ndarray, max_hold: int = MAX_HOLD,
                 execute: str = "fecho") -> pd.DataFrame:
    """Transforma dias de entrada em negocios.

    `execute="fecho"`   -- entra e sai ao fecho do dia do sinal. E' o que os
                           backtests publicos fazem, e tem uma antevisao de uma
                           barra nas DUAS pontas: o RSI(2) e a media de 5 dias
                           so' se conhecem depois desse fecho.
    `execute="abertura"`-- entra e sai na abertura do dia SEGUINTE. E' a unica
                           versao executavel exactamente como a regra esta'
                           escrita."""
    close = bars["close"].to_numpy(float)
    open_ = bars["open"].to_numpy(float)
    n = len(bars)
    exit_ok = (bars["close"] > bars["ma_fast"]).to_numpy() & bars["ma_fast"].notna().to_numpy()
    nxt = next_exit_index(exit_ok)

    entries = np.flatnonzero(enter)
    if entries.size == 0:
        return pd.DataFrame(columns=["entry", "exit", "hold", "ret", "ts"])

    exits = np.minimum(nxt[entries], entries + max_hold)
    exits = np.minimum(exits, n - 1)
    keep = exits > entries
    entries, exits = entries[keep], exits[keep]
    if entries.size == 0:
        return pd.DataFrame(columns=["entry", "exit", "hold", "ret", "ts"])

    if execute == "fecho":
        buy, sell = close[entries], close[exits]
        stamps = bars["ts"].to_numpy()[entries]
    elif execute == "abertura":
        # Entra e sai na abertura seguinte; perde-se o ultimo sinal se nao
        # houver dia a seguir.
        valid = (exits + 1) < n
        entries, exits = entries[valid], exits[valid]
        if entries.size == 0:
            return pd.DataFrame(columns=["entry", "exit", "hold", "ret", "ts"])
        buy, sell = open_[entries + 1], open_[exits + 1]
        stamps = bars["ts"].to_numpy()[entries + 1]
    else:
        raise ValueError(f"execute desconhecido: {execute}")

    return pd.DataFrame({
        "entry": entries, "exit": exits, "hold": exits - entries,
        "ret": sell / buy - 1.0, "ts": stamps,
    })


# ---------------------------------------------------------------------------
# A escada de linhas de base
# ---------------------------------------------------------------------------

def random_mask(bars: pd.DataFrame, n: int, rng, pool: np.ndarray = None
                ) -> np.ndarray:
    """`n` dias ao acaso, opcionalmente so' de dentro de `pool`.

    O emparelhamento na CONTAGEM e' o que torna a comparacao justa: com a mesma
    regra de saida, mais entradas nao e' melhor nem pior, mas as estatisticas
    de uma amostra de 50 e de uma de 5000 nao se comparam."""
    candidates = np.flatnonzero(pool) if pool is not None else np.arange(len(bars))
    valid = bars[["ma_slow", "ma_fast"]].notna().all(axis=1).to_numpy()
    candidates = candidates[valid[candidates]]
    out = np.zeros(len(bars), dtype=bool)
    if candidates.size == 0:
        return out
    picked = rng.choice(candidates, size=min(n, candidates.size), replace=False)
    out[picked] = True
    return out


def buy_and_hold(bars: pd.DataFrame, holds: np.ndarray, rng,
                 execute: str = "fecho") -> np.ndarray:
    """Retorno da MESMA accao em janelas do MESMO comprimento, escolhidas ao
    acaso. E' a linha de base mais apertada que existe: controla a deriva, a
    accao e a duracao de uma vez. Qualquer coisa que a regra faca tem de ser
    feita CONTRA isto, e nao contra zero."""
    price = bars["close" if execute == "fecho" else "open"].to_numpy(float)
    n = len(price)
    if n < 2 or holds.size == 0:
        return np.array([])
    starts = rng.integers(0, n - 1, holds.size)
    ends = np.minimum(starts + holds, n - 1)
    keep = ends > starts
    return price[ends[keep]] / price[starts[keep]] - 1.0


# ---------------------------------------------------------------------------
# Painel
# ---------------------------------------------------------------------------

def load_bars(conn, ticker: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        "SELECT ts, open, high, low, close, volume FROM intraday_prices "
        "WHERE ticker = ? AND timeframe = ? ORDER BY ts",
        conn, params=(ticker, TIMEFRAME))
    if frame.empty:
        return frame
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame


def universe(conn, which: str = "sobreviventes") -> list:
    rows = [r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM intraday_prices WHERE timeframe = ?",
        (TIMEFRAME,))]
    fallen = set(FALLEN)
    if which == "sobreviventes":
        return sorted(t for t in rows if t not in fallen)
    if which == "caidas":
        return sorted(t for t in rows if t in fallen)
    return sorted(rows)


_CACHE = {}


def cached_bars(conn, ticker: str) -> pd.DataFrame:
    """A grelha corre 48 celulas sobre 160 accoes. Sem cache sao 7680 leituras
    da base de dados das mesmas 2 milhoes de linhas."""
    if ticker not in _CACHE:
        _CACHE[ticker] = load_bars(conn, ticker)
    return _CACHE[ticker]


def run_rule(conn, tickers: list, threshold: float = 10.0, execute: str = "fecho",
             max_hold: int = MAX_HOLD, use_slow: bool = True,
             use_fast: bool = True, use_rsi: bool = True,
             rsi_period: int = 2, fast: int = 5, slow: int = 200,
             baselines: bool = False, seed: int = 18) -> dict:
    """Corre a regra sobre um universo e, se pedido, a escada de linhas de base
    sobre exactamente as mesmas accoes e com a mesma contagem de entradas."""
    rng = np.random.default_rng(seed)
    parts = {"regra": []}
    if baselines:
        parts.update({"acaso": [], "acaso_tendencia": [], "acaso_recuo": [],
                      "comprar_e_esperar": []})

    for ticker in tickers:
        bars = cached_bars(conn, ticker)
        if len(bars) < slow + 50:
            continue
        bars = indicators(bars, rsi_period, fast, slow)

        mask = entry_mask(bars, threshold, use_slow, use_fast, use_rsi)
        trades = build_trades(bars, mask, max_hold, execute)
        if trades.empty:
            continue
        trades["ticker"] = ticker
        parts["regra"].append(trades)

        if not baselines:
            continue
        count = len(trades)
        pools = {
            "acaso": None,
            "acaso_tendencia": entry_mask(bars, use_slow=True, use_fast=False,
                                          use_rsi=False),
            "acaso_recuo": entry_mask(bars, use_slow=True, use_fast=True,
                                      use_rsi=False),
        }
        for name, pool in pools.items():
            picked = random_mask(bars, count, rng, pool)
            alt = build_trades(bars, picked, max_hold, execute)
            if not alt.empty:
                alt["ticker"] = ticker
                parts[name].append(alt)
        held = buy_and_hold(bars, trades["hold"].to_numpy(), rng, execute)
        if held.size:
            parts["comprar_e_esperar"].append(pd.DataFrame({
                "ret": held, "hold": trades["hold"].to_numpy()[:held.size],
                "ts": trades["ts"].to_numpy()[:held.size], "ticker": ticker}))

    return {name: (pd.concat(frames).reset_index(drop=True) if frames
                   else pd.DataFrame(columns=["ret", "hold", "ts", "ticker"]))
            for name, frames in parts.items()}


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------

def summary(trades: pd.DataFrame, cost_bps: float = 10.0) -> dict:
    if trades.empty:
        return {"n": 0}
    net = trades["ret"] - cost_bps / 1e4
    wins = net > 0
    win_mean = float(net[wins].mean()) if wins.any() else 0.0
    loss_mean = float(-net[~wins].mean()) if (~wins).any() else 0.0
    denominator = win_mean + loss_mean
    return {
        "n": len(trades),
        "acerto": float(wins.mean()),
        "ganho_medio": win_mean,
        "perda_media": loss_mean,
        "ponto_morto": loss_mean / denominator if denominator > 0 else np.nan,
        "media": float(net.mean()),
        "dias": float(trades["hold"].mean()),
        "por_dia": float(net.mean() / trades["hold"].mean()),
    }


def date_bootstrap(trades: pd.DataFrame, cost_bps: float = 10.0,
                   draws: int = 3000, seed: int = 18) -> tuple:
    """Reamostra DATAS e nao negocios.

    Duzentas accoes entram no mesmo recuo do mercado no mesmo dia. Tratar esses
    negocios como independentes e' o erro das janelas sobrepostas da ronda 7
    outra vez, por outro caminho: a contagem de negocios e' enorme e o numero
    de dias distintos e' uma fraccao dela."""
    if trades.empty:
        return (np.nan, np.nan, np.nan)
    work = trades.copy()
    work["net"] = work["ret"] - cost_bps / 1e4
    work["day"] = pd.to_datetime(work["ts"]).dt.date
    # Reamostrar dias concatenando os seus negocios custaria, com 15 mil dias
    # e 3 mil sorteios, 45 milhoes de concatenacoes. A media de uma uniao de
    # grupos e' soma(somas)/soma(contagens), portanto basta reamostrar os dois
    # vectores -- exacto, e em segundos em vez de horas.
    grouped = work.groupby("day")["net"]
    sums = grouped.sum().to_numpy(float)
    counts = grouped.size().to_numpy(float)
    rng = np.random.default_rng(seed)
    days = len(sums)
    means = np.empty(draws)
    chunk = max(1, min(draws, 2_000_000 // max(days, 1)))
    done = 0
    while done < draws:
        size = min(chunk, draws - done)
        picked = rng.integers(0, days, (size, days))
        means[done:done + size] = sums[picked].sum(1) / counts[picked].sum(1)
        done += size
    return (float(work["net"].mean()), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def by_decade(trades: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    work = trades.copy()
    work["decada"] = (pd.to_datetime(work["ts"]).dt.year // 10 * 10)
    work["net"] = work["ret"] - cost_bps / 1e4
    table = work.groupby("decada").agg(
        n=("net", "size"), acerto=("net", lambda s: float((s > 0).mean())),
        media=("net", "mean"), dias=("hold", "mean"))
    table["accoes"] = work.groupby("decada")["ticker"].nunique()
    return table.reset_index()


# ---------------------------------------------------------------------------
# Grelha e varredor de sorte
# ---------------------------------------------------------------------------

GRID = {
    "rsi_period": (2, 3, 4),
    "threshold": (5.0, 10.0, 15.0, 20.0),
    "fast": (5, 10),
    "slow": (100, 200),
}


def grid(conn, tickers: list, cost_bps: float = 10.0,
         execute: str = "abertura") -> pd.DataFrame:
    rows = []
    for period in GRID["rsi_period"]:
        for threshold in GRID["threshold"]:
            for fast in GRID["fast"]:
                for slow in GRID["slow"]:
                    result = run_rule(conn, tickers, threshold=threshold,
                                      execute=execute, rsi_period=period,
                                      fast=fast, slow=slow)["regra"]
                    if len(result) < 200:
                        continue
                    stats = summary(result, cost_bps)
                    rows.append({"rsi": period, "limiar": threshold,
                                 "rapida": fast, "lenta": slow, **stats})
    return pd.DataFrame(rows)


def noise_grid(conn, tickers: list, cells: int, cost_bps: float = 10.0,
               execute: str = "abertura", repeats: int = 40,
               seed: int = 18) -> pd.DataFrame:
    """A melhor celula de uma grelha do mesmo tamanho sobre entradas ao acaso.

    As entradas aleatorias ficam restritas a dias acima da media de 200 e
    abaixo da de 5 -- ou seja, ao mesmo POOL de recuos em tendencia que a regra
    usa. Procurar sorte dentro do pool errado seria uma fasquia baixa de mais."""
    rng = np.random.default_rng(seed)
    cache = []
    for ticker in tickers:
        bars = cached_bars(conn, ticker)
        if len(bars) < 250:
            continue
        bars = indicators(bars)
        pool = entry_mask(bars, use_slow=True, use_fast=True, use_rsi=False)
        counts = int(entry_mask(bars).sum())
        if counts:
            cache.append((bars, pool, counts))

    best = []
    for _ in range(repeats):
        scores = []
        for _ in range(cells):
            parts = []
            for bars, pool, counts in cache:
                picked = random_mask(bars, counts, rng, pool)
                alt = build_trades(bars, picked, MAX_HOLD, execute)
                if not alt.empty:
                    parts.append(alt)
            if parts:
                merged = pd.concat(parts)
                scores.append(float((merged["ret"] - cost_bps / 1e4).mean()))
        if scores:
            best.append(max(scores))
    return pd.DataFrame({"melhor_media": best})


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------

def run(cost_bps: float = 10.0, execute: str = "abertura") -> dict:
    conn = database.get_connection()
    survivors = universe(conn, "sobreviventes")
    fallen = universe(conn, "caidas")
    print(f"universo: {len(survivors)} sobreviventes, {len(fallen)} caidas "
          f"do indice (ronda 8), historico diario completo")
    out = {}

    # -- 1. a aritmetica da alegacao ---------------------------------------
    print("\n1. A ARITMETICA DA ALEGACAO PUBLICA")
    claimed_win, claimed_loss, claimed_rate = 0.0122, 0.0210, 0.712
    point = claimed_loss / (claimed_win + claimed_loss)
    print(f"   ganho medio 1.22%, perda media 2.10% -> ponto morto {point:.4f}")
    print(f"   acerto alegado {claimed_rate:.3f} -> margem {(claimed_rate - point) * 100:+.2f}pp")
    print(f"   esperanca alegada: {claimed_rate * claimed_win - (1 - claimed_rate) * claimed_loss:+.5f}")
    out["aritmetica"] = {"ponto_morto": point}

    # -- 2. o limite de dias e' um botao ------------------------------------
    print("\n2. A TAXA DE ACERTO CONTRA O LIMITE DE DIAS")
    rows = []
    for hold in (2, 3, 5, 10, 20, 40):
        result = run_rule(conn, survivors, max_hold=hold, execute=execute)["regra"]
        rows.append({"max_dias": hold, **summary(result, cost_bps)})
    dial = pd.DataFrame(rows)
    print(dial.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out["botao"] = dial

    # -- 3. a escada de linhas de base --------------------------------------
    print("\n3. A ESCADA: O QUE CADA CONDICAO ACRESCENTA")
    for label, names in (("SOBREVIVENTES", survivors), ("CAIDAS DO INDICE", fallen)):
        print(f"\n   -- {label} ({len(names)} accoes) --")
        parts = run_rule(conn, names, execute=execute, baselines=True)
        rows = []
        for name, frame in parts.items():
            stats = summary(frame, cost_bps)
            mean, low, high = date_bootstrap(frame, cost_bps)
            rows.append({"caso": name, **stats, "ic_lo": low, "ic_hi": high})
        table = pd.DataFrame(rows)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        out[f"escada_{label}"] = table
        out[f"negocios_{label}"] = parts["regra"]

    # -- 4. replica por decada ----------------------------------------------
    print("\n4. REPLICA POR DECADA (sobreviventes)")
    decades = by_decade(out["negocios_SOBREVIVENTES"], cost_bps)
    print(decades.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out["decadas"] = decades

    # -- 5. custos -----------------------------------------------------------
    print("\n5. A ESCADA DE CUSTOS (sobreviventes)")
    rows = []
    for name, bps in COSTS_BPS.items():
        stats = summary(out["negocios_SOBREVIVENTES"], bps)
        rows.append({"custo": name, "bps": bps, "acerto": stats["acerto"],
                     "media": stats["media"], "por_dia": stats["por_dia"]})
    costs = pd.DataFrame(rows)
    print(costs.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    out["custos"] = costs

    # -- 6. grelha e varredor de sorte ---------------------------------------
    print("\n6. A GRELHA E O VARREDOR DE SORTE (sobreviventes)")
    cells = grid(conn, survivors, cost_bps, execute).sort_values(
        "media", ascending=False)
    print(f"   {len(cells)} celulas, positivas: {int((cells['media'] > 0).sum())}")
    print(cells.head(5).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    noise = noise_grid(conn, survivors, len(cells), cost_bps, execute, repeats=20)
    best = float(cells["media"].max())
    print(f"   melhor celula real: {best:+.5f}")
    print(f"   melhor de RUIDO: mediana {noise['melhor_media'].median():+.5f}, "
          f"p95 {noise['melhor_media'].quantile(0.95):+.5f}")
    print(f"   p(ruido >= real) = {float((noise['melhor_media'] >= best).mean()):.4f}")
    out["grelha"], out["ruido"] = cells, noise
    return out


if __name__ == "__main__":
    run()
