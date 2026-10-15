"""EXPERIENCIA V -- quatro horas a negociar ao minuto com 3000 euros.

O PEDIDO: um algoritmo continuo, a negociar ao minuto durante quatro horas,
com uma conta de 3000 euros. A resposta util nao e' UMA simulacao -- e' a
DISTRIBUICAO sobre todas as janelas de quatro horas que existem nos dados
reais, porque uma simulacao sozinha e' uma amostra de tamanho um e nao
distingue sorte de sistema.

A PREMISSA QUE ESTA EXPERIENCIA TEM DE CORRIGIR. A ronda 17 nao encontrou
"acerto alto". Encontrou que o acerto alto nao e' informacao: com stop 25 e
alvo 5, entradas completamente ALEATORIAS acertam 83.3%, e os 89-91% alegados
sao o que o acaso produz quando se deixa a barra de entrada arrumar o negocio.
Medida honestamente, a regra de pivots acerta 82% e perde 1.28 pontos por
negocio. Acertar muito e perder dinheiro sao, neste bracket, a MESMA COISA --
e' para isso que serve esta simulacao: por isso em euros.

A ARITMETICA QUE DECIDE TUDO, antes de qualquer codigo. Cada ida-e-volta custa
cerca de 1.10 pontos = 2.20 dolares num contrato MNQ. Negociar ao minuto
durante quatro horas sao ate' 240 negocios. 240 x 2.20 = 528 dolares de custos
num so' dia, sobre uma conta de 3366 dolares: **15.7% do capital por dia, so'
em custos**, antes de o mercado fazer o que quer que seja. Para ficar
empatado, a estrategia tem de gerar 15.7% de lucro bruto por dia. Nenhuma
gera.

E' por isso que a frequencia e' o inimigo de uma conta pequena: o custo escala
com o numero de negocios e o capital nao.

DOIS MODOS, porque "algoritmo continuo ao minuto" admite duas leituras:

  continuo -- entra em cada minuto em que esta' de fora, sentido a' sorte. E' a
              leitura literal, e e' a linha de base que mostra o que a
              frequencia por si so' faz a uma conta.
  sinal    -- so' entra quando a regra de pivots da ronda 17 dispara. Negoceia
              menos, portanto paga menos custos, portanto perde menos.

E uma TERCEIRA coluna que nao e' uma simulacao mas uma conta: o que aconteceria
SE a taxa de acerto alegada de 89% fosse verdade. Serve para mostrar que o
pedido nao e' absurdo -- com 89% isto daria mesmo dinheiro. O problema nao e' a
aritmetica do plano, e' que 89% nao e' a taxa de acerto; e' 82%.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from db import database
from src.research import pivot_bracket as T

import numpy as np
import pandas as pd

EUR_USD = 1.1221
ACCOUNT_EUR = 3000.0
ACCOUNT_USD = ACCOUNT_EUR * EUR_USD

MINUTES = 240                 # quatro horas
STOP, TARGET = 25.0, 5.0
COST_POINTS = 1.10            # taxas + 1 tick de spread + 1 tick no stop
POINT_VALUE = T.POINT_VALUE   # 2 dolares por ponto, contrato MNQ
DAY_MARGIN = 200.0            # margem intradiaria por contrato (varia por corretor)


def windows(frame: pd.DataFrame, minutes: int = MINUTES) -> list:
    """Janelas de `minutes` barras que nao se sobrepoem e nao atravessam
    sessoes. Sem sobreposicao porque janelas que partilham barras partilham
    negocios, e o intervalo de confianca estreitaria por uma razao falsa -- o
    erro das janelas sobrepostas da ronda 7."""
    out = []
    for _, part in frame.groupby("session", sort=True):
        index = part.index.to_numpy()
        for start in range(0, len(index) - minutes, minutes):
            out.append((index[start], index[start + minutes - 1]))
    return out


def simulate(frame: pd.DataFrame, lo: int, hi: int, mode: str = "continuo",
             contracts: int = 1, stop: float = STOP, target: float = TARGET,
             cost: float = COST_POINTS, balance: float = ACCOUNT_USD,
             rng=None, levels_table: pd.DataFrame = None,
             signals: dict = None) -> dict:
    """Uma janela, uma posicao de cada vez, sequencialmente.

    Sequencial e nao vectorizado de proposito: uma conta real nao pode ter
    quarenta posicoes abertas ao mesmo tempo, e e' a impossibilidade de as
    sobrepor que limita quantos negocios cabem em quatro horas."""
    rng = rng or np.random.default_rng(0)
    close = frame["close"].to_numpy(float)

    ready = set()
    if mode == "sinal":
        if levels_table is None:
            levels_table = T.levels(frame, 10, 10)
        window = levels_table[(levels_table["ready_index"] >= lo - 500)
                              & (levels_table["ready_index"] <= hi)]
        touch = T.entries(frame, window, rule="fade", lag=0)
        ready = {int(v) for v in touch["entry_index"].to_numpy()}
        sides = dict(zip(touch["entry_index"].to_numpy(),
                         touch["side"].to_numpy()))

    trades, equity = [], balance
    i = lo
    while i < hi:
        if mode == "continuo":
            side = int(rng.choice([-1, 1]))
        elif mode == "sinal":
            if i not in ready:
                i += 1
                continue
            side = int(sides[i])
        elif mode in ("rsi_bracket", "rsi_connors"):
            if i not in signals:
                i += 1
                continue
            side = int(signals[i])
        elif mode == "acaso_connors":
            side = int(rng.choice([-1, 1]))
        else:
            raise ValueError(f"modo desconhecido: {mode}")

        if equity < DAY_MARGIN * contracts:      # sem margem, acabou
            break

        if mode in ("rsi_connors", "acaso_connors"):
            gross, held = resolve_connors(frame, i, side, hi,
                                          max_bars=min(120, hi - i))
        else:
            result = T.resolve(frame, np.array([i]), np.array([close[i]]),
                               np.array([side]), stop, target,
                               max_bars=min(120, hi - i), model="honesta",
                               start_offset=1)
            gross = float(result["gross"].iloc[0])
            held = int(result["bars_held"].iloc[0])
        net_usd = (gross - cost) * POINT_VALUE * contracts
        equity += net_usd
        trades.append({"gross": gross, "net_usd": net_usd, "won": gross > 0})
        i += max(held, 1)

    table = pd.DataFrame(trades)
    if table.empty:
        return {"n": 0, "fim": balance}
    return {
        "n": len(table),
        "session": frame["session"].to_numpy()[lo],
        "acerto": float(table["won"].mean()),
        "bruto_usd": float((table["gross"] * POINT_VALUE * contracts).sum()),
        "custos_usd": float(len(table) * cost * POINT_VALUE * contracts),
        "liquido_usd": float(table["net_usd"].sum()),
        "fim": equity,
        "retorno": equity / balance - 1.0,
        "arruinado": equity < DAY_MARGIN * contracts,
    }


def run_all(mode: str = "continuo", contracts: int = 1, seed: int = 19,
            **kwargs) -> pd.DataFrame:
    conn = database.get_connection()
    frame = T.load(conn)
    frame = frame[frame.groupby("session")["close"].transform("size") > 200]
    frame = T.add_session(frame.reset_index(drop=True))
    table = T.levels(frame, 10, 10) if mode == "sinal" else None
    signals = None
    if mode in ("rsi_bracket", "rsi_connors"):
        frame = rsi_frame(frame)
        signals = rsi_signals(frame, short_side=kwargs.pop("short_side", False))
    elif mode == "acaso_connors":
        frame = rsi_frame(frame)

    rng = np.random.default_rng(seed)
    rows = []
    for lo, hi in windows(frame):
        rows.append(simulate(frame, lo, hi, mode, contracts, rng=rng,
                             levels_table=table, signals=signals, **kwargs))
    return pd.DataFrame([r for r in rows if r.get("n", 0) > 0])


def if_the_claim_were_true(win_rate: float = 0.89, n_trades: int = 240,
                           contracts: int = 1, stop: float = STOP,
                           target: float = TARGET, cost: float = COST_POINTS
                           ) -> dict:
    """Nao e' simulacao, e' uma conta. O que daria SE 89% fosse verdade.

    Existe para mostrar que o plano nao e' absurdo em aritmetica -- com 89% de
    acerto isto daria mesmo dinheiro, e bastante. O que falha e' a premissa."""
    edge = win_rate * target - (1 - win_rate) * stop - cost
    total = edge * POINT_VALUE * contracts * n_trades
    return {"acerto": win_rate, "por_negocio_pts": edge,
            "por_negocio_usd": edge * POINT_VALUE * contracts,
            "dia_usd": total, "dia_pct": total / ACCOUNT_USD}


def summarise(result: pd.DataFrame, label: str) -> dict:
    if result.empty:
        return {"caso": label, "janelas": 0}
    return {
        "caso": label,
        "janelas": len(result),
        "negocios": float(result["n"].mean()),
        "acerto": float((result["acerto"] * result["n"]).sum() / result["n"].sum()),
        "bruto_usd": float(result["bruto_usd"].mean()),
        "custos_usd": float(result["custos_usd"].mean()),
        "liquido_usd": float(result["liquido_usd"].mean()),
        "mediana_usd": float(result["liquido_usd"].median()),
        "pior_usd": float(result["liquido_usd"].min()),
        "melhor_usd": float(result["liquido_usd"].max()),
        "janelas_positivas": float((result["liquido_usd"] > 0).mean()),
    }


def compound(result: pd.DataFrame, sessions: int = 252,
             balance: float = ACCOUNT_USD, draws: int = 2000,
             seed: int = 19) -> dict:
    """E se se fizesse isto todos os dias durante um ano?

    Reamostra JANELAS REAIS com reposicao em vez de assumir uma distribuicao.
    O retorno de cada dia aplica-se ao saldo do dia anterior, que e' o que
    transforma uma perda media pequena numa ruina quase certa."""
    if result.empty:
        return {}
    returns = result["retorno"].to_numpy(float)
    rng = np.random.default_rng(seed)
    ends, ruined = np.empty(draws), 0
    for draw in range(draws):
        equity = balance
        for _ in range(sessions):
            equity *= 1.0 + rng.choice(returns)
            if equity < DAY_MARGIN:
                ruined += 1
                equity = 0.0
                break
        ends[draw] = equity
    return {"mediana_fim": float(np.median(ends)),
            "p5": float(np.quantile(ends, 0.05)),
            "p95": float(np.quantile(ends, 0.95)),
            "prob_ruina": ruined / draws,
            "prob_lucro": float((ends > balance).mean())}


def run() -> dict:
    print(f"Conta: {ACCOUNT_EUR:.0f} EUR = {ACCOUNT_USD:.0f} USD "
          f"(EUR/USD {EUR_USD})")
    print(f"MNQ: {POINT_VALUE:.0f} USD/ponto, bracket {STOP:.0f}/{TARGET:.0f}, "
          f"custo {COST_POINTS} pontos = {COST_POINTS * POINT_VALUE:.2f} USD/negocio")
    print(f"Risco por negocio com 1 contrato: "
          f"{STOP * POINT_VALUE:.0f} USD = {STOP * POINT_VALUE / ACCOUNT_USD * 100:.1f}% da conta\n")

    print("1. A ARITMETICA DOS CUSTOS, ANTES DE SIMULAR SEJA O QUE FOR")
    for n in (60, 120, 240):
        drag = n * COST_POINTS * POINT_VALUE
        print(f"   {n:3d} negocios em 4h -> {drag:6.0f} USD de custos = "
              f"{drag / ACCOUNT_USD * 100:5.1f}% da conta, por dia")
    print()

    print("2. SE OS 89% FOSSEM VERDADE (conta, nao simulacao)")
    for rate in (0.89, 0.8333, 0.82):
        claim = if_the_claim_were_true(rate)
        print(f"   acerto {rate:.4f} -> {claim['por_negocio_usd']:+6.2f} USD/negocio, "
              f"{claim['dia_usd']:+8.0f} USD/dia = {claim['dia_pct'] * 100:+7.1f}%")
    print("   (83.33% e' a entrada ao acaso; 82% foi o que a regra deu medida honestamente)\n")

    print("3. A SIMULACAO SOBRE JANELAS REAIS DE 4 HORAS (MNQ ao minuto)")
    out, rows = {}, []
    for mode, contracts, label in (("continuo", 1, "ao minuto, 1 contrato"),
                                   ("continuo", 2, "ao minuto, 2 contratos"),
                                   ("sinal", 1, "so' no sinal, 1 contrato")):
        result = run_all(mode, contracts)
        out[label] = result
        rows.append(summarise(result, label))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()

    print("4. E SE SE FIZESSE ISTO TODOS OS DIAS DURANTE UM ANO")
    for label, result in out.items():
        year = compound(result)
        if year:
            print(f"   {label:26s} mediana {year['mediana_fim']:8.0f} USD, "
                  f"p5 {year['p5']:7.0f}, p95 {year['p95']:8.0f}, "
                  f"ruina {year['prob_ruina'] * 100:5.1f}%, "
                  f"lucro {year['prob_lucro'] * 100:5.1f}%")
    out["resumo"] = table
    return out


if __name__ == "__main__":
    run()


# ---------------------------------------------------------------------------
# Um ano inteiro a esta cadencia
# ---------------------------------------------------------------------------

def year(result: pd.DataFrame, sessions: int = 252, sizing: str = "fixo",
         balance: float = ACCOUNT_USD, draws: int = 5000, seed: int = 20,
         by_day: bool = True) -> dict:
    """Um ano de sessoes, reamostrando janelas reais.

    `sizing` E' A CORRECCAO QUE ESTA FUNCAO EXISTE PARA FAZER. A ronda 19
    compos retornos percentuais, o que pressupoe que se ajusta o tamanho da
    posicao ao saldo. Com MNQ o minimo e' UM CONTRATO INTEIRO: a perda em
    dolares nao encolhe quando a conta encolhe, portanto a conta desce em LINHA
    RECTA e nao geometricamente, e chega ao fim mais depressa do que o
    composto sugeria.

      fixo         -- 1 contrato sempre. E' o que se pode fazer de facto.
      proporcional -- escala com o saldo. Nao e' executavel com um contrato
                      minimo; serve so' para mostrar quanto da diferenca vem da
                      indivisibilidade.

    `by_day=True` REAMOSTRA DIAS E NAO JANELAS. Sao 99 janelas mas apenas 20
    dias distintos, e janelas do mesmo dia partilham regime e volatilidade.
    Tratar 99 observacoes como independentes estreitaria o intervalo pelo mesmo
    erro da ronda 7."""
    if result.empty:
        return {}
    rng = np.random.default_rng(seed)

    if by_day and "session" in result.columns:
        groups = [part.index.to_numpy() for _, part in result.groupby("session")]
    else:
        groups = [np.array([i]) for i in result.index]

    pnl = result["liquido_usd"].to_numpy(float)
    ret = result["retorno"].to_numpy(float)
    ends = np.empty(draws)
    ruin_at = []

    for draw in range(draws):
        equity, ruined = balance, False
        for step in range(sessions):
            pool = groups[rng.integers(len(groups))]
            pick = pool[rng.integers(len(pool))]
            if sizing == "fixo":
                equity += pnl[pick]
            elif sizing == "proporcional":
                equity *= 1.0 + ret[pick]
            else:
                raise ValueError(f"sizing desconhecido: {sizing}")
            if equity < DAY_MARGIN:
                ruin_at.append(step + 1)
                equity, ruined = 0.0, True
                break
        ends[draw] = equity

    return {
        "sizing": sizing,
        "mediana_fim": float(np.median(ends)),
        "p5": float(np.quantile(ends, 0.05)),
        "p95": float(np.quantile(ends, 0.95)),
        "prob_ruina": len(ruin_at) / draws,
        "sessoes_ate_ruina": float(np.median(ruin_at)) if ruin_at else np.nan,
        "prob_lucro": float((ends > balance).mean()),
    }


# ---------------------------------------------------------------------------
# O RSI(2) ao minuto -- a regra da ronda 18 no instrumento da ronda 17
# ---------------------------------------------------------------------------
#
# PORQUE VALE A PENA. A ronda 18 encontrou no RSI(2) um efeito bruto REAL em
# barras diarias: +0.51% por negocio contra +0.27% de entradas ao acaso no
# mesmo poco de recuos, com intervalos que nao se tocam. A ronda 17 nao
# encontrou efeito nenhum no bracket de pivots. Transplantar o RSI(2) para o
# minuto pergunta se o efeito da ronda 18 e' uma propriedade do prazo diario
# ou reversao a media em geral.
#
# E separa duas coisas que o pedido confunde: trocar o SINAL DE ENTRADA e
# trocar a REGRA DE SAIDA. Por isso correm-se as duas versoes --
#   rsi_bracket -- entrada por RSI(2), saida pelo bracket 25/5 da ronda 17.
#                  Isola o sinal de entrada, com tudo o resto igual.
#   rsi_connors -- entrada por RSI(2), saida de Connors (fecho acima da media
#                  rapida), sem stop. E' a regra verdadeira do livro.
#
# A media de 200 BARRAS ao minuto sao 3h20. Calcula-se DENTRO da sessao, como
# tudo neste projecto desde a ronda 10: uma media que atravessasse a noite
# misturaria o fecho de ontem com a abertura de hoje e o salto overnight
# apareceria como sinal.

RSI_PERIOD, RSI_FAST, RSI_SLOW = 2, 5, 200
RSI_THRESHOLD = 10.0


def rsi_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Indicadores de Connors calculados sessao a sessao."""
    from src.research import connors_rsi2 as U

    parts = []
    for _, part in frame.groupby("session", sort=True):
        part = part.copy()
        close = part["close"].to_numpy(float)
        part["ma_fast"] = part["close"].rolling(RSI_FAST).mean()
        part["ma_slow"] = part["close"].rolling(RSI_SLOW).mean()
        part["rsi"] = U.wilder_rsi(close, RSI_PERIOD)
        parts.append(part)
    return pd.concat(parts).sort_index()


def rsi_signals(frame: pd.DataFrame, short_side: bool = False) -> dict:
    """Barras de entrada -> sentido. Longo e' a regra do livro; o curto e' o
    espelho (abaixo da media lenta, acima da rapida, RSI acima de 100-limiar),
    e existe para o resultado nao depender de o indice ter subido na amostra."""
    ok = frame[["ma_fast", "ma_slow", "rsi"]].notna().all(axis=1).to_numpy()
    close = frame["close"].to_numpy(float)
    fast = frame["ma_fast"].to_numpy(float)
    slow = frame["ma_slow"].to_numpy(float)
    rsi = frame["rsi"].to_numpy(float)

    longs = ok & (close > slow) & (close < fast) & (rsi < RSI_THRESHOLD)
    out = {int(i): 1 for i in np.flatnonzero(longs)}
    if short_side:
        shorts = ok & (close < slow) & (close > fast) & (rsi > 100 - RSI_THRESHOLD)
        out.update({int(i): -1 for i in np.flatnonzero(shorts)})
    return out


def resolve_connors(frame: pd.DataFrame, index: int, side: int, hi: int,
                    max_bars: int = 120) -> tuple:
    """Saida de Connors: sai quando o fecho volta acima da media rapida (ou
    abaixo, se estiver vendido). Sem stop -- Connors recomenda explicitamente
    nao o usar, e e' por isso que a perda media dele e' maior do que o ganho.

    Fica limitado pelo fim da janela e pela sessao: nao se deixa uma posicao
    sem stop atravessar a noite num futuro alavancado."""
    close = frame["close"].to_numpy(float)
    fast = frame["ma_fast"].to_numpy(float)
    session = frame["session"].to_numpy()
    stop_at = min(index + max_bars, hi, len(frame) - 1)

    entry = close[index]
    last = index
    for step in range(index + 1, stop_at + 1):
        # A fronteira de sessao apanhou-me duas vezes. A versao anterior fazia
        # `break` aqui mas depois saia em close[stop_at], que ja' era da sessao
        # seguinte -- o negocio arrecadava o salto overnight inteiro, 410
        # pontos no teste que o apanhou. E' o mesmo erro da ronda 17, em codigo
        # novo: nao basta parar o ciclo, tem de se guardar a ultima barra
        # VALIDA e sair nessa.
        if session[step] != session[index]:
            break
        last = step
        if np.isnan(fast[step]):
            continue
        recovered = close[step] > fast[step] if side > 0 else close[step] < fast[step]
        if recovered:
            return side * (close[step] - entry), step - index
    return side * (close[last] - entry), last - index
