"""EXPERIENCIA W -- a Polymarket esta' calibrada?

DE ONDE VEM. Chegou-me uma ferramenta que promete "engenharia inversa" de bots
da Polymarket e responder se cada um foi pericia ou sorte. A pergunta que ela
nao faz e' a unica que decide: **o mercado esta' calibrado?**

Num mercado binario compra-se a `p` e recebe-se 1 dolar. Logo o ponto morto E'
O PRECO: comprar a 90 centimos exige 90% de acerto para empatar. E' a mesma
aritmetica da ronda 17 -- S/(S+A) no bracket de futuros -- mas aqui o mercado
IMPRIME o ponto morto no ecra. Se os contratos a 80 centimos resolverem a favor
80% das vezes, a esperanca e' exactamente zero para toda a gente antes de
taxas, e todo o lucro de qualquer bot e' desvio ou sorte. Se nao resolverem, a
diferenca e' a unica coisa negociavel que existe ali.

Portanto nao se testa um bot. Testa-se o MERCADO, e os bots passam a ser um
caso particular.

DUAS PREVISOES QUE A LITERATURA FAZ, e que esta experiencia pode separar:

  1. VIES FAVORITO-AZARAO. Em mercados de apostas, os azaroes tendem a estar
     caros e os favoritos baratos. Se existir aqui, a regra do coinman2 -- "so'
     comprar entre 20 e 95 centimos, porque abaixo de 20 perdeu dinheiro" --
     deixa de ser um filtro escolhido depois de ver o resultado e passa a ser a
     captura de um fenomeno documentado. E' a leitura GENEROSA da regra, e tem
     de ser testada a serio em vez de assumida num sentido ou no outro.

  2. CALIBRACAO PERFEITA. Mercados liquidos e de resolucao rapida, com milhares
     de participantes, sao o caso onde a teoria preve calibracao quase exacta.
     Os mercados de 5 minutos de Bitcoin sao o exemplo extremo: resolvem-se
     288 vezes por dia e qualquer desvio sistematico seria arbitrado em horas.

TRES ARMADILHAS, e a primeira invalidaria tudo:

  A. A UNIDADE DE OBSERVACAO NAO E' O NEGOCIO, E' O MERCADO. Milhares de
     negocios acontecem no mesmo mercado e esse mercado resolve-se UMA vez.
     Tratar negocios como independentes estreitaria o intervalo por um factor
     enorme -- e' o erro das janelas sobrepostas da ronda 7 na sua forma mais
     agressiva. Todo o intervalo aqui vem de reamostrar MERCADOS.

  B. COMPRA E VENDA NAO SE SOMAM EM BRUTO. Vender o "Up" a 0.30 nao e' apostar
     a 0.30 -- e' comprar o "Down" a 0.70. Sem normalizar, metade da amostra
     entra com o preco trocado e a curva de calibracao fica simetrica por
     construcao, que e' exactamente o resultado que se procura.

  C. A AMOSTRA E' DE SOBREVIVENTES AO CONTRARIO. Estes mercados sao os que a
     Polymarket ainda corre. Mas ao contrario das accoes da ronda 8, aqui isso
     ajuda: queremos saber se ha' dinheiro em cima da mesa HOJE, nao em 2021.
"""

import sqlite3
import sys
import time
import urllib.request
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

import numpy as np
import pandas as pd

GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
PAUSE = 0.28              # segundos entre pedidos; a API devolve 429 sem isto

# Familias de mercado enumeraveis por slug. O passo e' a duracao do mercado,
# por isso os slugs sao previsiveis para tras no tempo -- e' o que torna
# possivel construir uma amostra historica sem conta nem chave.
FAMILIES = {
    "btc5m": {"prefix": "btc-updown-5m-", "step": 300},
    "eth5m": {"prefix": "eth-updown-5m-", "step": 300},
    "btc15m": {"prefix": "btc-updown-15m-", "step": 900},
    "eth15m": {"prefix": "eth-updown-15m-", "step": 900},
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS pm_markets (
    condition_id TEXT PRIMARY KEY,
    slug         TEXT NOT NULL,
    family       TEXT NOT NULL,
    end_epoch    INTEGER NOT NULL,
    winner       TEXT,
    closed       INTEGER NOT NULL,
    fetched_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pm_trades (
    condition_id TEXT NOT NULL,
    wallet       TEXT NOT NULL,
    side         TEXT NOT NULL,
    outcome      TEXT NOT NULL,
    price        REAL NOT NULL,
    size         REAL NOT NULL,
    ts           INTEGER NOT NULL,
    tx           TEXT NOT NULL,
    PRIMARY KEY (tx, wallet, outcome, price, size)
);
CREATE INDEX IF NOT EXISTS idx_pm_trades_cid ON pm_trades(condition_id);
"""


# ---------------------------------------------------------------------------
# Recolha
# ---------------------------------------------------------------------------

def fetch(url: str, tries: int = 4):
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "research"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except Exception as error:
            text = str(error)
            if "429" in text:
                time.sleep(1.5 * (attempt + 1))
                continue
            if "400" in text or "404" in text:
                return None
            if attempt == tries - 1:
                return None
            time.sleep(0.8 * (attempt + 1))
    return None


def market_from_slug(slug: str) -> dict:
    """O mercado de 5 minutos NAO esta' indexado em /markets?slug= -- devolve
    zero resultados. Esta' em /events?slug=, com o mercado la' dentro. Custou
    meia hora a descobrir e e' a unica porta que funciona."""
    events = fetch(f"{GAMMA}/events?slug={slug}")
    if not events:
        return {}
    markets = events[0].get("markets") or []
    if not markets:
        return {}
    market = markets[0]
    outcomes = json.loads(market.get("outcomes") or "[]")
    prices = json.loads(market.get("outcomePrices") or "[]")
    winner = None
    if len(outcomes) == len(prices) == 2 and market.get("closed"):
        values = [float(p) for p in prices]
        if max(values) > 0.99 and min(values) < 0.01:
            winner = outcomes[int(np.argmax(values))]
    return {"condition_id": market.get("conditionId"), "slug": slug,
            "winner": winner, "closed": bool(market.get("closed"))}


def trades_for(condition_id: str, cap: int = 4000) -> list:
    """A API devolve os negocios MAIS RECENTES PRIMEIRO. Com um limite baixo
    guardava-se so' o fim de cada mercado -- medi 1500 de 2371 num mercado
    tipico, 63%, e todos agrupados nos ultimos segundos. Para uma curva de
    calibracao isso trocaria "o mercado esta' calibrado" por "os ultimos
    segundos estao calibrados", que e' outra pergunta. O limite cobre agora o
    mercado inteiro."""
    out, offset = [], 0
    while offset < cap:
        page = fetch(f"{DATA}/trades?market={condition_id}&limit=500&offset={offset}")
        if not page:
            break
        out.extend(page)
        if len(page) < 500:
            break
        offset += 500
        time.sleep(PAUSE)
    return out


def connect(path=None):
    conn = sqlite3.connect(path or config.DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def ingest(conn, family: str = "btc5m", markets: int = 500,
           skip_recent: int = 4, verbose: bool = True) -> dict:
    """Enumera slugs para tras no tempo e guarda mercado + negocios.

    `skip_recent` salta os mercados mais recentes, que ainda nao resolveram --
    um mercado por resolver entraria com `winner` nulo e seria descartado mais
    tarde, mas gastaria dois pedidos."""
    spec = FAMILIES[family]
    now = int(time.time())
    base = (now // spec["step"]) * spec["step"] - spec["step"] * skip_recent
    stamp = pd.Timestamp.utcnow().isoformat()

    known = {r[0] for r in conn.execute(
        "SELECT slug FROM pm_markets WHERE family = ?", (family,))}
    saved_markets = saved_trades = 0

    for step in range(markets):
        epoch = base - spec["step"] * step
        slug = f"{spec['prefix']}{epoch}"
        if slug in known:
            continue
        info = market_from_slug(slug)
        time.sleep(PAUSE)
        if not info or not info.get("condition_id"):
            continue
        conn.execute(
            "INSERT OR REPLACE INTO pm_markets "
            "(condition_id, slug, family, end_epoch, winner, closed, fetched_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (info["condition_id"], slug, family, epoch, info["winner"],
             int(info["closed"]), stamp))
        saved_markets += 1

        if info["winner"]:
            rows = trades_for(info["condition_id"])
            time.sleep(PAUSE)
            conn.executemany(
                "INSERT OR IGNORE INTO pm_trades "
                "(condition_id, wallet, side, outcome, price, size, ts, tx) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [(info["condition_id"], t.get("proxyWallet", ""), t.get("side", ""),
                  t.get("outcome", ""), float(t.get("price", 0)),
                  float(t.get("size", 0)), int(t.get("timestamp", 0)),
                  t.get("transactionHash", "")) for t in rows])
            saved_trades += len(rows)

        if saved_markets % 25 == 0 and saved_markets and verbose:
            conn.commit()
            print(f"  {family}: {saved_markets} mercados, {saved_trades} negocios",
                  flush=True)
    conn.commit()
    return {"mercados": saved_markets, "negocios": saved_trades}


# ---------------------------------------------------------------------------
# Normalizacao -- a armadilha B
# ---------------------------------------------------------------------------

def load(conn, family: str = None) -> pd.DataFrame:
    """Cada negocio vira uma APOSTA: um preco efectivo e um resultado binario.

    Vender o "Up" a 0.30 nao e' apostar a 0.30, e' comprar o "Down" a 0.70. Sem
    isto metade da amostra entra com o preco trocado e a curva de calibracao
    sai simetrica por construcao -- isto e', sai exactamente o resultado que se
    esta' a tentar medir."""
    sql = ("SELECT t.*, m.winner, m.family, m.end_epoch FROM pm_trades t "
           "JOIN pm_markets m ON m.condition_id = t.condition_id "
           "WHERE m.winner IS NOT NULL")
    params = ()
    if family:
        sql += " AND m.family = ?"
        params = (family,)
    frame = pd.read_sql_query(sql, conn, params=params)
    if frame.empty:
        return frame

    buy = frame["side"].str.upper() == "BUY"
    frame["stake_price"] = np.where(buy, frame["price"], 1.0 - frame["price"])
    hit = frame["outcome"] == frame["winner"]
    frame["won"] = np.where(buy, hit, ~hit)
    frame["dollars"] = frame["size"] * frame["stake_price"]
    # Lucro bruto por dolar arriscado: ganha (1-p)/p, perde tudo.
    frame["profit"] = np.where(frame["won"],
                               frame["size"] * (1.0 - frame["stake_price"]),
                               -frame["size"] * frame["stake_price"])
    return frame[(frame["stake_price"] > 0.0) & (frame["stake_price"] < 1.0)]


# ---------------------------------------------------------------------------
# Calibracao
# ---------------------------------------------------------------------------

BUCKETS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 1.0)


def calibration(frame: pd.DataFrame, buckets=BUCKETS,
                weight: str = "igual") -> pd.DataFrame:
    """Preco pago contra frequencia realizada. Calibrado = os dois coincidem.

    `weight="dolar"` pondera pelo dinheiro arriscado, que e' o que determina o
    P&L; `"igual"` pondera por negocio. Reportam-se os dois porque divergirem
    significaria que o dinheiro grande se comporta de forma diferente."""
    work = frame.copy()
    work["faixa"] = pd.cut(work["stake_price"], bins=list(buckets),
                           include_lowest=True)
    if weight == "dolar":
        grouped = work.groupby("faixa", observed=True)
        table = grouped.apply(lambda g: pd.Series({
            "n": len(g),
            "mercados": g["condition_id"].nunique(),
            "preco_medio": np.average(g["stake_price"], weights=g["dollars"]),
            "frequencia": np.average(g["won"], weights=g["dollars"]),
        }), include_groups=False).reset_index()
    else:
        table = work.groupby("faixa", observed=True).agg(
            n=("won", "size"), mercados=("condition_id", "nunique"),
            preco_medio=("stake_price", "mean"),
            frequencia=("won", "mean")).reset_index()
    table["desvio"] = table["frequencia"] - table["preco_medio"]
    return table


def market_bootstrap(frame: pd.DataFrame, column: str = "desvio_unitario",
                     draws: int = 4000, seed: int = 22) -> tuple:
    """Reamostra MERCADOS -- a armadilha A.

    Milhares de negocios partilham UMA resolucao. Reamostrar negocios trataria
    cada um como uma observacao independente de uma moeda que so' foi lancada
    uma vez por mercado."""
    if frame.empty:
        return (np.nan, np.nan, np.nan)
    grouped = frame.groupby("condition_id")[column]
    sums = grouped.sum().to_numpy(float)
    counts = grouped.size().to_numpy(float)
    rng = np.random.default_rng(seed)
    total = len(sums)
    means = np.empty(draws)
    chunk = max(1, min(draws, 2_000_000 // max(total, 1)))
    done = 0
    while done < draws:
        size = min(chunk, draws - done)
        picked = rng.integers(0, total, (size, total))
        means[done:done + size] = sums[picked].sum(1) / counts[picked].sum(1)
        done += size
    return (float(frame[column].mean()), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def edge_by_band(frame: pd.DataFrame, bands=((0.0, 0.20), (0.20, 0.95),
                                             (0.95, 1.0))) -> pd.DataFrame:
    """A regra do coinman2 posta a' prova: comprar so' entre 20 e 95 centimos.

    Se o desvio for negativo abaixo de 20 e nao-negativo dentro da banda, a
    regra captura um fenomeno real. Se for indistinguivel de zero em todas as
    bandas, a regra foi escolhida depois de ver o resultado."""
    work = frame.copy()
    work["desvio_unitario"] = work["won"].astype(float) - work["stake_price"]
    rows = []
    for low, high in bands:
        part = work[(work["stake_price"] >= low) & (work["stake_price"] < high)]
        if part.empty:
            continue
        mean, lo, hi = market_bootstrap(part)
        rows.append({"banda": f"{low:.2f}-{high:.2f}", "n": len(part),
                     "mercados": part["condition_id"].nunique(),
                     "preco_medio": float(part["stake_price"].mean()),
                     "frequencia": float(part["won"].mean()),
                     "desvio": mean, "ic_lo": lo, "ic_hi": hi})
    return pd.DataFrame(rows)


def topup_trades(conn, cap: int = 4000, verbose: bool = True) -> int:
    """Completa os negocios de mercados ja' guardados com um limite maior.

    Existe porque a primeira recolha usou um limite que truncava o inicio de
    cada mercado. `INSERT OR IGNORE` torna isto idempotente."""
    rows = conn.execute(
        "SELECT condition_id FROM pm_markets WHERE winner IS NOT NULL").fetchall()
    added = 0
    for index, (condition_id,) in enumerate(rows, 1):
        trades = trades_for(condition_id, cap)
        time.sleep(PAUSE)
        before = conn.execute(
            "SELECT COUNT(*) FROM pm_trades WHERE condition_id = ?",
            (condition_id,)).fetchone()[0]
        conn.executemany(
            "INSERT OR IGNORE INTO pm_trades "
            "(condition_id, wallet, side, outcome, price, size, ts, tx) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [(condition_id, t.get("proxyWallet", ""), t.get("side", ""),
              t.get("outcome", ""), float(t.get("price", 0)),
              float(t.get("size", 0)), int(t.get("timestamp", 0)),
              t.get("transactionHash", "")) for t in trades])
        after = conn.execute(
            "SELECT COUNT(*) FROM pm_trades WHERE condition_id = ?",
            (condition_id,)).fetchone()[0]
        added += after - before
        if index % 25 == 0:
            conn.commit()
            if verbose:
                print(f"  completados {index}/{len(rows)}, +{added} negocios",
                      flush=True)
    conn.commit()
    return added


# ---------------------------------------------------------------------------
# Pericia ou sorte, por carteira
# ---------------------------------------------------------------------------

def wallet_skill(frame: pd.DataFrame, min_markets: int = 20) -> pd.DataFrame:
    """A pergunta que a ferramenta promete responder, feita a serio.

    Sob a hipotese nula -- mercado calibrado -- cada posicao tem esperanca ZERO
    e variancia conhecida. Uma posicao de `a` contratos no resultado A e `b` no
    resultado B, por um custo `c`, rende (a - c) se A sair e (b - c) se B sair.
    Logo a variancia e' p(1-p)(a-b)^2, com p = P(A).

    ESCREVER A VARIANCIA EM CONTRATOS, E NAO NUM PRECO MEDIO, E' O QUE TORNA
    ISTO CORRECTO. Uma carteira que compra Up E Down -- a estrategia do
    gabagool22 -- tem a = b, portanto variancia ZERO: nao esta' a apostar, esta'
    a arbitrar, e qualquer lucro dela e' margem e nao sorte. Um preco medio
    nunca poderia exprimir isso, e trataria um arbitragista como um apostador
    com muita sorte.

    A AGREGACAO E' POR CARTEIRA-MERCADO: cinquenta negocios da mesma carteira
    no mesmo mercado partilham UMA resolucao e nao sao cinquenta apostas
    independentes. Sem isto a variancia vem subestimada e toda a gente parece
    genial."""
    work = frame.copy()
    buy = work["side"].str.upper() == "BUY"
    # Contratos assinados no resultado nomeado pelo negocio, e dinheiro gasto.
    work["contracts"] = np.where(buy, work["size"], -work["size"])
    work["cash"] = np.where(buy, work["size"] * work["price"],
                            -work["size"] * work["price"])
    work["is_winner_side"] = work["outcome"] == work["winner"]

    keys = ["wallet", "condition_id"]
    per = work.groupby(keys, sort=False).agg(cash=("cash", "sum"),
                                             stake=("dollars", "sum"))
    # Contratos do lado que ganhou e do que perdeu, por posicao.
    sides = work.pivot_table(index=keys, columns="is_winner_side",
                             values="contracts", aggfunc="sum", fill_value=0.0)
    per["a"] = sides[True] if True in sides.columns else 0.0
    per["b"] = sides[False] if False in sides.columns else 0.0
    per["profit"] = per["a"] - per["cash"]

    # p vem do PROPRIO mercado: preco medio ponderado por volume do lado que
    # veio a ganhar. Usar o resultado seria circular; usar o preco nao e'.
    price_of_winner = (work[work["is_winner_side"] & buy]
                       .groupby("condition_id")
                       .apply(lambda g: np.average(
                           g["price"], weights=g["size"].clip(lower=1e-9)),
                           include_groups=False))
    per = per.reset_index()
    per["p"] = per["condition_id"].map(price_of_winner).fillna(0.5).clip(0.01, 0.99)
    per["variance"] = per["p"] * (1 - per["p"]) * (per["a"] - per["b"]) ** 2

    table = per.groupby("wallet").agg(
        mercados=("condition_id", "nunique"),
        lucro=("profit", "sum"),
        apostado=("stake", "sum"),
        variancia=("variance", "sum"))
    table = table[table["mercados"] >= min_markets].copy()
    table["risco"] = (np.sqrt(table["variancia"])
                      / table["apostado"].clip(lower=1e-9))
    table["z"] = table["lucro"] / np.sqrt(table["variancia"].clip(lower=1e-9))
    table["retorno"] = table["lucro"] / table["apostado"].clip(lower=1e-9)
    return table.sort_values("z", ascending=False).reset_index()


ARBITRAGE_RISK = 0.02     # desvio-padrao por dolar apostado


def split_by_risk(table: pd.DataFrame, cutoff: float = ARBITRAGE_RISK) -> tuple:
    """Separa arbitragistas de apostadores, e NAO e' uma limpeza de dados.

    Uma carteira coberta tem variancia perto de zero, portanto o seu z explode
    -- medi 97 440 numa delas. Isso nao e' um erro: uma arbitragem verdadeira
    TEM mesmo razao de Sharpe infinita. Mas o teste de sorte assume uma nula
    normal, e misturar as duas populacoes torna-o sem sentido nos dois
    sentidos: os arbitragistas inflacionam a cauda e, ao faze-lo, escondem que
    os apostadores nao a inflacionam de todo.

    Sao duas perguntas diferentes e tem de ser respondidas em separado:
      - os apostadores batem o mercado? (teste de z contra a normal)
      - os arbitragistas ainda tem margem? (lucro sobre capital, sem risco)"""
    arbitrage = table[table["risco"] < cutoff]
    punters = table[table["risco"] >= cutoff]
    return punters.reset_index(drop=True), arbitrage.reset_index(drop=True)


def luck_baseline(table: pd.DataFrame, threshold: float = 2.0) -> dict:
    """Quantas carteiras pareceriam habilidosas se ninguem o fosse.

    Sob a nula os z sao aproximadamente normais padrao, logo a fraccao acima de
    `threshold` e' conhecida de antemao. Comparar a contagem observada com essa
    expectativa e' o teste inteiro -- e' o mesmo raciocinio do varredor de
    sorte da ronda 17, aplicado a pessoas em vez de celulas de uma grelha."""
    from scipy import stats

    if table.empty:
        return {}
    n = len(table)
    expected = n * (1 - stats.norm.cdf(threshold))
    observed = int((table["z"] > threshold).sum())
    return {"carteiras": n, "limiar": threshold,
            "esperadas_por_acaso": expected, "observadas": observed,
            "racio": observed / expected if expected > 0 else np.nan,
            "z_medio": float(table["z"].mean()),
            "z_maximo": float(table["z"].max())}


# ---------------------------------------------------------------------------
# A arbitragem do gabagool22 ainda existe?
# ---------------------------------------------------------------------------

def pair_arbitrage(frame: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """Compra-se "Up" e "Down" ao mesmo tempo: se o par custar menos de 1
    dolar, o lucro e' certo e nao depende do preco do Bitcoin.

    Mede-se a partir de negocios EXECUTADOS, nao de cotacoes: se no mesmo
    segundo alguem comprou Up a 0.40 e outro comprou Down a 0.55, entao um par
    custava 0.95 e havia 5 centimos em cima da mesa. E' a estrategia do
    gabagool22 testada directamente nos dados publicos, em vez de aceite a
    partir da descricao."""
    buys = frame[frame["side"].str.upper() == "BUY"].copy()
    if buys.empty:
        return pd.DataFrame()
    # LIMITE SUPERIOR, e nao uma estimativa. Toma o melhor preco de cada lado
    # dentro da janela, que podem ser dois negocios de momentos diferentes e
    # de contrapartes diferentes. Quem quisesse mesmo o par teria de acertar
    # nos dois. Serve para responder "havia dinheiro em cima da mesa", nao
    # "quanto e' que alguem teria apanhado".
    buys["bucket"] = buys["ts"] // window
    cheapest = buys.groupby(["condition_id", "bucket", "outcome"])["price"].min()
    wide = cheapest.unstack("outcome")
    if wide.shape[1] < 2:
        return pd.DataFrame()
    first, second = wide.columns[:2]
    pairs = wide.dropna(subset=[first, second]).copy()
    pairs["par"] = pairs[first] + pairs[second]
    pairs["margem"] = 1.0 - pairs["par"]
    return pairs.reset_index()


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------

def run(min_markets: int = 10) -> dict:
    conn = connect()
    frame = load(conn)
    if frame.empty:
        raise SystemExit("sem dados: correr ingest() primeiro")
    frame["desvio_unitario"] = frame["won"].astype(float) - frame["stake_price"]
    out = {}

    markets = frame["condition_id"].nunique()
    print(f"{len(frame)} negocios normalizados, {markets} mercados resolvidos, "
          f"{frame['wallet'].nunique()} carteiras")
    print("unidade de observacao = MERCADO (os negocios partilham a resolucao)\n")

    print("1. O MERCADO ESTA' CALIBRADO?")
    for weight in ("igual", "dolar"):
        table = calibration(frame, weight=weight)
        print(f"   -- ponderado por {weight} --")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        out[f"calibracao_{weight}"] = table

    mean, low, high = market_bootstrap(frame)
    print(f"\n   DESVIO GLOBAL {mean:+.5f}  IC95% [{low:+.5f}, {high:+.5f}]")
    out["desvio_global"] = (mean, low, high)

    print("\n2. A REGRA DO COINMAN2 (so' 20c a 95c)")
    bands = edge_by_band(frame)
    print(bands.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out["bandas"] = bands

    print("\n3. PERICIA OU SORTE")
    table = wallet_skill(frame, min_markets=min_markets)
    punters, arbitrage = split_by_risk(table)
    print(f"   {len(table)} carteiras com >={min_markets} mercados: "
          f"{len(punters)} apostadores, {len(arbitrage)} arbitragistas")
    print("   -- apostadores (assumem risco) --")
    for threshold in (1.0, 2.0, 3.0):
        result = luck_baseline(punters, threshold)
        if result:
            print(f"     z>{threshold:.0f}: observadas {result['observadas']:4d}, "
                  f"esperadas por acaso {result['esperadas_por_acaso']:6.1f}, "
                  f"racio {result['racio']:.2f}")
    if len(punters):
        print(f"     z medio {punters['z'].mean():+.3f} (nula: 0), "
              f"maximo {punters['z'].max():.2f}")
    if len(arbitrage):
        print("   -- arbitragistas (risco ~zero) --")
        print(f"     lucro total {arbitrage['lucro'].sum():+.2f} USD sobre "
              f"{arbitrage['apostado'].sum():.0f} apostados = "
              f"{arbitrage['lucro'].sum() / arbitrage['apostado'].sum() * 100:+.2f}%")
        print(f"     com lucro positivo: {int((arbitrage['lucro'] > 0).sum())} "
              f"de {len(arbitrage)}")
    out["apostadores"], out["arbitragistas"] = punters, arbitrage

    print("\n4. A ARBITRAGEM DO PAR AINDA EXISTE? (limite superior)")
    pairs = pair_arbitrage(frame)
    if len(pairs):
        print(f"   {len(pairs)} instantes com os dois lados negociados")
        print(f"   par medio {pairs['par'].mean():.4f} USD; "
              f"mediana da margem {pairs['margem'].median():+.4f}")
        for slack in (0.0, 0.01, 0.02, 0.05):
            share = float((pairs["margem"] > slack).mean())
            print(f"   par abaixo de {1 - slack:.2f}: {share * 100:5.1f}% dos instantes")
    out["pares"] = pairs
    return out


if __name__ == "__main__":
    run()
