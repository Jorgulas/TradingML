"""EXPERIENCIA X -- a sazonalidade de volatilidade da ronda 5 nestes mercados.

DE ONDE VEM. A ronda 5 e' o unico resultado inequivocamente positivo e duravel
de todo o programa: a volatilidade intradiaria tem uma FORMA por hora do dia
que varia menos de 12% por hora de mes para mes, amplitude 4,09x +- 0,45, e um
perfil estimado ha' doze meses serve tao bem como um estimado no mes passado.
Tudo o resto que encontramos ou nao sobreviveu aos custos ou nao sobreviveu aos
controlos. A ronda 22 deu-nos um banco de ensaio com poder a serio: mercados
binarios que resolvem 288 vezes por dia e imprimem a probabilidade no ecra.
Esta ronda cruza as duas coisas.

PORQUE ISTO NAO E' OBVIO, E PORQUE PODE NAO EXISTIR AQUI. O "J" da ronda 5 vem
dos leiloes de abertura e fecho das accoes americanas. Cripto negocia 24 horas
por dia, sete dias por semana, e nao tem leilao nenhum. A primeira pergunta nao
e' se o mercado paga pela sazonalidade -- e' se a sazonalidade EXISTE. Pode nao
existir, e nesse caso a ronda acaba com uma nao-replicacao honesta.

A ARITMETICA, que e' o que torna isto testavel. Num mercado "Up ou Down" de
cinco minutos o strike e' o preco no inicio da janela, portanto no instante
zero P(Up) = 0,5 PARA QUALQUER VOLATILIDADE e a sazonalidade e' irrelevante. A
meio da janela ja' nao: com o subjacente em `S_t`, strike `K` e `tau` segundos
por correr, o preco justo do digital e'

    p = Phi( (S_t - K) / (sigma * raiz(tau)) )

e aqui sigma manda em tudo. A sazonalidade entra pelo MEIO da janela e so' por
la'. Isso da' duas coisas raras neste projecto:

  UMA PREVISAO ASSINADA. Se o mercado usar uma volatilidade aproximadamente
  constante e a hora for de volatilidade ALTA, o |d| verdadeiro e' menor do que
  o mercado supoe, a probabilidade verdadeira esta' mais perto de 0,5, e os
  favoritos estao CAROS -- ganham menos vezes do que o preco diz. Numa hora de
  volatilidade baixa, o contrario. Portanto o erro de nitidez do mercado deve
  correlacionar-se NEGATIVAMENTE com a volatilidade realizada da hora. Nao e'
  "ha' alguma relacao": e' um sinal especifico, previsto antes de medir.

  UM PLACEBO DE GRACA, dado pela propria fisica. Perto de p = 0,5 o preco nao
  contem informacao sobre sigma nenhuma, porque Phi(0) = 0,5 para toda e
  qualquer volatilidade. Se o efeito aparecer tambem nessa banda, nao e'
  volatilidade -- e' composicao ou artefacto. Nao foi preciso inventar um
  controlo: ele estava la'.

ARMADILHAS, por ordem de gravidade:

  A. UMA PASSAGEM POR HORA NAO E' SAZONALIDADE. Os 1250 mercados da ronda 22
     cobrem 25 horas seguidas. Cada hora do dia aparece UMA vez, portanto medir
     "a hora 14" seria medir aquela tarde de 28 de Outubro. Esta ronda recolhe
     por amostragem estratificada -- um mercado a cada 30 minutos ao longo de
     semanas -- para que cada hora do dia tenha observacoes em dias diferentes.

  B. A UNIDADE DA CORRELACAO ENTRE HORAS E' O DIA, nao o mercado. As 24 horas
     do mesmo dia partilham o regime desse dia. O intervalo da correlacao vem
     de reamostrar DIAS; o intervalo de cada hora isolada vem de reamostrar
     mercados, como na ronda 22.

  C. COMPOSICAO. Se nas horas volateis se negociar mais perto de 0,5 ou mais
     perto do fim da janela, uma media simples por hora mistura o efeito com a
     mudanca de mistura. O erro de nitidez e' por isso padronizado: calcula-se
     dentro de cada celula (banda de preco x banda de tempo) e reagrega-se com
     os pesos globais, iguais para todas as horas.

  D. ERRO DE MEDIDA NA VOL IMPLICITA. Inverter a formula precisa de `S_t - K`,
     que so' tenho a resolucao de um minuto. O ruido intra-minuto entra no
     numerador e atenua tudo para zero. A vol implicita fica como mecanismo,
     com a atenuacao assumida; a previsao assinada nao precisa dela.
"""

import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

import numpy as np
import pandas as pd
from scipy import stats

CRYPTO_SCHEMA = """
CREATE TABLE IF NOT EXISTS crypto_bars (
    asset  TEXT NOT NULL,
    ts     INTEGER NOT NULL,
    open   REAL NOT NULL,
    high   REAL NOT NULL,
    low    REAL NOT NULL,
    close  REAL NOT NULL,
    PRIMARY KEY (asset, ts)
);
"""

# O subjacente de cada familia de mercado.
UNDERLYING = {"btc5m": "BTC-USD", "btc15m": "BTC-USD",
              "eth5m": "ETH-USD", "eth15m": "ETH-USD"}

STEP = {"btc5m": 300, "eth5m": 300, "btc15m": 900, "eth15m": 900}

# Bandas da armadilha C. A banda central e' o placebo: perto de 0,5 o preco
# nao diz nada sobre sigma.
PRICE_BANDS = (0.0, 0.15, 0.30, 0.44, 0.56, 0.70, 0.85, 1.0)
PLACEBO_BAND = (0.44, 0.56)
# Bandas de tempo por correr em FRACCAO da janela, nao em segundos: assim uma
# celula quer dizer o mesmo num mercado de 5 e de 15 minutos, e os 15 minutos
# nao caem todos na ultima banda.
TAU_BANDS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Minimos para inverter a formula do digital sem explodir.
MIN_EDGE = 0.06          # |p - 0,5| minimo; abaixo disto Phi^-1 ~ 0
MAX_EDGE = 0.44          # |p - 0,5| maximo; acima disto Phi^-1 -> infinito
MIN_TAU = 45             # segundos; abaixo disto raiz(tau) ~ 0


# ---------------------------------------------------------------------------
# O subjacente
# ---------------------------------------------------------------------------

def connect(path=None):
    conn = sqlite3.connect(path or config.DB_PATH, timeout=180)
    conn.executescript(CRYPTO_SCHEMA)
    return conn


def fetch_crypto(symbol: str = "BTC-USD", days: int = 30,
                 chunk: int = 7) -> pd.DataFrame:
    """O yfinance da' barras de um minuto apenas dos ultimos 30 dias e em
    pedacos de no maximo 8 dias, por isso pede-se em janelas de 7."""
    import yfinance as yf

    end = pd.Timestamp.utcnow().floor("min")
    pieces = []
    for step in range(0, days, chunk):
        stop = end - pd.Timedelta(days=step)
        start = stop - pd.Timedelta(days=min(chunk, days - step))
        frame = yf.download(symbol, interval="1m", start=start, end=stop,
                            auto_adjust=True, progress=False, threads=False)
        if frame is None or frame.empty:
            continue
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        pieces.append(frame[["Open", "High", "Low", "Close"]])
        time.sleep(0.4)

    if not pieces:
        return pd.DataFrame()
    bars = pd.concat(pieces)
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    bars.columns = ["open", "high", "low", "close"]
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    else:
        bars.index = bars.index.tz_convert("UTC")
    return bars.dropna()


def store_crypto(conn, asset: str, bars: pd.DataFrame) -> int:
    rows = [(asset, int(ts.timestamp()), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"])) for ts, r in bars.iterrows()]
    conn.executemany("INSERT OR REPLACE INTO crypto_bars "
                     "(asset, ts, open, high, low, close) VALUES (?,?,?,?,?,?)",
                     rows)
    conn.commit()
    return len(rows)


def load_crypto(conn, asset: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        "SELECT ts, open, high, low, close FROM crypto_bars "
        "WHERE asset = ? ORDER BY ts", conn, params=(asset,))
    if frame.empty:
        return frame
    frame.index = pd.to_datetime(frame.pop("ts"), unit="s", utc=True)
    return frame


# ---------------------------------------------------------------------------
# Parte A -- a ronda 5, replicada num activo de 24 horas
# ---------------------------------------------------------------------------

def minute_panel(bars: pd.DataFrame) -> pd.DataFrame:
    """Painel minimo com a MESMA definicao da ronda 5: hora do dia e |retorno|.

    A hora e' UTC. Nao ha' aqui escolha de fuso "local" a fazer -- cripto nao
    tem sessao -- e UTC e' tambem a hora em que os mercados da Polymarket
    resolvem, o que mantem os dois lados da experiencia no mesmo eixo."""
    if bars.empty:
        return pd.DataFrame()
    index = bars.index
    panel = pd.DataFrame({
        "ts": index,
        "bucket": index.hour,
        "day": index.normalize(),
        "abs_return": bars["close"].pct_change().abs().to_numpy(),
    })
    panel.index = range(len(panel))
    # Um salto entre pedacos do yfinance nao e' um retorno de um minuto.
    gap = pd.Series(index).diff().dt.total_seconds().to_numpy()
    panel.loc[(gap != 60) & ~np.isnan(gap), "abs_return"] = np.nan
    return panel.dropna()


def profile_of(panel: pd.DataFrame) -> pd.Series:
    """Factor por hora, normalizado pela media do periodo. A definicao e'
    exactamente a da ronda 5 para que os numeros sejam comparaveis."""
    by_bucket = panel.groupby("bucket")["abs_return"].mean()
    return by_bucket / by_bucket.mean()


def score_profile(panel: pd.DataFrame, profile: pd.Series) -> float:
    """Quao bem o perfil ordena os movimentos reais. Ronda 5, teste 2."""
    predicted = profile.reindex(panel["bucket"]).to_numpy()
    if np.all(np.isnan(predicted)) or panel["abs_return"].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(predicted, panel["abs_return"].to_numpy(),
                                 nan_policy="omit").statistic)


def amplitude(profile: pd.Series) -> float:
    """Quantas vezes a hora mais volatil se move face a' menos volatil. A
    ronda 5 mediu 4,09x +- 0,45 em accoes."""
    if profile.empty or profile.min() <= 0:
        return np.nan
    return float(profile.max() / profile.min())


def dispersion_by_hour(panel: pd.DataFrame, by: str = "day") -> pd.DataFrame:
    """Dispersao do factor de cada hora de periodo para periodo. A ronda 5
    mediu 4,8% a 11,4% de mes para mes; aqui a unidade e' o DIA, que e' muito
    mais exigente -- um dia tem uma fraccao dos dados de um mes, logo parte da
    dispersao e' ruido de amostra e nao deriva do fenomeno."""
    rows = []
    for key, piece in panel.groupby(by):
        rows.append(profile_of(piece).rename(key))
    if not rows:
        return pd.DataFrame()
    wide = pd.concat(rows, axis=1)
    return pd.DataFrame({
        "factor_medio": wide.mean(axis=1),
        "dispersao": wide.std(axis=1) / wide.mean(axis=1),
        "periodos": wide.notna().sum(axis=1),
    })


def dispersion_floor(panel: pd.DataFrame, by: str = "day", draws: int = 40,
                     seed: int = 5) -> pd.Series:
    """O CHAO de dispersao que o puro ruido de amostra produz.

    Sem isto a dispersao nao se interpreta. A ronda 5 mediu 4,8% a 11,4% de
    mes para mes com 160 accoes; aqui a unidade e' um dia de um activo, com
    uma fraccao dos dados, logo esperar os mesmos numeros seria comparar
    tamanhos de amostra e nao estabilidades. Este chao responde a' pergunta
    certa: num mundo onde o perfil fosse EXACTAMENTE constante, e so' variasse
    a amostra, quanta dispersao se veria? Simula-se isso reamostrando cada
    hora de cada dia da poca global dessa hora, com as contagens reais."""
    if panel.empty:
        return pd.Series(dtype=float)
    pools = {h: g.to_numpy() for h, g in panel.groupby("bucket")["abs_return"]}
    counts = panel.groupby([by, "bucket"]).size()
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(draws):
        rows = []
        for (key, bucket), n in counts.items():
            pool = pools[bucket]
            rows.append(pd.DataFrame({
                by: key, "bucket": bucket,
                "abs_return": pool[rng.integers(0, len(pool), n)]}))
        fake = pd.concat(rows, ignore_index=True)
        wide = pd.concat([profile_of(p).rename(k)
                          for k, p in fake.groupby(by)], axis=1)
        runs.append(wide.std(axis=1) / wide.mean(axis=1))
    return pd.concat(runs, axis=1).mean(axis=1)


def shuffled_score(panel: pd.DataFrame, profile: pd.Series,
                   draws: int = 40, seed: int = 5) -> tuple:
    """Baseline de ruido para `score_profile`: embaralha as horas e volta a
    pontuar. Diz se um Spearman pequeno mas positivo e' sinal ou acaso."""
    if panel.empty:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    work = panel.copy()
    buckets = work["bucket"].to_numpy()
    scores = []
    for _ in range(draws):
        work["bucket"] = rng.permutation(buckets)
        scores.append(score_profile(work, profile))
    scores = np.array([s for s in scores if not np.isnan(s)])
    return (float(scores.mean()), float(np.quantile(scores, 0.975)))


def profile_shape_correlation(panel: pd.DataFrame, by: str = "day") -> float:
    """Correlacao mediana entre perfis de periodos diferentes. Ronda 5 em
    accoes: 0,79 entre meses."""
    profiles = [profile_of(p) for _, p in panel.groupby(by)]
    profiles = [p for p in profiles if p.notna().sum() >= 12]
    if len(profiles) < 2:
        return np.nan
    wide = pd.concat(profiles, axis=1)
    matrix = wide.corr(method="spearman").to_numpy()
    upper = matrix[np.triu_indices(len(matrix), k=1)]
    upper = upper[~np.isnan(upper)]
    return float(np.median(upper)) if len(upper) else np.nan


def profile_age(panel: pd.DataFrame, lags=(0, 1, 3, 7, 14),
                window: int = 7) -> pd.DataFrame:
    """Ronda 5, teste 2 -- o decisivo. Estima o perfil numa janela de `window`
    dias que termina `lag` dias antes de cada dia-alvo e pontua-o nesse dia.
    `lag = 0` e' o oraculo: o perfil do proprio dia, impossivel na pratica, e
    serve de tecto -- diz quanto se perde por o perfil ser do passado."""
    days = np.sort(panel["day"].unique())
    rows = []
    for lag in lags:
        scores = []
        for position, day in enumerate(days):
            target = panel[panel["day"] == day]
            if len(target) < 200:
                continue
            if lag == 0:
                train = target
            else:
                stop = position - lag
                start = max(0, stop - window + 1)
                if stop < 0:
                    continue
                train = panel[panel["day"].isin(set(days[start:stop + 1]))]
            if len(train) < 1000:
                continue
            profile = profile_of(train)
            if profile.notna().sum() < 20:
                continue
            scores.append(score_profile(target, profile))
        scores = [s for s in scores if not np.isnan(s)]
        rows.append({"atraso_dias": lag, "dias": len(scores),
                     "spearman": float(np.mean(scores)) if scores else np.nan})
    table = pd.DataFrame(rows)
    oracle = table.loc[table["atraso_dias"] == 0, "spearman"]
    if len(oracle) and oracle.iloc[0]:
        table["perda_vs_oraculo"] = table["spearman"] / oracle.iloc[0] - 1.0
    return table


def history_needed(panel: pd.DataFrame, spans=(1, 3, 7, 14),
                   lag: int = 1) -> pd.DataFrame:
    """Ronda 5, teste 3 -- quanto historico chega para estimar o perfil."""
    days = np.sort(panel["day"].unique())
    rows = []
    for span in spans:
        scores = []
        for position, day in enumerate(days):
            target = panel[panel["day"] == day]
            stop = position - lag
            start = stop - span + 1
            if len(target) < 200 or start < 0:
                continue
            train = panel[panel["day"].isin(set(days[start:stop + 1]))]
            profile = profile_of(train)
            if profile.notna().sum() < 20:
                continue
            scores.append(score_profile(target, profile))
        scores = [s for s in scores if not np.isnan(s)]
        rows.append({"dias_usados": span, "dias_testados": len(scores),
                     "spearman": float(np.mean(scores)) if scores else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parte B -- o mercado paga por isso?
# ---------------------------------------------------------------------------

def derive(frame: pd.DataFrame, step: int = None) -> pd.DataFrame:
    """Um negocio por linha, visto SEMPRE do lado "Up".

    Porque o lado "Up" e nao o lado apostado: para calibracao interessa o preco
    pago, e a ronda 22 normalizou-o para isso. Para VOLATILIDADE interessa a
    cotacao do mercado, que e' a mesma para quem compra e para quem vende --
    vender "Up" a 0,30 diz que o mercado cota "Up" a 0,30, igual a comprar
    "Up" a 0,30. O lado so' decide quem ganha, e aqui nao e' o que se mede.
    Confundir os dois trocaria o preco de metade da amostra.

    O EPOCH DO SLUG E' O INICIO DA JANELA, NAO O FIM. A coluna da ronda 22
    chama-se `end_epoch` e eu li o nome em vez de medir: subtrai-la de `ts` dava
    um tempo-por-correr NEGATIVO em 95% dos negocios -- mediana -132 segundos
    num mercado de 300 -- e o filtro `tau > 0` deitava fora 1,8 dos 1,9 milhoes
    de negocios do btc5m, deixando so' os poucos cotados antes da abertura. A
    medicao sobreviveu porque a perda foi grande o suficiente para se notar;
    uma perda de 20% teria passado. A prova esta' na propria distribuicao:
    94,6% dos negocios caem em [epoch, epoch+passo] com mediana +132s contra os
    +150 esperados, e nos mercados de 15 minutos +603s contra 450. Por isso o
    passo e' obrigatorio e nao tem omissao silenciosa."""
    if frame.empty:
        return frame
    frame = frame.copy()
    if step is None:
        mapped = frame["family"].map(STEP)
        if mapped.isna().any():
            unknown = sorted(frame.loc[mapped.isna(), "family"].unique())
            raise ValueError(f"passo desconhecido para {unknown}; passa `step`")
        step_series = mapped.astype(int)
    else:
        step_series = pd.Series(int(step), index=frame.index)

    up = frame["outcome"].str.lower() == "up"
    frame["p_up"] = np.where(up, frame["price"], 1.0 - frame["price"])
    frame["won_up"] = (frame["winner"].str.lower() == "up").astype(float)
    frame["inicio"] = frame["end_epoch"]
    frame["fim"] = frame["end_epoch"] + step_series
    frame["tau"] = frame["fim"] - frame["ts"]
    frame["decorrido"] = frame["ts"] - frame["inicio"]

    # Erro de probabilidade e erro de NITIDEZ.
    #
    # `desvio` e' o erro cru: quanto o resultado excedeu o preco. `nitidez` e'
    # esse erro visto do lado do favorito: positivo quando os favoritos ganham
    # MAIS do que o preco diz (mercado sub-confiante, sobrestimou sigma),
    # negativo quando ganham menos (sobre-confiante, subestimou sigma). E' a
    # unica forma de somar favoritos e azaroes sem que se cancelem, porque o
    # erro de um e' o simetrico do erro do outro por construcao.
    frame["desvio"] = frame["won_up"] - frame["p_up"]
    frame["sinal"] = np.sign(frame["p_up"] - 0.5)
    frame["nitidez"] = frame["sinal"] * frame["desvio"]
    frame["aposta"] = np.abs(frame["p_up"] - 0.5)

    stamps = pd.to_datetime(frame["ts"], unit="s", utc=True)
    frame["hora"] = stamps.dt.hour
    frame["dia"] = stamps.dt.normalize()
    frame["faixa_preco"] = pd.cut(frame["p_up"], bins=list(PRICE_BANDS),
                                  include_lowest=True)
    frame["fraccao"] = frame["tau"] / step_series
    frame["faixa_tau"] = pd.cut(frame["fraccao"], bins=list(TAU_BANDS),
                                include_lowest=True)
    # Fora da janela ha' negocios a mais e a menos: cotacao antecipada antes da
    # abertura e liquidacao depois do fecho. Nem uns nem outros tem um
    # tempo-por-correr que signifique algo.
    keep = ((frame["decorrido"] >= 0) & (frame["tau"] > 0)
            & (frame["p_up"] > 0) & (frame["p_up"] < 1))
    return frame[keep].reset_index(drop=True)


def market_frame(conn, family: str = "btc5m") -> pd.DataFrame:
    """`derive` aplicado ao que esta' na base de dados."""
    return derive(pd.read_sql_query(
        "SELECT t.condition_id, t.outcome, t.price, t.size, t.ts, "
        "       m.winner, m.end_epoch, m.family "
        "FROM pm_trades t JOIN pm_markets m ON m.condition_id = t.condition_id "
        "WHERE m.winner IS NOT NULL AND m.family = ?", conn, params=(family,)))


def cell_weights(frame: pd.DataFrame) -> pd.Series:
    cells = frame.groupby(["faixa_preco", "faixa_tau"], observed=True).size()
    return cells / cells.sum()


def standardised_mean(frame: pd.DataFrame, column: str,
                      weights: pd.Series) -> float:
    """Media de `column` reagregada com os pesos globais de celula -- a
    armadilha C. Sem isto, uma hora onde se negoceia mais perto de 0,5 parece
    diferente das outras so' por causa da mistura."""
    if frame.empty:
        return np.nan
    cells = frame.groupby(["faixa_preco", "faixa_tau"],
                          observed=True)[column].mean()
    common = cells.index.intersection(weights.index)
    if not len(common):
        return np.nan
    share = weights.loc[common]
    return float((cells.loc[common] * share).sum() / share.sum())


def by_hour(frame: pd.DataFrame, column: str = "nitidez",
            weights: pd.Series = None) -> pd.DataFrame:
    """Erro de nitidez por hora do dia, cru e padronizado.

    `weights` permite impor os pesos de celula de FORA. E' necessario no
    bootstrap: pesos recalculados em cada reamostra fariam variar a propria
    definicao da estatistica entre reamostras."""
    if frame.empty:
        return pd.DataFrame()
    if weights is None:
        weights = cell_weights(frame)
    rows = []
    for hour, piece in frame.groupby("hora"):
        rows.append({
            "hora": int(hour),
            "negocios": len(piece),
            "mercados": piece["condition_id"].nunique(),
            "dias": piece["dia"].nunique(),
            "cru": float(piece[column].mean()),
            "padronizado": standardised_mean(piece, column, weights),
        })
    return pd.DataFrame(rows).sort_values("hora").reset_index(drop=True)


def hour_correlation(table: pd.DataFrame, profile: pd.Series,
                     column: str = "padronizado") -> dict:
    """A PREVISAO ASSINADA: o erro de nitidez por hora contra a volatilidade
    realizada dessa hora. Se o mercado ignora a sazonalidade, a correlacao e'
    NEGATIVA -- nas horas volateis os favoritos estao caros."""
    if table.empty or profile is None or profile.empty:
        return {}
    series = table.set_index("hora")[column].dropna()
    common = series.index.intersection(profile.dropna().index)
    if len(common) < 6:
        return {}
    x = profile.loc[common].to_numpy(float)
    y = series.loc[common].to_numpy(float)
    spearman = stats.spearmanr(x, y)
    pearson = stats.pearsonr(x, y)
    return {"horas": int(len(common)),
            "spearman": float(spearman.statistic),
            "p_spearman": float(spearman.pvalue),
            "pearson": float(pearson.statistic),
            "p_pearson": float(pearson.pvalue)}


def slope_vs_profile(table: pd.DataFrame, profile: pd.Series,
                     column: str = "padronizado") -> dict:
    """MAGNITUDE do efeito, em nitidez por unidade de factor de volatilidade.

    Porque nao basta a correlacao: a correlacao e' adimensional e um efeito
    pequeno mas consistente correlaciona-se tao bem como um grande. Para
    comparar a banda informativa com a banda-placebo -- onde a fisica diz que o
    efeito tem de ser ATENUADO, nao necessariamente ausente -- e' preciso um
    declive. E para comparar duas bandas de tamanhos diferentes e' preciso o
    erro-padrao, porque a amplitude crua de um perfil horario cresce com o
    ruido de amostra e a banda pequena parece sempre mais agitada."""
    if table.empty or profile is None or profile.empty:
        return {}
    series = table.set_index("hora")[column].dropna()
    common = series.index.intersection(profile.dropna().index)
    if len(common) < 6:
        return {}
    fit = stats.linregress(profile.loc[common].to_numpy(float),
                           series.loc[common].to_numpy(float))
    return {"horas": int(len(common)), "declive": float(fit.slope),
            "erro": float(fit.stderr), "p": float(fit.pvalue),
            "amplitude": float(series.loc[common].max()
                               - series.loc[common].min())}


def day_bootstrap(frame: pd.DataFrame, profile: pd.Series,
                  column: str = "nitidez", draws: int = 1000,
                  seed: int = 23) -> tuple:
    """Intervalo da correlacao reamostrando DIAS -- a armadilha B. As 24 horas
    do mesmo dia partilham o regime desse dia, portanto nao sao 24 observacoes
    independentes."""
    if frame.empty:
        return (np.nan, np.nan, np.nan)
    days = frame["dia"].unique()
    if len(days) < 4:
        return (np.nan, np.nan, np.nan)
    weights = cell_weights(frame)
    pieces = [frame[frame["dia"] == d] for d in days]
    point = hour_correlation(by_hour(frame, column, weights), profile)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        picked = rng.integers(0, len(pieces), len(pieces))
        sample = pd.concat([pieces[i] for i in picked])
        result = hour_correlation(by_hour(sample, column, weights), profile)
        if result:
            values.append(result["spearman"])
    if not values:
        return (point.get("spearman", np.nan), np.nan, np.nan)
    return (point.get("spearman", np.nan),
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)))


def market_bootstrap(frame: pd.DataFrame, column: str = "nitidez",
                     draws: int = 4000, seed: int = 23) -> tuple:
    """Reamostra MERCADOS, como a ronda 22. Milhares de negocios partilham UMA
    resolucao."""
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


def placebo_split(frame: pd.DataFrame, profile: pd.Series,
                  band=PLACEBO_BAND) -> dict:
    """O placebo que a fisica da' de graca: perto de 0,5 o preco nao contem
    informacao sobre sigma, porque Phi(0) = 0,5 para qualquer volatilidade. O
    efeito TEM de desaparecer nessa banda. Se nao desaparecer, nao e'
    volatilidade -- e' composicao ou artefacto."""
    low, high = band
    inside = frame[(frame["p_up"] >= low) & (frame["p_up"] <= high)]
    outside = frame[(frame["p_up"] < low) | (frame["p_up"] > high)]
    return {
        "placebo": slope_vs_profile(by_hour(inside), profile),
        "informativo": slope_vs_profile(by_hour(outside), profile),
        "corr_placebo": hour_correlation(by_hour(inside), profile),
        "corr_informativo": hour_correlation(by_hour(outside), profile),
        "n_placebo": int(len(inside)), "n_informativo": int(len(outside)),
    }


# ---------------------------------------------------------------------------
# Volatilidade implicita -- mecanismo, com a armadilha D assumida
# ---------------------------------------------------------------------------

def join_underlying(frame: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Junta `S_t` (fecho do minuto do negocio) e `K` (abertura do primeiro
    minuto da janela). O strike e' uma HIPOTESE sobre como a Polymarket define
    o mercado, e `strike_check` testa-a antes de se usar para alguma coisa."""
    if frame.empty or bars.empty:
        return frame
    # `astype("int64") // 10**9` NAO serve aqui. O pandas guarda o indice na
    # resolucao que inferiu -- segundos, milissegundos ou microsegundos -- e
    # `astype("int64")` devolve a contagem NESSA unidade. Dividir por 10**9
    # presume nanossegundos: com um indice em microsegundos os seis minutos
    # deste modulo colapsavam todos no mesmo valor e a juncao ligava o negocio
    # a' barra errada, ou rebentava por indice repetido. Subtrair a epoca e
    # dividir por um Timedelta e' exacto em qualquer resolucao.
    seconds = (bars.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    closes = pd.Series(bars["close"].to_numpy(), index=seconds)
    opens = pd.Series(bars["open"].to_numpy(), index=seconds)

    out = frame.copy()
    out["minuto"] = (out["ts"] // 60) * 60
    out["spot"] = out["minuto"].map(closes)
    out["strike"] = ((out["inicio"] // 60) * 60).map(opens)
    return out.dropna(subset=["spot", "strike"])


def strike_check(frame: pd.DataFrame, min_edge: float = 0.10) -> dict:
    """Valida a hipotese do strike. Se `K` esta' certo, o sinal de `S_t - K`
    tem de concordar com o sinal de `p - 0,5` quase sempre: um mercado que cota
    "Up" acima de 0,5 e' um mercado onde o subjacente esta' acima do strike. Se
    concordar metade das vezes, a hipotese esta' errada e a vol implicita nao
    vale nada. E' o teste que diz se a juncao serve para alguma coisa."""
    work = frame[frame["aposta"] >= min_edge]
    work = work[np.abs(work["spot"] - work["strike"]) > 0]
    if work.empty:
        return {}
    agree = np.sign(work["spot"] - work["strike"]) == work["sinal"]
    return {"n": int(len(work)), "concordancia": float(agree.mean())}


def implied_sigma(frame: pd.DataFrame) -> pd.DataFrame:
    """Inverte p = Phi(d) para sigma relativa por raiz de segundo.

    ATENUACAO ASSUMIDA (armadilha D): `S_t` vem de barras de um minuto e o
    movimento dentro do minuto entra inteiro no numerador como ruido. Ruido no
    numerador de uma razao atenua para zero qualquer relacao medida, portanto
    isto SUBESTIMA. Serve para dar magnitude em unidades de volatilidade, nao
    para decidir."""
    work = frame[(frame["aposta"] >= MIN_EDGE) & (frame["aposta"] <= MAX_EDGE)
                 & (frame["tau"] >= MIN_TAU)].copy()
    if work.empty:
        return work
    z = stats.norm.ppf(work["p_up"].to_numpy())
    deviation = (work["spot"] - work["strike"]).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = deviation / (z * np.sqrt(work["tau"].to_numpy()))
    work["sigma_implicita"] = sigma / work["strike"].to_numpy()
    return work[np.isfinite(work["sigma_implicita"])
                & (work["sigma_implicita"] > 0)]


def implied_profile(frame: pd.DataFrame) -> pd.Series:
    """Perfil horario da volatilidade IMPLICITA, normalizado como o realizado.

    E' a pergunta na sua forma mais directa: a volatilidade que o mercado cota
    curva-se por hora do dia como a realizada? Normalizar pela media torna as
    duas comparaveis apesar da atenuacao da armadilha D -- o ruido do minuto
    enviesa o NIVEL de sigma, mas enviesa-o por igual em todas as horas, logo a
    FORMA sobrevive. Usa-se a mediana por hora e nao a media porque a cauda de
    1/Phi^-1 perto de 0,5 e' pesada e a media seguiria os extremos."""
    if frame.empty or "sigma_implicita" not in frame:
        return pd.Series(dtype=float)
    by_hour_value = frame.groupby("hora")["sigma_implicita"].median()
    return by_hour_value / by_hour_value.mean()


def implied_vs_realised(sigma: pd.DataFrame, profile: pd.Series) -> dict:
    """A MEDIDA DIRECTA: o declive do perfil implicito sobre o realizado.

    1,0 quer dizer que o mercado cota a sazonalidade inteira; 0 que a ignora
    por completo. E' mais informativo que a correlacao porque tem unidades, e
    mais robusto que a nitidez porque nao depende dos desfechos -- le-se nos
    PRECOS. Pela atenuacao da armadilha D (o ruido do minuto pesa mais nas horas
    calmas e achata o perfil) este declive e' um LIMITE INFERIOR."""
    implied = implied_profile(sigma)
    if implied.empty or profile is None or profile.empty:
        return {}
    common = implied.index.intersection(profile.dropna().index)
    if len(common) < 6:
        return {}
    x = profile.loc[common].to_numpy(float)
    y = implied.loc[common].to_numpy(float)
    fit = stats.linregress(x, y)
    # Um perfil implicito constante -- o caso "o mercado ignora a sazonalidade
    # por completo" -- nao tem correlacao definida, mas tem declive zero, que
    # e' a leitura que interessa.
    flat = np.ptp(y) == 0 or np.ptp(x) == 0
    return {"horas": int(len(common)), "declive": float(fit.slope),
            "erro": float(fit.stderr),
            "spearman": (np.nan if flat
                         else float(stats.spearmanr(x, y).statistic)),
            "amplitude_implicita": amplitude(implied),
            "amplitude_realizada": amplitude(profile)}


def halves_check(frame: pd.DataFrame, sigma: pd.DataFrame,
                 profile: pd.Series) -> dict:
    """O controlo que decidiu as rondas 22 e 23: parte-se a amostra em duas
    metades no tempo e repete-se. Um nivel real repete-se; uma media de
    estimativas instaveis nao. E' o unico teste que separou o btc15m, onde o
    declive agregado de 0,751 se revelou a media de 0,325 e 1,057 -- duas
    medidas a tres sigma uma da outra, portanto nao duas medidas da mesma
    coisa."""
    if frame.empty or sigma.empty:
        return {}
    days = np.sort(frame["dia"].unique())
    if len(days) < 6:
        return {}
    cut = len(days) // 2
    out = {}
    for label, chosen in (("primeira", set(days[:cut])),
                          ("segunda", set(days[cut:]))):
        piece = sigma[sigma["dia"].isin(chosen)]
        result = implied_vs_realised(piece, profile)
        out[label] = (round(result.get("declive", np.nan), 3),
                      round(result.get("erro", np.nan), 3))
    return out


def realised_sigma_by_hour(bars: pd.DataFrame) -> pd.Series:
    """Sigma realizada por hora do dia na MESMA unidade que `implied_sigma`
    devolve -- retorno relativo por raiz de segundo."""
    if bars.empty:
        return pd.Series(dtype=float)
    returns = bars["close"].pct_change()
    spacing = pd.Series(bars.index).diff().dt.total_seconds().to_numpy()
    returns = returns[spacing == 60]
    if returns.empty:
        return pd.Series(dtype=float)
    return returns.groupby(returns.index.hour).std() / np.sqrt(60.0)


# ---------------------------------------------------------------------------
# Parte C -- o erro persiste de uns dias para os outros?
# ---------------------------------------------------------------------------

def persistence(frame: pd.DataFrame, column: str = "nitidez") -> dict:
    """Ronda 5, teste 2, aplicado ao ERRO DO MERCADO em vez da volatilidade.

    A sazonalidade da volatilidade persiste -- isso a ronda 5 estabeleceu. A
    pergunta negociavel e' outra: o perfil horario do erro DO MERCADO estimado
    na primeira metade dos dias prediz a segunda metade? Se nao, nao ha' nada a
    negociar, e e' a mesma resposta que a ronda 22 deu a' pericia das
    carteiras."""
    if frame.empty:
        return {}
    days = np.sort(frame["dia"].unique())
    if len(days) < 6:
        return {}
    cut = len(days) // 2
    first = frame[frame["dia"].isin(set(days[:cut]))]
    second = frame[frame["dia"].isin(set(days[cut:]))]
    weights = cell_weights(frame)
    a = by_hour(first, column, weights).set_index("hora")["padronizado"]
    b = by_hour(second, column, weights).set_index("hora")["padronizado"]
    common = a.dropna().index.intersection(b.dropna().index)
    if len(common) < 6:
        return {}
    result = stats.spearmanr(a.loc[common], b.loc[common])
    return {"horas": int(len(common)), "dias_total": int(len(days)),
            "spearman": float(result.statistic), "p": float(result.pvalue),
            "primeira_metade": float(first[column].mean()),
            "segunda_metade": float(second[column].mean())}


def detectable(frame: pd.DataFrame, column: str = "nitidez") -> dict:
    """Minimo detectavel por negocio, com o MERCADO como unidade. A ronda 22
    chegou a 1,18% por aposta; aqui o eixo e' 24 vezes mais fino porque se
    divide por hora, e e' isso que se tem de saber antes de acreditar em
    qualquer hora individual."""
    if frame.empty:
        return {}
    per_market = frame.groupby("condition_id")[column].mean()
    n = len(per_market)
    if n < 2:
        return {}
    sd = float(per_market.std())
    return {"mercados": n, "sd_por_mercado": sd,
            "mde_global": 2.8 * sd / np.sqrt(n),
            "mde_por_hora": 2.8 * sd / np.sqrt(max(n / 24.0, 1.0))}


# ---------------------------------------------------------------------------

def run(families=("btc5m", "eth5m", "btc15m"), refresh: bool = False) -> dict:
    conn = connect()
    out = {}

    for asset in sorted({UNDERLYING[f] for f in families}):
        stored = load_crypto(conn, asset)
        if refresh or len(stored) < 10_000:
            fresh = fetch_crypto(asset, days=30)
            if not fresh.empty:
                store_crypto(conn, asset, fresh)
                stored = load_crypto(conn, asset)
        out[f"barras_{asset}"] = stored
        print(f"{asset}: {len(stored)} barras de 1 minuto", flush=True)

    print("\n=== PARTE A -- a ronda 5 num activo de 24 horas ===", flush=True)
    for key, bars in list(out.items()):
        if not key.startswith("barras_"):
            continue
        asset = key.removeprefix("barras_")
        panel = minute_panel(bars)
        if panel.empty:
            continue
        profile = profile_of(panel)
        disp = dispersion_by_hour(panel)
        print(f"\n{asset}: {len(panel)} minutos, {panel['day'].nunique()} dias")
        print(f"  amplitude {amplitude(profile):.2f}x   forma entre dias "
              f"{profile_shape_correlation(panel):.3f}")
        print(f"  dispersao por hora {disp['dispersao'].min()*100:.1f}% a "
              f"{disp['dispersao'].max()*100:.1f}%")
        print(profile_age(panel).to_string(index=False))
        print(history_needed(panel).to_string(index=False))
        out[f"perfil_{asset}"] = profile

    print("\n=== PARTE B -- o mercado paga por isso? ===", flush=True)
    for family in families:
        frame = market_frame(conn, family)
        if frame.empty:
            print(f"{family}: sem dados")
            continue
        asset = UNDERLYING[family]
        profile = out.get(f"perfil_{asset}")
        print(f"\n{family}: {len(frame)} negocios, "
              f"{frame['condition_id'].nunique()} mercados, "
              f"{frame['dia'].nunique()} dias, {frame['hora'].nunique()} horas")
        print("  nitidez global {:.5f} [{:.5f}, {:.5f}]".format(
            *market_bootstrap(frame)))
        print(f"  poder: {detectable(frame)}")
        if profile is not None:
            table = by_hour(frame)
            print(f"  previsao assinada {hour_correlation(table, profile)}")
            print(f"  declive de nitidez {slope_vs_profile(table, profile)}")
            print("     (um mercado que IGNORASSE a sazonalidade daria -0,105;"
                  " um que a precasse bem, +0,003)")
            print("  IC por dias {:.3f} [{:.3f}, {:.3f}]".format(
                *day_bootstrap(frame, profile, draws=300)))
            print(f"  placebo {placebo_split(frame, profile)}")
        print(f"  persistencia {persistence(frame)}  (ruido sd ~0,27)")

        joined = join_underlying(frame, out[f"barras_{asset}"])
        if not joined.empty:
            print(f"  juncao {len(joined)}/{len(frame)}, "
                  f"strike {strike_check(joined)}")
            out[f"juncao_{family}"] = joined
            sigma = implied_sigma(joined)
            if not sigma.empty and profile is not None:
                print(f"  A MEDIDA DIRECTA -- a vol implicita curva-se como a "
                      f"realizada?\n      {implied_vs_realised(sigma, profile)}")
                print(f"      metades: {halves_check(frame, sigma, profile)}")
        out[f"mercado_{family}"] = frame
    conn.close()
    return out


if __name__ == "__main__":
    run()
