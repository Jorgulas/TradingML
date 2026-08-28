"""Reduzir cada algoritmo a um VEREDICTO, para todos poderem votar juntos.

O problema que isto resolve. Os algoritmos deste projecto falam linguagens
diferentes: um devolve tres probabilidades, outro um tipo de padrao entre
vinte, outro uma probabilidade de cauda, outro um factor de sazonalidade. Nao
se somam nem se comparam. Uma tabela de verdade precisa que cada um diga uma de
poucas coisas, e que essas coisas signifiquem o mesmo para todos.

Cinco sinais, todos avaliados ao fecho de cada dia:

    curto     modelo de 5 dias        SOBE / NEUTRO / DESCE
    rapido    modelo de 1 dia         SOBE / NEUTRO / DESCE
    padrao    padrao grafico recente  ALTA / NENHUM / BAIXA
    regime    volatilidade do mercado CALMO / NORMAL / AGITADO
    alarme    risco de cauda          SIM / NAO

REGRA QUE NAO SE QUEBRA: tudo o que define um veredicto -- modelos, limiares,
tercis -- e' estimado SO' na porcao de treino. Um tercil calculado sobre a serie
inteira ja' contem o futuro: dizer "hoje a volatilidade esta' no tercil de
baixo" usando um corte que so' se conhece no fim da amostra e' fuga de
informacao, e das mais dificeis de ver depois.

Os dois primeiros sinais sao TRANSVERSAIS: a fasquia e' a distribuicao do
proprio dia, nao um limiar absoluto. "SOBE" quer dizer "esta' no tercil de topo
do universo hoje", nao "a probabilidade passou de 0.6". E' assim que devem ser
usados -- foi essa a forma que mediu sinal em src/decision/short_term.py --
e tem a vantagem de o veredicto nao mudar de sentido quando o mercado inteiro
fica optimista ou pessimista.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.decision import panel as P
from src.decision import short_term
from src.research.earnings_data import load as load_earnings
from src.research.tail_probability import TAIL_MULTIPLE, add_earnings_features

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

D = config.DECISION

# Ordem fixa dos valores de cada sinal. E' o alfabeto da tabela de verdade e
# tem de ser estavel entre execucoes, senao as celulas trocam de nome.
SIGNAL_VALUES = {
    "curto": ["DESCE", "NEUTRO", "SOBE"],
    "rapido": ["DESCE", "NEUTRO", "SOBE"],
    "padrao": ["BAIXA", "NENHUM", "ALTA"],
    "regime": ["CALMO", "NORMAL", "AGITADO"],
    "alarme": ["NAO", "SIM"],
}
SIGNAL_NAMES = list(SIGNAL_VALUES)


def _cross_sectional_tercile(frame: pd.DataFrame, confidence: np.ndarray,
                             labels=("DESCE", "NEUTRO", "SOBE")) -> pd.Series:
    """Tercil da confianca DENTRO de cada dia.

    Dias com poucas accoes (feriados parciais, falhas de dados) ficam NEUTRO em
    vez de serem cortados em tercis de dois elementos, que nao significariam
    nada."""
    series = pd.Series(confidence, index=frame.index)
    out = pd.Series(labels[1], index=frame.index, dtype=object)
    for _, positions in frame.groupby("ts").groups.items():
        group = series.loc[positions]
        if len(group) < 15 or group.nunique() < 3:
            continue
        # Percentil pelo PONTO MEDIO do posto, (k - 0.5)/n, e nao k/n.
        #
        # Com k/n o posto mais alto vale exactamente 1.0 e o mais baixo 1/n, o
        # que desloca toda a escala para cima: com 60 accoes, `<= 1/3` apanha 20
        # e `>= 2/3` apanha 21. Um nome a mais do lado comprado, todos os dias,
        # numa carteira que se quer neutra. E' pouco, mas e' um vies de sentido
        # fixo, e desses nao ha' nenhum que se acumule a favor.
        ranked = (group.rank(method="first") - 0.5) / len(group)
        out.loc[positions] = np.where(ranked <= 1 / 3, labels[0],
                                      np.where(ranked >= 2 / 3, labels[2], labels[1]))
    return out


def pattern_signal(conn, frame: pd.DataFrame, timeframe: str = "1h") -> pd.Series:
    """Vies do ultimo padrao CONFIRMADO antes do fecho de cada dia.

    Usa-se `confirmed_ts` e nao `end_ts`: um padrao so' e' um facto conhecido
    quando o preco confirma a rotura, e datar a informacao pelo fim da formacao
    daria ao modelo umas barras que ele nao teria tido.

    Padroes mais velhos do que `pattern_max_age_days` sao ignorados -- um
    triangulo de ha' dois meses nao informa a decisao de hoje, e deixa-lo entrar
    so' encheria a tabela de celulas com vies aparente e nenhum conteudo."""
    rows = conn.execute(
        "SELECT ticker, confirmed_ts, pattern_type FROM detected_patterns "
        "WHERE timeframe = ? AND confirmed_ts IS NOT NULL", (timeframe,)
    ).fetchall()

    out = pd.Series("NENHUM", index=frame.index, dtype=object)
    if not rows:
        return out

    patterns = pd.DataFrame([dict(r) for r in rows])
    patterns["ts"] = pd.to_datetime(patterns["confirmed_ts"], utc=True,
                                    format="ISO8601").dt.normalize()
    patterns["bias"] = patterns["pattern_type"].map(config.PATTERN_BIAS).fillna("neutral")
    patterns = (patterns[patterns["bias"] != "neutral"]
                .sort_values("ts")
                .drop_duplicates(subset=["ticker", "ts"], keep="last"))

    horizon = pd.Timedelta(days=D["pattern_max_age_days"])
    mapping = {"bullish": "ALTA", "bearish": "BAIXA"}

    for ticker, group in frame.groupby("ticker"):
        history = patterns[patterns["ticker"] == ticker]
        if history.empty:
            continue
        stamps = group["ts"].values.astype("datetime64[ns]")
        known = history["ts"].values.astype("datetime64[ns]")
        # searchsorted "right" com -1: o ultimo padrao ESTRITAMENTE anterior ou
        # do proprio dia, nunca um posterior.
        position = np.searchsorted(known, stamps, side="right") - 1
        valid = position >= 0
        if not valid.any():
            continue
        age = (stamps - known[np.clip(position, 0, len(known) - 1)])
        fresh = valid & (age <= horizon.to_timedelta64())
        verdict = history["bias"].map(mapping).values[np.clip(position, 0, len(known) - 1)]
        out.loc[group.index[fresh]] = verdict[fresh]

    return out


def regime_signal(frame: pd.DataFrame, fit_mask: pd.Series) -> pd.Series:
    """Tercil da volatilidade do MERCADO, com cortes estimados so' no treino.

    E' o unico sinal partilhado por todas as accoes no mesmo dia: descreve o
    ambiente, nao a accao. Entra na tabela precisamente por isso -- e' a
    dimensao que pode dizer QUANDO os outros sinais funcionam, que e' uma coisa
    que nenhum deles consegue dizer sobre si proprio.

    A escolha de `mkt_vol_20` e nao da volatilidade de cada accao vem da ronda
    5: a estrutura de volatilidade e' a unica coisa que se mostrou estavel ao
    longo do tempo neste projecto, e a do indice e' a mais estavel de todas."""
    if "mkt_vol_20" not in frame.columns:
        return pd.Series("NORMAL", index=frame.index, dtype=object)
    reference = frame.loc[fit_mask, "mkt_vol_20"].dropna()
    if len(reference) < 100:
        return pd.Series("NORMAL", index=frame.index, dtype=object)
    low, high = np.percentile(reference, [33.3, 66.7])
    values = frame["mkt_vol_20"]
    return pd.Series(np.where(values <= low, "CALMO",
                              np.where(values >= high, "AGITADO", "NORMAL")),
                     index=frame.index, dtype=object)


def tail_alarm(conn, frame: pd.DataFrame, fit_mask: pd.Series) -> pd.Series:
    """Alarme de cauda: hoje ha' risco desproporcionado de movimento extremo?

    Herda directamente a ronda 6, que mediu duas coisas relevantes. A primeira:
    a volatilidade recente preve caudas com AUC 0.5071, ou seja acaso puro --
    e' por isso que a referencia obvia nao serve. A segunda: o calendario de
    RESULTADOS preve-as com AUC 0.629 sozinho, e 0.663 combinado com o resto.
    O alarme util e' portanto sobretudo um alarme de "ha' resultados a chegar".

    Entra na tabela como sinal de RISCO e nao de direccao. A pergunta que faz e'
    "vale a pena agir hoje?", nao "para que lado?"."""
    earnings = load_earnings(conn)
    enriched = add_earnings_features(frame, earnings)

    level = (enriched.groupby("ticker")["abs_return"]
             .transform(lambda s: s.shift(1).rolling(60, min_periods=20).mean()))
    surprise = enriched["abs_return"] / level.replace(0, np.nan)
    is_tail = (surprise > TAIL_MULTIPLE).astype(float)

    columns = ["days_to_earnings", "days_since_earnings", "is_earnings_day",
               "earnings_window", "vol_20", "vol_60", "atr14_norm", "mkt_vol_20"]
    columns = [c for c in columns if c in enriched.columns]

    usable = fit_mask & np.isfinite(surprise) & is_tail.notna() & frame["label"].notna()
    if usable.sum() < 500 or is_tail[usable].nunique() < 2:
        return pd.Series("NAO", index=frame.index, dtype=object)

    params = {k: v for k, v in D["xgb_params"].items() if k != "max_depth"}
    model = XGBClassifier(**params, max_depth=4, objective="binary:logistic")
    model.fit(enriched.loc[usable, columns], is_tail[usable])
    probability = model.predict_proba(enriched[columns])[:, 1]

    base_rate = float(is_tail[usable].mean())
    return pd.Series(np.where(probability > 3 * base_rate, "SIM", "NAO"),
                     index=frame.index, dtype=object)


def attach(conn, frame: pd.DataFrame, fit_mask: pd.Series) -> pd.DataFrame:
    """Acrescenta as cinco colunas de veredicto ao painel.

    `fit_mask` marca as linhas que os modelos podem ver. Tudo o resto e'
    pontuado por modelos que nunca as viram."""
    frame = frame.copy()
    columns = short_term.feature_sets(frame)["transversal (percentis do dia)"]
    # Treinar so' onde o rotulo ja' existe: as ultimas barras entram no painel
    # para poderem ser decididas, mas nao tem futuro conhecido para ensinar.
    fit_frame = P.labelled(frame[fit_mask])

    slow = short_term.fit(fit_frame, columns, label="label")
    fast = short_term.fit(fit_frame, columns, label="fast_label")

    frame["curto_confidence"] = short_term.score(slow, frame, columns)
    frame["rapido_confidence"] = short_term.score(fast, frame, columns)
    frame["curto"] = _cross_sectional_tercile(frame, frame["curto_confidence"].values)
    frame["rapido"] = _cross_sectional_tercile(frame, frame["rapido_confidence"].values)
    frame["padrao"] = pattern_signal(conn, frame)
    frame["regime"] = regime_signal(frame, fit_mask)
    frame["alarme"] = tail_alarm(conn, frame, fit_mask)

    return frame


def coverage(frame: pd.DataFrame) -> dict:
    """Quanto de cada valor existe de facto. Um sinal que diz sempre a mesma
    coisa nao acrescenta nada a' tabela, e e' melhor ve-lo agora."""
    return {name: frame[name].value_counts(normalize=True).round(4).to_dict()
            for name in SIGNAL_NAMES if name in frame.columns}


if __name__ == "__main__":
    connection = database.get_connection()
    cache = Path(config.DATA_DIR) / "decision_panel.pkl"
    panel = pd.read_pickle(cache) if cache.exists() else P.build(connection)

    train, validation, test = P.split(panel)
    mask = panel["ts"] < validation["ts"].max()
    enriched = attach(connection, panel, mask)

    print("=" * 78)
    print("SINAIS -- distribuicao de cada veredicto")
    for name, distribution in coverage(enriched).items():
        parts = "  ".join(f"{k}={v:.1%}" for k, v in sorted(distribution.items()))
        print(f"  {name:>8}: {parts}")

    print()
    print("  Combinacoes distintas observadas: "
          f"{enriched.groupby(SIGNAL_NAMES).ngroups} de "
          f"{np.prod([len(v) for v in SIGNAL_VALUES.values()])} possiveis")
    enriched.to_pickle(Path(config.DATA_DIR) / "decision_signals.pkl")
    print("  gravado em data/decision_signals.pkl")
    connection.close()
