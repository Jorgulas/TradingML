"""ALGORITMO DE CURTO PRAZO -- 5 dias, o degrau que faltava.

Os prazos que ja' existiam neste projecto eram os extremos: as arvores decidem
a barra seguinte (1 minuto) e a carteira LONG decide a 20 dias. No meio nao
havia nada, e o meio e' precisamente onde a maior parte das decisoes reais e'
tomada.

Este modelo preve a proxima SEMANA de bolsa (5 dias), em EXCESSO sobre o
indice, com tres classes: SOBE / ESPERAR / DESCE. ESPERAR e' uma decisao a
serio -- e' a correcta quando o movimento previsto nao chega para pagar o custo
de agir.

A HIPOTESE QUE ESTE MODULO TESTA e' outra, e e' a razao de ele existir com este
desenho. Seis rondas de investigacao mostraram a direccao TEMPORAL de cada
accao como imprevisivel (49.1% a 53.5%, nenhum regime sobrevive a correccao de
Bonferroni). Mas nunca foi testada a informacao TRANSVERSAL: nao "esta accao
vai subir", e sim "esta accao vai subir MAIS DO QUE AS OUTRAS". Sao perguntas
diferentes. A segunda e' neutra ao mercado por construcao -- o factor comum
desaparece ao ordenar -- e e' onde a literatura encontra os efeitos mais
duraveis.

Por isso treinam-se TRES conjuntos de features sobre exactamente os mesmos
dados e o mesmo alvo:

    temporal      features de series temporais, como sempre se fez
    transversal   percentis dentro do dia, e mais nada
    ambos         a uniao

Se o transversal sozinho bater o temporal sozinho, a resposta a' pergunta em
aberto e' "o sinal e' transversal" -- e isso muda o que vale a pena construir.

AVALIACAO. A exactidao de 3 classes e' uma metrica fraca para isto. O que
importa e' a ordenacao: separa-se o universo pela confianca prevista e mede-se
o retorno em excesso de comprar o topo e vender o fundo. E' assim que um sinal
transversal e' usado, e o intervalo de confianca e' calculado por bootstrap ao
nivel do DIA, porque 158 tickers no mesmo dia sao uma observacao e nao 158.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.decision import panel as P

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBClassifier

D = config.DECISION

# Spread tipico por ponta em large-caps americanas. Mesmo valor usado na
# experiencia E, para os numeros serem comparaveis entre modulos.
STOCK_COST_BPS = 3.0


def feature_sets(frame: pd.DataFrame) -> dict:
    everything = P.feature_columns(frame, cross_sectional=True)
    temporal = [c for c in everything if not c.startswith("xs_")]
    transversal = [c for c in everything if c.startswith("xs_")]
    return {
        "temporal (series de cada accao)": temporal,
        "transversal (percentis do dia)": transversal,
        "ambos": everything,
    }


def fit(train: pd.DataFrame, columns: list, label: str = "label") -> XGBClassifier:
    model = XGBClassifier(**D["xgb_params"], objective="multi:softprob", num_class=3)
    model.fit(train[columns], train[label].astype(int))
    return model


def score(model: XGBClassifier, frame: pd.DataFrame, columns: list) -> np.ndarray:
    """Confianca direccional: P(sobe) - P(desce), em [-1, 1].

    Um unico numero ordenavel, que e' o que um sinal transversal precisa de
    ser. A classe prevista pelo argmax perde a intensidade."""
    probabilities = model.predict_proba(frame[columns])
    return probabilities[:, P.UP] - probabilities[:, P.DOWN]


def long_short_return(frame: pd.DataFrame, confidence: np.ndarray,
                      fraction: float = 0.1) -> pd.Series:
    """Retorno diario de comprar o topo e vender o fundo do universo.

    Dentro de cada dia ordena-se pela confianca, compra-se a fraccao de topo e
    vende-se a de baixo, com peso igual. Como o alvo ja' e' retorno em excesso
    sobre o indice, e como se compra e vende ao mesmo tempo, o que sobra e'
    ordenacao pura."""
    work = frame[["ts", "forward_return"]].copy()
    work["confidence"] = confidence
    out = {}
    for stamp, group in work.groupby("ts"):
        if len(group) < 20:
            continue
        size = max(1, int(len(group) * fraction))
        ordered = group.sort_values("confidence")
        top = ordered["forward_return"].iloc[-size:].mean()
        bottom = ordered["forward_return"].iloc[:size].mean()
        out[stamp] = float(top - bottom)
    return pd.Series(out).sort_index()


def information_coefficient(frame: pd.DataFrame, confidence: np.ndarray) -> pd.Series:
    """Correlacao de Spearman, DENTRO de cada dia, entre a confianca prevista e
    o retorno realizado.

    Porque esta metrica e nao so' o decil de topo. O spread topo-menos-fundo usa
    20% do universo e deita fora os outros 80%; e' a metrica com que se ganha
    dinheiro mas e' ruidosa por construcao. O IC usa as 158 accoes de cada dia e
    pergunta apenas se a ORDEM esta' certa. Tem por isso muito menos variancia,
    e num universo desta dimensao um IC de 0.02-0.04 ja' e' considerado util na
    literatura de factores. Continua a ser uma observacao por dia -- o problema
    da sobreposicao e' o mesmo e trata-se da mesma maneira."""
    work = frame[["ts", "forward_return"]].copy()
    work["confidence"] = confidence
    out = {}
    for stamp, group in work.groupby("ts"):
        if len(group) < 20 or group["confidence"].nunique() < 3:
            continue
        out[stamp] = float(stats.spearmanr(group["confidence"],
                                           group["forward_return"]).statistic)
    return pd.Series(out).sort_index()


def block_bootstrap_mean(values: np.ndarray, block: int = None,
                         n: int = 4000, seed: int = 17) -> tuple:
    """IC95% da media com BLOCOS, porque as janelas se sobrepoem.

    O erro que isto corrige. Forma-se uma carteira todos os dias e mede-se o
    que ela rende nos 5 dias seguintes -- logo a observacao de segunda e a de
    terca partilham 4 dos 5 dias. Sao quase a mesma medicao feita duas vezes.
    Um bootstrap que reamostra dias soltos assume 144 observacoes
    independentes quando ha' cerca de 144/5 = 29, e estreita o intervalo por um
    factor de ~sqrt(5). Um resultado a roçar a significancia passaria a
    "significativo" so' por causa disto.

    O bootstrap de blocos reamostra TROCOS contiguos do comprimento do
    horizonte, preservando a sobreposicao dentro de cada troco em vez de
    fingir que ela nao existe."""
    block = block or D["horizon_days"]
    rng = np.random.default_rng(seed)
    if len(values) < 2 * block:
        return (float("nan"), float("nan"))

    n_blocks = int(np.ceil(len(values) / block))
    starts_max = len(values) - block
    draws = []
    for _ in range(n):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        sample = np.concatenate([values[s:s + block] for s in starts])[:len(values)]
        draws.append(sample.mean())
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def non_overlapping(daily: pd.Series, step: int = None) -> np.ndarray:
    """So' uma janela em cada `step` dias: observacoes genuinamente disjuntas.

    E' a verificacao mais crua e a mais dificil de contestar -- nao depende de
    nenhuma hipotese sobre a estrutura de dependencia, so' evita-a. Custa poder
    estatistico (fica ~1/5 dos pontos), e por isso serve de confirmacao ao
    bootstrap de blocos e nao de substituto."""
    step = step or D["horizon_days"]
    return daily.values[::step]


def actionable(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Decompoe o acerto em "sabia que vinha movimento" e "sabia para onde".

    A distincao nao e' cosmetica: num teste anterior o modelo acertava 82.0% a
    detectar que vinha movimento e 53.5% a dizer o lado, e reportar so' a
    exactidao global creditava-lhe a segunda competencia por causa da
    primeira."""
    acted = y_pred != P.WAIT
    if not acted.any():
        return {"coverage": 0.0, "precision": None, "move_detection": None,
                "direction_accuracy": None}
    moved = y_true[acted] != P.WAIT
    return {
        "coverage": float(acted.mean()),
        "precision": float((y_pred[acted] == y_true[acted]).mean()),
        "move_detection": float(moved.mean()),
        "direction_accuracy": (float((y_pred[acted][moved] == y_true[acted][moved]).mean())
                               if moved.any() else None),
    }


def selection_turnover(frame: pd.DataFrame, confidence: np.ndarray,
                       fraction: float = 0.1, step: int = None) -> float:
    """Fraccao da carteira que muda de nomes entre rebalanceamentos.

    Medida em vez de assumida. Sem isto o custo de transaccao seria um palpite,
    e o palpite conservador (100% de rotacao) pode facilmente ser o dobro do
    real se o sinal for persistente."""
    step = step or D["horizon_days"]
    work = frame[["ts"]].copy()
    work["confidence"] = confidence
    stamps = sorted(work["ts"].unique())[::step]

    previous = None
    changes = []
    for stamp in stamps:
        group = work[work["ts"] == stamp]
        if len(group) < 20:
            continue
        size = max(1, int(len(group) * fraction))
        ordered = group.sort_values("confidence")
        names = set(frame.loc[ordered.index[-size:], "ticker"]) | \
                set(frame.loc[ordered.index[:size], "ticker"])
        if previous is not None and previous:
            changes.append(1 - len(names & previous) / len(previous))
        previous = names
    return float(np.mean(changes)) if changes else 1.0


def evaluate(train: pd.DataFrame, test: pd.DataFrame, columns: list) -> dict:
    model = fit(train, columns)
    confidence = score(model, test, columns)
    predicted = model.predict(test[columns])
    truth = test["label"].astype(int).values

    daily = long_short_return(test, confidence)
    values = daily.values
    low, high = block_bootstrap_mean(values)

    # Verificacao sem sobreposicao nenhuma: uma janela em cada 5 dias.
    disjoint = non_overlapping(daily)
    disjoint_low, disjoint_high = block_bootstrap_mean(disjoint, block=1)

    # Custo: 2 unidades de exposicao bruta (1 comprada + 1 vendida), duas
    # pontas por ciclo, ao spread tipico de large-caps americanas, vezes a
    # rotacao efectivamente medida.
    ic = information_coefficient(test, confidence)
    ic_low, ic_high = block_bootstrap_mean(ic.values)

    turnover = selection_turnover(test, confidence)
    cost = 2 * 2 * (STOCK_COST_BPS / 10_000) * turnover

    # Decis, para ver se a relacao e' monotona ou so' um extremo.
    work = test[["forward_return"]].copy()
    work["decile"] = pd.qcut(pd.Series(confidence, index=test.index), 10,
                             labels=False, duplicates="drop")
    by_decile = work.groupby("decile")["forward_return"].mean()

    mean = float(values.mean()) if len(values) else float("nan")
    return {
        "n_features": len(columns),
        "accuracy": float((predicted == truth).mean()),
        **actionable(truth, predicted),
        "long_short_mean": mean,
        "long_short_ci": (low, high),
        "long_short_days": int(len(values)),
        "hit_days": float((values > 0).mean()) if len(values) else float("nan"),
        "significant": low > 0,
        "disjoint_mean": float(disjoint.mean()) if len(disjoint) else float("nan"),
        "disjoint_ci": (disjoint_low, disjoint_high),
        "disjoint_n": int(len(disjoint)),
        "disjoint_significant": disjoint_low > 0,
        "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
        "ic_ci": (ic_low, ic_high),
        "ic_significant": ic_low > 0,
        "ic_positive_days": float((ic > 0).mean()) if len(ic) else float("nan"),
        "turnover": turnover,
        "cost": cost,
        "net_mean": mean - cost,
        "net_significant": (low - cost) > 0,
        "deciles": by_decile.to_dict(),
        "model": model,
        "columns": columns,
    }


def run(conn, panel_frame: pd.DataFrame = None) -> dict | None:
    frame = panel_frame if panel_frame is not None else P.build(conn)
    if frame.empty:
        return None

    train, validation, test = P.split(frame)
    if test.empty:
        return None

    sets = feature_sets(frame)

    # Escolha na VALIDACAO, com o teste intocado.
    on_validation = {name: evaluate(train, validation, columns)
                     for name, columns in sets.items()}
    chosen = max(on_validation, key=lambda k: on_validation[k]["long_short_mean"])

    # O modelo final treina em treino+validacao, e so' entao ve o teste.
    fit_frame = pd.concat([train, validation])
    on_test = {name: evaluate(fit_frame, test, columns)
               for name, columns in sets.items()}

    return {
        "n_rows": len(frame), "n_tickers": int(frame["ticker"].nunique()),
        "train_days": P.effective_days(train),
        "validation_days": P.effective_days(validation),
        "test_days": P.effective_days(test),
        "validation": on_validation, "test": on_test, "chosen": chosen,
    }


def _print_block(title, results):
    print(f"  {title}")
    print(f"    {'conjunto':>32} {'n':>4} {'exact.':>7} {'direcc.':>8} "
          f"{'bruto/5d':>11} {'IC95% (blocos)':>26} "
          f"{'rotacao':>8} {'custo':>8} {'liquido':>10}")
    for name, r in results.items():
        direction = f"{r['direction_accuracy']:.1%}" if r["direction_accuracy"] else "--"
        low, high = r["long_short_ci"]
        mark = "*" if r["significant"] else " "
        net = "*" if r["net_significant"] else " "
        print(f"    {name:>32} {r['n_features']:>4} {r['accuracy']:>7.1%} "
              f"{direction:>8} {r['long_short_mean']:>+10.4%}{mark} "
              f"[{low:>+8.4%}, {high:>+8.4%}] {r['turnover']:>8.0%} "
              f"{r['cost']:>8.3%} {r['net_mean']:>+9.4%}{net}")

    print(f"    {'':>32} {'IC medio':>10} {'IC95% (blocos)':>26} {'dias com IC>0':>14}")
    for name, r in results.items():
        low, high = r["ic_ci"]
        mark = "*" if r["ic_significant"] else " "
        print(f"    {name:>32} {r['ic_mean']:>+10.4f}{mark} "
              f"[{low:>+8.4f}, {high:>+8.4f}] {r['ic_positive_days']:>14.1%}")

    print()
    print(f"    sem sobreposicao (1 janela em cada {D['horizon_days']} dias):")
    for name, r in results.items():
        low, high = r["disjoint_ci"]
        mark = "*" if r["disjoint_significant"] else " "
        print(f"      {name:>30} n={r['disjoint_n']:>3}  "
              f"{r['disjoint_mean']:>+10.4%}{mark}  [{low:>+8.4%}, {high:>+8.4%}]")


if __name__ == "__main__":
    connection = database.get_connection()
    cache = Path(config.DATA_DIR) / "decision_panel.pkl"
    cached = pd.read_pickle(cache) if cache.exists() else None
    outcome = run(connection, cached)
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print("=" * 118)
    print("ALGORITMO DE CURTO PRAZO -- 5 dias, retorno em excesso sobre o indice")
    print(f"  {outcome['n_rows']:,} linhas, {outcome['n_tickers']} tickers | "
          f"treino {outcome['train_days']} dias, validacao {outcome['validation_days']}, "
          f"teste {outcome['test_days']}")
    print()
    _print_block("VALIDACAO (onde se escolhe)", outcome["validation"])
    print(f"    -> escolhido: {outcome['chosen']}")
    print()
    _print_block("TESTE (intocado ate' aqui)", outcome["test"])
    print()
    print("  Retorno a 5 dias por decil de confianca, no teste:")
    for name, r in outcome["test"].items():
        row = "  ".join(f"{v:+.2%}" for v in r["deciles"].values())
        print(f"    {name:>32}: {row}")
    print()
    print("  * = intervalo de confianca inteiramente acima de zero (bootstrap ao dia)")
    connection.close()
