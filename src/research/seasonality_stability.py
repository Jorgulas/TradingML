"""EXPERIENCIA H -- a sazonalidade intradiaria e' estavel mes a mes?

Porque importa. O "J" intradiario e' a UNICA coisa neste projecto que preve de
forma fiavel: sozinho vale 74% do poder preditivo do modelo de 40 features. Se
for estavel, e' a descoberta duravel e um perfil estimado uma vez serve
sempre. Se derivar, e' preciso reestimar -- e a propria descoberta fica menos
solida do que parecia.

TRES TESTES, do mais barato ao mais decisivo:

  1. FORMA. Os perfis mensais parecem-se uns com os outros? Mede-se a
     correlacao entre meses e a dispersao de cada faixa horaria.

  2. ENVELHECIMENTO -- e' este que decide. Estimar o perfil no mes M-L e
     aplica-lo ao mes M, para varios L. Se um perfil de ha' 12 meses funcionar
     tao bem como o do mes passado, e' estavel de facto. E' o teste pratico:
     ninguem quer saber se os perfis se correlacionam, quer saber se o perfil
     velho ainda serve.

  3. QUANTO HISTORICO CHEGA. Um mes basta para estimar o perfil, ou sao
     precisos seis? Determina com que frequencia vale a pena reestimar.

Comparacao de referencia em todos: o perfil "oraculo", estimado no proprio mes
que se esta' a prever. E' impossivel na pratica e serve de tecto -- diz quanto
se perde por o perfil ser necessariamente do passado.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.patterns.ingest_intraday import load_bars

import numpy as np
import pandas as pd
from scipy import stats

N_BUCKETS = 8


def build_panel(conn, timeframe: str = "1h", tickers=None) -> pd.DataFrame:
    """Painel minimo: ticker, timestamp, faixa horaria, mes e |retorno|."""
    tickers = tickers or config.TREE_TICKERS
    frames = []
    for ticker in tickers:
        bars = load_bars(conn, ticker, timeframe)
        if len(bars) < 400:
            continue
        # Faixa = HORA do dia, nao posicao normalizada.
        #
        # A versao anterior usava int(posicao/(n-1) * 8) com 8 faixas, mas ha'
        # 7 barras por dia: as posicoes 0..6 mapeiam para 0,1,2,4,5,6,7 e a
        # faixa 3 fica SEMPRE vazia nos dias normais. So' recebia dados dos
        # raros dias curtos (3 barras), o que explicava os 63% de dispersao
        # nessa faixa -- era ruido de amostra minuscula, nao um fenomeno.
        # Com barras horarias a hora e' directa e sem ambiguidade.
        local_hour = bars.index.tz_convert("America/New_York").hour

        piece = pd.DataFrame({
            "ticker": ticker,
            "ts": bars.index,
            "bucket": local_hour,
            # tz_localize(None) antes de to_period: converter directamente
            # descarta o fuso com aviso, e o mes e' o mesmo de qualquer forma.
            "month": bars.index.tz_localize(None).to_period("M").astype(str),
            "abs_return": bars["close"].pct_change().abs().values,
        })
        # O indice vem de load_bars ja' chamado "ts"; sem isto o sort_values
        # fica ambiguo entre nivel de indice e coluna.
        piece.index = range(len(piece))
        frames.append(piece.dropna())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values("ts").reset_index(drop=True)


def profile_of(subset: pd.DataFrame) -> pd.Series:
    """Factor por faixa: quanto essa faixa se move face a' media do periodo.
    Normalizado, portanto capta a FORMA e nao o nivel."""
    by_bucket = subset.groupby("bucket")["abs_return"].mean()
    return by_bucket / by_bucket.mean()


def score_profile(subset: pd.DataFrame, profile: pd.Series) -> float:
    """Quao bem o perfil ordena os movimentos reais dentro do mes."""
    predicted = profile.reindex(subset["bucket"]).values
    if np.all(np.isnan(predicted)) or subset["abs_return"].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(predicted, subset["abs_return"].values,
                                 nan_policy="omit").statistic)


def run(conn, timeframe: str = "1h") -> dict | None:
    panel = build_panel(conn, timeframe)
    if panel.empty:
        return None

    months = sorted(panel["month"].unique())
    monthly = {m: profile_of(panel[panel["month"] == m]) for m in months}
    matrix = pd.DataFrame(monthly)

    # --- Teste 1: a forma e' parecida entre meses? ---
    correlations = matrix.corr(method="spearman")
    upper = correlations.values[np.triu_indices_from(correlations.values, k=1)]
    dispersion = matrix.std(axis=1) / matrix.mean(axis=1)

    # --- Teste 2: envelhecimento ---
    ageing = []
    for lag in (1, 2, 3, 6, 12):
        scores = []
        for i, month in enumerate(months):
            if i - lag < 0:
                continue
            source = panel[panel["month"] == months[i - lag]]
            target = panel[panel["month"] == month]
            if len(source) < 500 or len(target) < 500:
                continue
            scores.append(score_profile(target, profile_of(source)))
        scores = [s for s in scores if np.isfinite(s)]
        if scores:
            ageing.append({"lag_months": lag, "mean_spearman": float(np.mean(scores)),
                           "n_months": len(scores)})

    # Tecto: perfil estimado no proprio mes (impossivel na pratica)
    oracle = [score_profile(panel[panel["month"] == m], monthly[m]) for m in months]
    oracle = [s for s in oracle if np.isfinite(s)]

    # --- Teste 3: quanto historico chega ---
    history = []
    for window in (1, 3, 6, 12):
        scores = []
        for i, month in enumerate(months):
            if i - window < 0:
                continue
            source = panel[panel["month"].isin(months[i - window:i])]
            target = panel[panel["month"] == month]
            if len(source) < 500 or len(target) < 500:
                continue
            scores.append(score_profile(target, profile_of(source)))
        scores = [s for s in scores if np.isfinite(s)]
        if scores:
            history.append({"months_used": window, "mean_spearman": float(np.mean(scores)),
                            "n_months": len(scores)})

    amplitude = matrix.max() / matrix.min()

    return {
        "timeframe": timeframe, "n_months": len(months), "months": months,
        "profile_matrix": matrix,
        "median_correlation": float(np.median(upper)),
        "min_correlation": float(np.min(upper)),
        "bucket_dispersion": dispersion.to_dict(),
        "ageing": ageing,
        "oracle_spearman": float(np.mean(oracle)) if oracle else np.nan,
        "history": history,
        "amplitude_by_month": amplitude.to_dict(),
        "amplitude_mean": float(amplitude.mean()),
        "amplitude_std": float(amplitude.std()),
    }


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection)
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print("=" * 78)
    print(f"EXPERIENCIA H -- estabilidade da sazonalidade intradiaria @ "
          f"{outcome['timeframe']}")
    print(f"  {outcome['n_months']} meses: {outcome['months'][0]} a {outcome['months'][-1]}")

    print()
    print("  TESTE 1 -- a forma repete-se de mes para mes?")
    print(f"    correlacao mediana entre perfis mensais: {outcome['median_correlation']:.4f}")
    print(f"    pior par de meses                      : {outcome['min_correlation']:.4f}")
    print("    variacao de cada faixa entre meses (desvio relativo):")
    for bucket, value in outcome["bucket_dispersion"].items():
        print(f"      faixa {bucket}/{N_BUCKETS - 1}: {value:.1%}")

    print()
    print("  TESTE 2 -- um perfil VELHO ainda serve? (o teste que decide)")
    print(f"    {'idade do perfil':>18} {'spearman':>10} {'perda vs oraculo':>18}")
    oracle = outcome["oracle_spearman"]
    print(f"    {'oraculo (o proprio mes)':>18} {oracle:>10.4f} {'--':>18}")
    for row in outcome["ageing"]:
        loss = (row["mean_spearman"] - oracle) / oracle * 100 if oracle else 0
        print(f"    {str(row['lag_months']) + ' mes(es) antes':>18} "
              f"{row['mean_spearman']:>10.4f} {loss:>17.1f}%")

    print()
    print("  TESTE 3 -- quanto historico chega para estimar o perfil?")
    print(f"    {'meses usados':>14} {'spearman':>10}")
    for row in outcome["history"]:
        print(f"    {row['months_used']:>14} {row['mean_spearman']:>10.4f}")

    print()
    print(f"  Amplitude do 'J' (faixa mais volatil / mais calma):")
    print(f"    media {outcome['amplitude_mean']:.2f}x, desvio {outcome['amplitude_std']:.2f}")
    values = list(outcome["amplitude_by_month"].values())
    print(f"    intervalo: {min(values):.2f}x a {max(values):.2f}x")
    connection.close()
