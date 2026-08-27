"""EXPERIENCIA E -- o sinal residual sobrevive a custos de transaccao?

E' a pergunta que separa "existe" de "serve". O sinal residual (surpresa
des-sazonalizada) tem Spearman 0.17 e lift de 1.90x entre decis, e e'
estatisticamente significativo. Mas um sinal de MAGNITUDE nao se pode negociar
direccionalmente -- e' preciso um instrumento que pague pelo tamanho do
movimento, e esses tem custos proprios.

DOIS OBSTACULOS, ambos medidos em vez de assumidos:

1. PREMIO DE RISCO DE VOLATILIDADE. A volatilidade implicita excede a
   realizada 85-90% das vezes, em media 2-4 pontos percentuais, o que da'
   cerca de 11% do premio. Quem COMPRA volatilidade paga sistematicamente a
   mais -- e' a compensacao que os vendedores exigem por suportar risco de
   cauda. Um sinal de magnitude tem de superar isto antes de tocar em custos.

2. CUSTOS DE TRANSACCAO. Em accoes large-cap americanas o spread ronda 2-3
   pontos base por lado. Em opcoes e' muito mais, e paga-se em percentagem do
   premio, nao do nocional.

FORMULACAO. Uma posicao longa em volatilidade (straddle) paga o movimento
esperado e recebe o movimento realizado. Em unidades de movimento esperado:

    resultado = surpresa - (1 + premio_de_risco) - custo

onde surpresa = |retorno realizado| / |movimento esperado|. O ponto de
equilibrio e' portanto `surpresa > 1 + premio + custo`. A pergunta empirica e'
simples: o decil de topo do modelo chega la'?

A alternativa que NAO precisa de opcoes: se e' preciso negociar de qualquer
forma, negociar nas barras de baixa volatilidade prevista reduz o slippage.
Isso e' um beneficio real e mede-se directamente.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.research.har_benchmark import build_frame
from src.research.har_plus_seasonality import N_SESSION_BUCKETS
from src.trees import train as tree_train

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor

# Numeros da literatura, nao inventados.
VOLATILITY_RISK_PREMIUM = 0.11      # implicita excede realizada em ~11% do premio
STOCK_COST_BPS = 3.0                # spread por lado em large-caps americanas
OPTION_COST_RATIOS = (0.02, 0.05, 0.10)   # custo em fraccao do premio


ROLLING_WINDOW = 200   # barras usadas para estimar o nivel corrente de volatilidade


def seasonal_factors(reference: pd.DataFrame) -> pd.Series:
    """Quanto cada faixa horaria se move face a' media do dia. So' a FORMA,
    estimada no treino -- e' o que se assume estavel, nao o nivel."""
    bucket = np.clip((reference["session_position"] * N_SESSION_BUCKETS).astype(int),
                     0, N_SESSION_BUCKETS - 1)
    by_bucket = reference.assign(_b=bucket).groupby("_b")["abs_return"].mean()
    return by_bucket / by_bucket.mean()


def rolling_expected_move(frame: pd.DataFrame, factors: pd.Series) -> pd.Series:
    """Movimento esperado = NIVEL CORRENTE x forma sazonal.

    A primeira versao desta experiencia usou uma media estatica do periodo de
    treino, e deu 76% de "lucro" no decil de topo. Era falso: a surpresa media
    saiu 1.26 em vez de 1.00, ou seja a volatilidade no teste foi 26% mais
    alta do que no treino, e todo o ganho aparente era essa mudanca de regime.

    Um vendedor de opcoes cota a volatilidade de HOJE. O nivel tem de ser
    estimado com uma janela movel que so' olha para tras, e so' a forma
    intradiaria (que e' estrutural) vem do treino."""
    bucket = np.clip((frame["session_position"] * N_SESSION_BUCKETS).astype(int),
                     0, N_SESSION_BUCKETS - 1)
    shape = pd.Series(factors.reindex(bucket).values, index=frame.index).fillna(1.0)

    level = (frame.groupby("ticker")["abs_return"]
             .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=50).mean()))
    return level * shape


def run(conn, timeframe: str = "1h") -> dict | None:
    frame = build_frame(conn, timeframe)
    if frame.empty:
        return None

    train_df, validation_df, test_df = tree_train.split_by_time(
        frame, config.TREES["horizon_bars"])
    fit_df = pd.concat([train_df, validation_df])

    factors = seasonal_factors(fit_df)
    for part in (fit_df, test_df):
        part["expected_move"] = rolling_expected_move(part, factors)
        part["surprise"] = part["abs_return"] / part["expected_move"].replace(0, np.nan)
    fit_df = fit_df[np.isfinite(fit_df["surprise"])]
    test_df = test_df[np.isfinite(test_df["surprise"])].copy()

    features = [c for c in frame.columns
                if c not in ("ticker", "ts", "big_move", "magnitude_ratio",
                             "abs_return", "band", "surprise", "expected_move")]
    model = XGBRegressor(**{k: v for k, v in config.TREES["xgb_params"].items()
                            if k not in ("objective", "num_class")},
                         max_depth=4, objective="reg:squarederror")
    model.fit(fit_df[features], fit_df["surprise"])
    test_df["predicted"] = model.predict(test_df[features])

    # Decis por surpresa PREVISTA, com a surpresa REALIZADA em cada um
    test_df["decile"] = pd.qcut(test_df["predicted"], 10, labels=False, duplicates="drop")
    by_decile = test_df.groupby("decile").agg(
        predicted=("predicted", "mean"),
        realised=("surprise", "mean"),
        expected_move=("expected_move", "mean"),
        n=("surprise", "size"),
    ).reset_index()

    top = by_decile.iloc[-1]
    bottom = by_decile.iloc[0]

    # Verificacao obrigatoria: se a surpresa MEDIA nao for ~1, o movimento
    # esperado esta' mal calibrado e todos os lucros abaixo sao artefacto.
    mean_surprise = float(test_df["surprise"].mean())

    # Ponto de equilibrio: quanto custo suporta o decil de topo?
    break_even_cost = float(top["realised"] - 1 - VOLATILITY_RISK_PREMIUM)

    scenarios = []
    for cost in OPTION_COST_RATIOS:
        net = float(top["realised"] - 1 - VOLATILITY_RISK_PREMIUM - cost)
        scenarios.append({
            "cost_ratio": cost,
            "net_per_trade_in_expected_move_units": net,
            # Traduzido para pontos base do nocional: o movimento esperado
            # tipico vezes o resultado liquido em unidades desse movimento.
            "net_bps_of_notional": net * float(top["expected_move"]) * 10_000,
            "profitable": net > 0,
        })

    # Alternativa sem opcoes: poupanca de slippage ao negociar nas barras calmas
    calm_move = float(bottom["realised"] * bottom["expected_move"])
    average_move = float((test_df["surprise"] * test_df["expected_move"]).mean())
    slippage_saving_bps = (average_move - calm_move) * 10_000

    return {
        "timeframe": timeframe,
        "mean_surprise": mean_surprise,
        "calibrated": abs(mean_surprise - 1.0) < 0.05,
        "n_test": len(test_df),
        "effective_days": int(pd.Series(test_df["ts"]).dt.normalize().nunique()),
        "spearman": float(stats.spearmanr(test_df["predicted"], test_df["surprise"]).statistic),
        "deciles": by_decile.to_dict("records"),
        "top_realised_surprise": float(top["realised"]),
        "break_even_cost_ratio": break_even_cost,
        "scenarios": scenarios,
        "slippage_saving_bps": slippage_saving_bps,
        "average_move_bps": average_move * 10_000,
        "calm_move_bps": calm_move * 10_000,
    }


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run(connection, "1h")
    if outcome is None:
        print("dados insuficientes")
        raise SystemExit

    print("=" * 76)
    print("EXPERIENCIA E -- o sinal residual sobrevive a custos?")
    print(f"  @ {outcome['timeframe']}, {outcome['n_test']:,} linhas, "
          f"{outcome['effective_days']} dias | Spearman {outcome['spearman']:.4f}")
    print()
    print("  Surpresa realizada por decil de surpresa PREVISTA")
    print("  (1.00 = o movimento foi exactamente o esperado para aquela hora)")
    print(f"    {'decil':>6} {'previsto':>10} {'realizado':>11} {'n':>9}")
    for d in outcome["deciles"]:
        marker = "  <-- topo" if d["decile"] == 9 else ("  <-- fundo" if d["decile"] == 0 else "")
        print(f"    {int(d['decile']) + 1:>6} {d['predicted']:>10.3f} {d['realised']:>11.3f} "
              f"{int(d['n']):>9,}{marker}")

    print()
    print(f"  VERIFICACAO DE CALIBRACAO: surpresa media = {outcome['mean_surprise']:.3f} "
          f"(tem de ser ~1.00)")
    if not outcome["calibrated"]:
        print("  !! MAL CALIBRADO -- o movimento esperado nao reflecte o regime do teste.")
        print("     Qualquer lucro abaixo e' artefacto de mudanca de regime, nao sinal.")
    else:
        print("  OK -- o movimento esperado acompanha o regime do periodo de teste.")
    print()
    print(f"  O decil de topo realiza surpresa de {outcome['top_realised_surprise']:.3f}")
    print(f"  Premio de risco de volatilidade a pagar: {VOLATILITY_RISK_PREMIUM:.2f}")
    print(f"  -> sobra para custos: {outcome['break_even_cost_ratio']:+.4f} do premio")
    print()
    print("  Cenarios de custo (opcoes, custo em fraccao do premio):")
    print(f"    {'custo':>8} {'liquido':>12} {'em bps do nocional':>20}  veredicto")
    for s in outcome["scenarios"]:
        verdict = "LUCRATIVO" if s["profitable"] else "perde dinheiro"
        print(f"    {s['cost_ratio']:>7.0%} {s['net_per_trade_in_expected_move_units']:>+12.4f} "
              f"{s['net_bps_of_notional']:>+19.2f}  {verdict}")

    print()
    print("  Alternativa sem opcoes -- cronometrar a execucao:")
    print(f"    movimento medio numa barra qualquer : {outcome['average_move_bps']:.1f} bps")
    print(f"    movimento no decil previsto calmo   : {outcome['calm_move_bps']:.1f} bps")
    print(f"    poupanca potencial de slippage      : {outcome['slippage_saving_bps']:.1f} bps")
    print(f"    (contra um custo de spread de ~{STOCK_COST_BPS:.0f} bps por lado)")
    connection.close()
