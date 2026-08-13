"""Terceira carteira simulada: negoceia o sinal de direccao pos-padrao.

Cadencia propria, e nao a do simulator.py das outras duas carteiras. Aquele
opera em barras DIARIAS; o horizonte que a validacao seleccionou aqui e' de 5
barras de 1h, cerca de 5 horas de mercado. Manter a posicao um dia inteiro
ultrapassaria o que o modelo preve, e o backtest deixaria de testar o sinal
para testar uma aproximacao. Aqui a regra e' exactamente a do modelo: entra na
CONFIRMACAO do padrao, sai horizon_bars barras depois.

O modelo e' treinado em treino+validacao e a carteira corre so' sobre o
periodo de TESTE -- o mesmo periodo que a avaliacao usa e que nunca serviu
para escolher nada. Rodar a carteira sobre dados de treino daria um resultado
bonito e sem significado.

Duas referencias obrigatorias, senao o numero final nao se sabe ler:
  buy & hold   -- comprar tudo no inicio e nao fazer nada
  entradas ao acaso -- as mesmas N entradas, nos mesmos instantes, mas com o
                      lado escolhido a moeda ao ar
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from db import database
from src.patterns import context, direction
from src.patterns.ingest_intraday import load_bars

import numpy as np
import pandas as pd

P = config.PATTERN_PORTFOLIO


def _price_lookup(conn, timeframe: str, tickers) -> dict:
    out = {}
    for ticker in tickers:
        bars = load_bars(conn, ticker, timeframe)
        if not bars.empty:
            out[ticker] = (bars["close"].values, {ts: i for i, ts in enumerate(bars.index)}, bars.index)
    return out


def run_backtest(conn, timeframe: str = "1h", horizon: int = None,
                 market_neutral: bool = None, seed: int = 42) -> dict | None:
    """Corre a carteira sobre o periodo de teste e devolve o resumo."""
    selected = conn.execute(
        "SELECT * FROM pattern_direction_models WHERE selected = 1 ORDER BY trained_at DESC LIMIT 1"
    ).fetchone()
    if selected is None:
        return None
    horizon = horizon if horizon is not None else selected["horizon_bars"]
    market_neutral = market_neutral if market_neutral is not None else bool(selected["market_neutral"])

    frame = direction.build_dataset(conn, timeframe, horizon, market_neutral)
    if frame.empty:
        return None
    train_df, validation_df, test_df = direction.split_with_embargo(frame, horizon)
    if test_df.empty or len(train_df) < config.PATTERN_DIRECTION["min_train_rows"]:
        return None

    # Treina em treino+validacao; negoceia so' no teste.
    fit_df = pd.concat([train_df, validation_df])
    features = context.FEATURE_NAMES
    model = direction._candidates()[selected["algorithm"]]
    model.fit(fit_df[features], fit_df["label"])
    probabilities = model.predict_proba(test_df[features])[:, 1]

    prices = _price_lookup(conn, timeframe, sorted(test_df["ticker"].unique()))

    events = test_df.copy()
    events["confidence"] = probabilities
    events = events.sort_values("confirmed_ts").reset_index(drop=True)

    cash = P["starting_cash"]
    open_positions = []   # (exit_ts, ticker, qty, entry_price, entry_ts, confidence, pattern)
    trades = []
    # Dicionario e nao lista: as barras de 1h sao as MESMAS para todos os
    # tickers, portanto varios padroes confirmam no mesmo instante. A curva
    # tem de ter um ponto por instante, com o estado depois de processados
    # todos os eventos desse instante.
    equity_by_ts = {}
    rng = np.random.default_rng(seed)
    random_returns = []

    def close_due(now_ts):
        nonlocal cash
        still_open = []
        for position in open_positions:
            exit_ts, ticker, qty, entry_price, entry_ts, confidence, pattern_type = position
            if exit_ts <= now_ts:
                closes, position_of, _ = prices[ticker]
                exit_price = float(closes[position_of[exit_ts]])
                cash += qty * exit_price
                trades.append({
                    "ticker": ticker, "pattern_type": pattern_type,
                    "entry_ts": entry_ts, "exit_ts": exit_ts,
                    "entry_price": entry_price, "exit_price": exit_price, "qty": qty,
                    "confidence": confidence, "pnl": qty * (exit_price - entry_price),
                    "return_pct": exit_price / entry_price - 1,
                })
            else:
                still_open.append(position)
        open_positions[:] = still_open

    for _, event in events.iterrows():
        now_ts = pd.Timestamp(event["confirmed_ts"])
        close_due(now_ts)

        ticker = event["ticker"]
        if ticker not in prices:
            continue
        closes, position_of, index = prices[ticker]
        entry = position_of.get(now_ts)
        if entry is None or entry + horizon >= len(closes):
            continue

        # Referencia ao acaso: mesmo instante, mesmo horizonte, lado a' sorte
        forward = closes[entry + horizon] / closes[entry] - 1
        random_returns.append(forward if rng.random() < 0.5 else -forward)

        already_open = any(p[1] == ticker for p in open_positions)
        if (event["confidence"] >= P["confidence_threshold"] and not already_open
                and len(open_positions) < P["max_positions"]):
            equity = cash + sum(
                q * float(prices[t][0][prices[t][1][e_ts]]) if e_ts in prices[t][1] else q * ep
                for e_ts, t, q, ep, _, _, _ in [(now_ts, p[1], p[2], p[3], 0, 0, 0) for p in open_positions]
            )
            target = equity * P["position_size_pct"]
            entry_price = float(closes[entry])
            if target <= cash and entry_price > 0:
                qty = target / entry_price
                cash -= target
                open_positions.append((index[entry + horizon], ticker, qty, entry_price,
                                       now_ts, float(event["confidence"]), event.get("from_pattern", "")))

        positions_value = sum(p[2] * float(prices[p[1]][0][prices[p[1]][1][now_ts]])
                              if now_ts in prices[p[1]][1] else p[2] * p[3]
                              for p in open_positions)
        equity_by_ts[now_ts.isoformat()] = {
            "ts": now_ts.isoformat(), "total_value": cash + positions_value,
            "cash": cash, "num_positions": len(open_positions),
        }

    # fecha o que sobrar ao ultimo preco disponivel
    for exit_ts, ticker, qty, entry_price, entry_ts, confidence, pattern_type in open_positions:
        closes, position_of, _ = prices[ticker]
        exit_price = float(closes[position_of.get(exit_ts, len(closes) - 1)])
        cash += qty * exit_price
        trades.append({
            "ticker": ticker, "pattern_type": pattern_type, "entry_ts": entry_ts, "exit_ts": exit_ts,
            "entry_price": entry_price, "exit_price": exit_price, "qty": qty,
            "confidence": confidence, "pnl": qty * (exit_price - entry_price),
            "return_pct": exit_price / entry_price - 1,
        })
    open_positions.clear()

    period_start = events["confirmed_ts"].min()
    period_end = events["confirmed_ts"].max()

    # Buy & hold do indice no mesmo periodo
    benchmark = load_bars(conn, config.BENCHMARK["ticker"], timeframe)
    buy_hold = None
    if not benchmark.empty:
        window = benchmark.loc[
            (benchmark.index >= pd.Timestamp(period_start)) & (benchmark.index <= pd.Timestamp(period_end))
        ]
        if len(window) > 1:
            buy_hold = float(window["close"].iloc[-1] / window["close"].iloc[0] - 1)

    returns = np.array([t["return_pct"] for t in trades])
    curve = [equity_by_ts[k] for k in sorted(equity_by_ts)]

    # Exposicao: que fraccao do capital esta' de facto no mercado. Com um
    # horizonte de poucas horas a carteira passa a maior parte do tempo em
    # caixa, e sem este numero o retorno total parece um fracasso quando na
    # verdade e' so' capital parado.
    exposure = [1 - (p["cash"] / p["total_value"]) for p in curve if p["total_value"] > 0]
    mean_exposure = float(np.mean(exposure)) if exposure else 0.0

    # t sobre DIAS, nao sobre trades: varios trades do mesmo dia partilham o
    # movimento do mercado e nao sao observacoes independentes.
    trade_days = len({str(t["entry_ts"])[:10] for t in trades})
    t_statistic = None
    if len(returns) > 1 and trade_days > 1 and returns.std(ddof=1) > 0:
        t_statistic = float(returns.mean() / (returns.std(ddof=1) / np.sqrt(trade_days)))

    return {
        "mean_exposure": mean_exposure,
        "t_statistic_days": t_statistic,
        "n_trade_days": trade_days,
        "timeframe": timeframe, "horizon_bars": horizon, "market_neutral": market_neutral,
        "threshold": P["confidence_threshold"], "n_trades": len(trades),
        "starting_cash": P["starting_cash"], "final_value": cash,
        "total_return": cash / P["starting_cash"] - 1,
        "win_rate": float((returns > 0).mean()) if len(returns) else None,
        "mean_trade_return": float(returns.mean()) if len(returns) else None,
        "buy_hold_return": buy_hold,
        "random_entry_return": float(np.mean(random_returns)) if random_returns else None,
        "period_start": period_start, "period_end": period_end,
        "trades": trades, "equity_curve": curve,
    }


def store(conn, result: dict) -> str:
    version = f"pf_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    conn.execute(
        """INSERT INTO pattern_portfolio_runs
             (version, created_at, timeframe, horizon_bars, market_neutral, threshold, n_trades,
              starting_cash, final_value, total_return, win_rate, mean_trade_return,
              buy_hold_return, random_entry_return, mean_exposure, t_statistic_days,
              n_trade_days, period_start, period_end, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (version, datetime.now(timezone.utc).isoformat(), result["timeframe"], result["horizon_bars"],
         int(result["market_neutral"]), result["threshold"], result["n_trades"],
         result["starting_cash"], result["final_value"], result["total_return"], result["win_rate"],
         result["mean_trade_return"], result["buy_hold_return"], result["random_entry_return"],
         result["mean_exposure"], result["t_statistic_days"], result["n_trade_days"],
         result["period_start"], result["period_end"],
         "sem comissoes nem slippage; so' o periodo de teste"),
    )
    conn.executemany(
        """INSERT INTO pattern_portfolio_trades
             (run_version, ticker, pattern_type, entry_ts, exit_ts, entry_price, exit_price,
              qty, confidence, pnl, return_pct)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(version, t["ticker"], t["pattern_type"], str(t["entry_ts"]), str(t["exit_ts"]),
          t["entry_price"], t["exit_price"], t["qty"], t["confidence"], t["pnl"], t["return_pct"])
         for t in result["trades"]],
    )
    conn.executemany(
        "INSERT INTO pattern_portfolio_equity (run_version, ts, total_value, cash, num_positions) VALUES (?, ?, ?, ?, ?)",
        [(version, e["ts"], e["total_value"], e["cash"], e["num_positions"]) for e in result["equity_curve"]],
    )
    conn.commit()
    return version


if __name__ == "__main__":
    connection = database.get_connection()
    outcome = run_backtest(connection)
    if outcome is None:
        print("sem modelo de direccao treinado -- corre src/patterns/direction.py primeiro")
    else:
        version = store(connection, outcome)
        currency = config.CURRENCY_SYMBOL
        print(f"=== carteira de padroes ({version}) ===")
        print(f"  periodo: {outcome['period_start'][:10]} a {outcome['period_end'][:10]}")
        print(f"  regra: entra na confirmacao, sai {outcome['horizon_bars']} barras depois, "
              f"confianca >= {outcome['threshold']}")
        print(f"  trades: {outcome['n_trades']}   taxa de acerto: "
              f"{(outcome['win_rate'] or 0):.1%}")
        print(f"  valor final: {currency}{outcome['final_value']:,.2f}  "
              f"({outcome['total_return']:+.2%})")
        t_stat = outcome["t_statistic_days"]
        print(f"  retorno medio por trade: {(outcome['mean_trade_return'] or 0):+.4%}  "
              f"(t={t_stat:.2f} sobre {outcome['n_trade_days']} dias -> "
              f"{'SIGNIFICATIVO' if t_stat and abs(t_stat) > 2 else 'dentro do ruido'})")
        print(f"  exposicao media ao mercado: {outcome['mean_exposure']:.1%}  "
              f"-- a carteira esta' em caixa quase sempre")
        print("  --- referencias no mesmo periodo ---")
        print(f"  entradas ao acaso:     {(outcome['random_entry_return'] or 0):+.4%} por trade")
        print(f"  buy & hold do indice:  {(outcome['buy_hold_return'] or 0):+.2%} "
              f"(NAO e' comparacao justa: 100% investido vs {outcome['mean_exposure']:.0%})")
    connection.close()
