"""Testes da calibracao de mercados de previsao (experiencia W).

DOIS TESTES CARREGAM A RONDA, e tem de vir aos pares:

  `test_a_calibrated_market_shows_no_edge` -- se o aparelho encontrasse desvio
  num mercado construido para ser perfeitamente calibrado, qualquer desvio
  medido nos dados reais seria indistinguivel de um erro de contabilidade.

  `test_a_miscalibrated_market_is_detected` -- e se NAO encontrasse desvio num
  mercado construido com vies, o zero medido nos dados reais nao provaria nada.

Um sem o outro nao vale. E o terceiro que importa e'
`test_selling_is_buying_the_other_side`: sem normalizar compra e venda, metade
da amostra entra com o preco trocado e a curva sai simetrica por construcao --
isto e', sai exactamente o resultado que se procura.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research import prediction_market as W


def _market_frame(rows):
    frame = pd.DataFrame(rows)
    buy = frame["side"].str.upper() == "BUY"
    frame["stake_price"] = np.where(buy, frame["price"], 1.0 - frame["price"])
    hit = frame["outcome"] == frame["winner"]
    frame["won"] = np.where(buy, hit, ~hit)
    frame["dollars"] = frame["size"] * frame["stake_price"]
    frame["desvio_unitario"] = frame["won"].astype(float) - frame["stake_price"]
    return frame


def _synthetic(n_markets=4000, bias=0.0, trades=3, seed=4):
    """Mercados onde a probabilidade verdadeira e' o preco mais `bias`."""
    rng = np.random.default_rng(seed)
    rows = []
    for market in range(n_markets):
        price = float(rng.uniform(0.05, 0.95))
        truth = min(max(price + bias, 0.001), 0.999)
        winner = "Up" if rng.random() < truth else "Down"
        for _ in range(trades):
            rows.append({"condition_id": f"m{market}", "side": "BUY",
                         "outcome": "Up", "price": price, "size": 10.0,
                         "winner": winner})
    return _market_frame(rows)


# --------------------------------------------------------------------------
# 1. O par que carrega a ronda
# --------------------------------------------------------------------------

def test_a_calibrated_market_shows_no_edge():
    frame = _synthetic(bias=0.0)
    mean, low, high = W.market_bootstrap(frame)
    assert abs(mean) < 0.015, mean
    assert low < 0 < high, (low, high)


def test_a_miscalibrated_market_is_detected():
    """Sem este, o zero medido nos dados reais nao provaria nada."""
    frame = _synthetic(bias=0.05, seed=9)
    mean, low, high = W.market_bootstrap(frame)
    assert mean == pytest.approx(0.05, abs=0.02), mean
    assert low > 0, (low, high)


def test_the_calibration_curve_follows_the_price():
    table = W.calibration(_synthetic(bias=0.0, n_markets=8000))
    full = table[table["n"] > 200]
    assert len(full) >= 4
    assert full["desvio"].abs().max() < 0.05, table


def test_the_calibration_curve_shows_a_bias_when_there_is_one():
    table = W.calibration(_synthetic(bias=0.06, n_markets=8000, seed=11))
    full = table[table["n"] > 200]
    assert (full["desvio"] > 0).all(), table


# --------------------------------------------------------------------------
# 2. Compra contra venda -- a armadilha B
# --------------------------------------------------------------------------

def test_selling_is_buying_the_other_side():
    """Vender "Up" a 0.30 e' comprar "Down" a 0.70, e ganha quando "Down" sai."""
    frame = _market_frame([
        {"condition_id": "m", "side": "SELL", "outcome": "Up", "price": 0.30,
         "size": 1.0, "winner": "Down"},
        {"condition_id": "m", "side": "SELL", "outcome": "Up", "price": 0.30,
         "size": 1.0, "winner": "Up"},
    ])
    assert frame["stake_price"].tolist() == [0.70, 0.70]
    assert frame["won"].tolist() == [True, False]


def test_buying_keeps_its_own_price():
    frame = _market_frame([
        {"condition_id": "m", "side": "BUY", "outcome": "Up", "price": 0.30,
         "size": 1.0, "winner": "Up"},
        {"condition_id": "m", "side": "BUY", "outcome": "Up", "price": 0.30,
         "size": 1.0, "winner": "Down"},
    ])
    assert frame["stake_price"].tolist() == [0.30, 0.30]
    assert frame["won"].tolist() == [True, False]


def test_ignoring_the_sell_side_would_fake_calibration():
    """A demonstracao do porque' da armadilha: se as vendas entrassem com o
    preco por normalizar, um mercado com vies de 0.10 leria como calibrado."""
    frame = _synthetic(bias=0.10, n_markets=3000, seed=13)
    honest = W.market_bootstrap(frame)[0]

    broken = frame.copy()
    half = broken.sample(frac=0.5, random_state=3).index
    broken.loc[half, "stake_price"] = 1.0 - broken.loc[half, "stake_price"]
    broken.loc[half, "won"] = ~broken.loc[half, "won"].astype(bool)
    broken["desvio_unitario"] = (broken["won"].astype(float)
                                 - broken["stake_price"])
    assert honest == pytest.approx(0.10, abs=0.02)
    assert abs(W.market_bootstrap(broken)[0]) < abs(honest) / 2


# --------------------------------------------------------------------------
# 3. A unidade de observacao -- a armadilha A
# --------------------------------------------------------------------------

def test_market_bootstrap_is_wider_than_resampling_trades():
    """Mil negocios no mesmo mercado sao UMA moeda lancada uma vez. Reamostrar
    negocios trataria cada um como um lancamento independente."""
    frame = _synthetic(bias=0.0, n_markets=120, trades=200, seed=17)
    _, low, high = W.market_bootstrap(frame, draws=2000)

    rng = np.random.default_rng(5)
    values = frame["desvio_unitario"].to_numpy()
    naive = np.array([rng.choice(values, len(values)).mean() for _ in range(1500)])
    naive_width = np.quantile(naive, 0.975) - np.quantile(naive, 0.025)
    assert (high - low) > 4 * naive_width


def test_bootstrap_counts_markets_not_trades():
    frame = _synthetic(bias=0.0, n_markets=50, trades=4, seed=19)
    assert frame["condition_id"].nunique() == 50
    assert len(frame) == 200


# --------------------------------------------------------------------------
# 4. Aritmetica do pagamento
# --------------------------------------------------------------------------

def test_profit_is_the_binary_payoff():
    """Ganha (1-p) por contrato, perde p. O ponto morto E' o preco."""
    rows = pd.DataFrame({"won": [True, False], "stake_price": [0.80, 0.80],
                         "size": [100.0, 100.0]})
    profit = np.where(rows["won"], rows["size"] * (1 - rows["stake_price"]),
                      -rows["size"] * rows["stake_price"])
    assert profit[0] == pytest.approx(20.0)
    assert profit[1] == pytest.approx(-80.0)
    # A 80% de acerto a esperanca e' exactamente zero.
    assert 0.80 * 20.0 + 0.20 * -80.0 == pytest.approx(0.0)


def test_breakeven_equals_the_price_at_every_price():
    for price in (0.05, 0.20, 0.5, 0.80, 0.95):
        win, loss = 1.0 - price, -price
        assert price * win + (1 - price) * loss == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# 5. Leitura do mercado
# --------------------------------------------------------------------------

def test_a_market_without_a_clean_resolution_has_no_winner():
    """Um mercado por resolver tem precos intermedios. Tomar o maior como
    vencedor transformaria o preco corrente em resultado -- e a calibracao
    sairia perfeita por construcao."""
    import json as _json

    class _Fake:
        payload = [{"markets": [{"conditionId": "0xabc", "closed": False,
                                 "outcomes": _json.dumps(["Up", "Down"]),
                                 "outcomePrices": _json.dumps(["0.62", "0.38"])}]}]

    original = W.fetch
    W.fetch = lambda url, tries=4: _Fake.payload
    try:
        assert W.market_from_slug("x")["winner"] is None
    finally:
        W.fetch = original


def test_a_resolved_market_names_the_winner():
    import json as _json

    payload = [{"markets": [{"conditionId": "0xabc", "closed": True,
                             "outcomes": _json.dumps(["Up", "Down"]),
                             "outcomePrices": _json.dumps(["1", "0"])}]}]
    original = W.fetch
    W.fetch = lambda url, tries=4: payload
    try:
        assert W.market_from_slug("x")["winner"] == "Up"
    finally:
        W.fetch = original


def test_edge_by_band_splits_on_the_coinman_rule():
    frame = _synthetic(bias=0.0, n_markets=3000, seed=23)
    table = W.edge_by_band(frame)
    assert set(table["banda"]) >= {"0.20-0.95"}
    assert (table["ic_lo"] < table["desvio"]).all()
    assert (table["desvio"] < table["ic_hi"]).all()


# --------------------------------------------------------------------------
# 6. Pericia ou sorte
# --------------------------------------------------------------------------

def _wallet_frame(rows):
    frame = pd.DataFrame(rows)
    buy = frame["side"].str.upper() == "BUY"
    frame["stake_price"] = np.where(buy, frame["price"], 1.0 - frame["price"])
    hit = frame["outcome"] == frame["winner"]
    frame["won"] = np.where(buy, hit, ~hit)
    frame["dollars"] = frame["size"] * frame["stake_price"]
    frame["profit"] = np.where(frame["won"],
                               frame["size"] * (1.0 - frame["stake_price"]),
                               -frame["size"] * frame["stake_price"])
    return frame


def test_a_hedged_wallet_has_no_risk():
    """O TESTE QUE JUSTIFICA ESCREVER A VARIANCIA EM CONTRATOS.

    Comprar Up e Down em partes iguais -- a estrategia do gabagool22 -- nao e'
    uma aposta: paga o mesmo saia o que sair. A variancia tem de dar ZERO. Um
    preco medio trataria isto como um apostador com muita sorte."""
    rows = []
    for market in range(30):
        rows += [
            {"wallet": "arb", "condition_id": f"m{market}", "side": "BUY",
             "outcome": "Up", "price": 0.48, "size": 100.0, "winner": "Up"},
            {"wallet": "arb", "condition_id": f"m{market}", "side": "BUY",
             "outcome": "Down", "price": 0.48, "size": 100.0, "winner": "Up"},
        ]
    table = W.wallet_skill(_wallet_frame(rows), min_markets=10)
    assert len(table) == 1
    assert table["variancia"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    # Cada par custou 0.96 e paga 1.00: 4 centimos certos por par.
    assert table["lucro"].iloc[0] == pytest.approx(30 * 4.0, rel=1e-6)


def test_a_coin_flipping_wallet_has_a_z_near_zero():
    rng = np.random.default_rng(7)
    rows = []
    for market in range(400):
        price = float(rng.uniform(0.2, 0.8))
        winner = "Up" if rng.random() < price else "Down"
        rows.append({"wallet": "coin", "condition_id": f"m{market}",
                     "side": "BUY", "outcome": "Up", "price": price,
                     "size": 100.0, "winner": winner})
    table = W.wallet_skill(_wallet_frame(rows), min_markets=10)
    assert abs(table["z"].iloc[0]) < 2.5, table["z"].iloc[0]


def test_a_genuinely_informed_wallet_stands_out():
    """Sem este, o teste anterior nao prova nada: um aparelho que desse sempre
    z proximo de zero passaria os dois."""
    rng = np.random.default_rng(11)
    rows = []
    for market in range(400):
        price = 0.50
        winner = "Up" if rng.random() < 0.62 else "Down"   # sabe mais que o mercado
        rows.append({"wallet": "esperto", "condition_id": f"m{market}",
                     "side": "BUY", "outcome": "Up", "price": price,
                     "size": 100.0, "winner": winner})
    table = W.wallet_skill(_wallet_frame(rows), min_markets=10)
    assert table["z"].iloc[0] > 4.0, table["z"].iloc[0]


def test_wallets_below_the_market_threshold_are_dropped():
    rows = [{"wallet": "pouco", "condition_id": f"m{i}", "side": "BUY",
             "outcome": "Up", "price": 0.5, "size": 10.0, "winner": "Up"}
            for i in range(5)]
    assert W.wallet_skill(_wallet_frame(rows), min_markets=20).empty


def test_luck_baseline_matches_the_normal_tail():
    """Com mil carteiras sem pericia nenhuma, cerca de 23 passam z>2 por acaso.
    E' o varredor de sorte da ronda 17 aplicado a pessoas."""
    rng = np.random.default_rng(3)
    table = pd.DataFrame({"z": rng.normal(0, 1, 1000)})
    result = W.luck_baseline(table, threshold=2.0)
    assert result["esperadas_por_acaso"] == pytest.approx(22.75, abs=0.5)
    assert 0.5 < result["racio"] < 2.0


# --------------------------------------------------------------------------
# 7. A arbitragem do par
# --------------------------------------------------------------------------

def test_pair_arbitrage_finds_a_pair_under_one_dollar():
    frame = pd.DataFrame({
        "condition_id": ["m", "m"], "side": ["BUY", "BUY"],
        "outcome": ["Down", "Up"], "price": [0.55, 0.40],
        "ts": [1000, 1000], "size": [10.0, 10.0]})
    pairs = W.pair_arbitrage(frame)
    assert len(pairs) == 1
    assert pairs["par"].iloc[0] == pytest.approx(0.95)
    assert pairs["margem"].iloc[0] == pytest.approx(0.05)


def test_pair_arbitrage_reports_a_negative_margin_when_there_is_none():
    """Depois das taxas o par passa a custar MAIS de um dolar, e foi por isso
    que o gabagool22 parou. A funcao tem de mostrar isso e nao esconde-lo."""
    frame = pd.DataFrame({
        "condition_id": ["m", "m"], "side": ["BUY", "BUY"],
        "outcome": ["Down", "Up"], "price": [0.56, 0.47],
        "ts": [1000, 1000], "size": [10.0, 10.0]})
    assert W.pair_arbitrage(frame)["margem"].iloc[0] == pytest.approx(-0.03)


def test_pair_arbitrage_needs_both_sides_in_the_same_window():
    frame = pd.DataFrame({
        "condition_id": ["m", "m"], "side": ["BUY", "BUY"],
        "outcome": ["Up", "Up"], "price": [0.40, 0.42],
        "ts": [1000, 1001], "size": [10.0, 10.0]})
    assert W.pair_arbitrage(frame).empty


def test_arbitrageurs_are_separated_from_punters():
    """Nao e' limpeza de dados. Uma carteira coberta tem z infinito porque uma
    arbitragem verdadeira TEM Sharpe infinito; misturada com os apostadores,
    inflaciona a cauda e esconde que os apostadores nao a inflacionam."""
    table = pd.DataFrame({
        "wallet": ["arb", "apostador"],
        "lucro": [100.0, 100.0], "apostado": [10000.0, 10000.0],
        "variancia": [0.0, 10000.0 ** 2 * 0.25],
    })
    table["risco"] = np.sqrt(table["variancia"]) / table["apostado"]
    punters, arbitrage = W.split_by_risk(table)
    assert list(arbitrage["wallet"]) == ["arb"]
    assert list(punters["wallet"]) == ["apostador"]


def test_the_luck_test_is_clean_once_arbitrageurs_are_out():
    """Mil apostadores sem pericia mais cinco arbitragistas: com todos
    juntos a cauda parece enorme; separados, volta ao esperado."""
    rng = np.random.default_rng(5)
    punters = pd.DataFrame({"z": rng.normal(0, 1, 1000),
                            "risco": np.full(1000, 0.30)})
    arbs = pd.DataFrame({"z": [5000.0] * 5, "risco": [0.0] * 5})
    everyone = pd.concat([punters, arbs], ignore_index=True)

    assert W.luck_baseline(everyone, 3.0)["observadas"] >= 5
    clean, _ = W.split_by_risk(everyone)
    result = W.luck_baseline(clean, 3.0)
    assert result["observadas"] <= 6                 # esperado ~1.35


def test_dollar_weighting_follows_the_money():
    """A ponderacao por dolar tem de ser dominada pelas apostas grandes. Se
    desse o mesmo que a ponderacao por negocio, nao estaria a ponderar."""
    rows = []
    for market in range(200):
        # Muitas apostas pequenas num lado calibrado...
        rows += [{"condition_id": f"m{market}", "side": "BUY", "outcome": "Up",
                  "price": 0.50, "size": 1.0, "winner": "Up"} for _ in range(20)]
        # ...e uma aposta enorme a 0.50 que perde sempre.
        rows.append({"condition_id": f"m{market}", "side": "BUY",
                     "outcome": "Down", "price": 0.50, "size": 5000.0,
                     "winner": "Up"})
    frame = _market_frame(rows)
    by_trade = W.calibration(frame, weight="igual")
    by_dollar = W.calibration(frame, weight="dolar")
    assert by_trade["frequencia"].iloc[0] > 0.9      # 20 em 21 negocios ganham
    assert by_dollar["frequencia"].iloc[0] < 0.1     # mas o dinheiro perde


def test_calibration_buckets_cover_the_whole_range():
    assert W.BUCKETS[0] == 0.0 and W.BUCKETS[-1] == 1.0
    assert list(W.BUCKETS) == sorted(W.BUCKETS)
