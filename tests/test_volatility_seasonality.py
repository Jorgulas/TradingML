"""Testes da experiencia X -- sazonalidade de volatilidade nos mercados binarios.

O teste de cabeca e' o par `test_um_mercado_que_ignora_a_sazonalidade_e_apanhado`
/ `test_um_mercado_que_a_preca_nao_da_sinal`: os dados sinteticos sao gerados
pela propria formula do digital, uma vez com o mercado a usar uma volatilidade
constante e outra com o mercado a usar a volatilidade verdadeira de cada hora.
A medida tem de distinguir os dois. Se nao distinguisse, nada do que se mede
nos dados reais queria dizer coisa alguma.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.research import volatility_seasonality as vs


# ---------------------------------------------------------------------------
# Geradores
# ---------------------------------------------------------------------------

def _hour_sigmas(low: float = 0.8, high: float = 2.0) -> np.ndarray:
    """Perfil horario em forma de corcova, como o que se mede na realidade."""
    hours = np.arange(24)
    return low + (high - low) * np.exp(-((hours - 14) ** 2) / 18.0)


BASE = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())   # meia-noite UTC
WINDOW = 300


def _synthetic(prices_flat: bool, days: int = 20, per_hour: int = 3,
               per_market: int = 40, seed: int = 1,
               sigma_bar: float = 1.3) -> pd.DataFrame:
    """Gera cada mercado a partir de UM percurso, como na realidade.

    O que importa fazer bem: todos os negocios de um mercado e o desfecho desse
    mercado tem de vir do MESMO percurso. Uma primeira versao sorteava a
    distancia de cada negocio independentemente e decidia o desfecho pelo
    negocio mediano -- com isso o preco deixava de ter relacao com o resultado
    do proprio negocio, o sinal afogava-se em ruido, e o teste de cabeca lia
    -0,29 onde devia ler -0,9. Nao era a medida que estava fraca, era o
    gerador que estava errado.

    Com `K = 0` e um passeio `W` de passo unitario por segundo, a probabilidade
    verdadeira de "Up" aos `tau` segundos do fim e' Phi(W_t / raiz(tau)), e o
    desfecho e' `W_fim > 0`. A volatilidade da hora entra apenas pelo RACIO: o
    mercado ve `sigma_h * W_t` e divide pela volatilidade que supoe, portanto
    cota Phi((sigma_h / sigma_bar) * W_t / raiz(tau)). Com sigma_h > sigma_bar
    cota demasiado longe de 0,5 e os favoritos saem caros -- e' o efeito que se
    quer apanhar. Com `prices_flat=False` o mercado usa a volatilidade certa de
    cada hora e nao ha' efeito nenhum.
    """
    rng = np.random.default_rng(seed)
    sigmas = _hour_sigmas()
    markets = days * 24 * per_hour
    # W_0 = 0 em frente, para que `paths[:, k]` seja W_k e `paths[:, WINDOW]`
    # o valor final. Sem o zero, `paths[:, t]` era W_(t+1) mas tau continuava
    # WINDOW - t: um segundo de variancia a mais, o que punha o mercado
    # ligeiramente sub-confiante POR CONSTRUCAO e deixava o controlo com um
    # vies de +0,009 -- ou seja, o controlo deixava de poder detectar um vies
    # meu do mesmo tamanho.
    paths = np.zeros((markets, WINDOW + 1))
    paths[:, 1:] = np.cumsum(rng.standard_normal((markets, WINDOW)), axis=1)
    times = rng.integers(30, WINDOW - 30, (markets, per_market))

    hours = np.tile(np.repeat(np.arange(24), per_hour), days)
    day_index = np.repeat(np.arange(days), 24 * per_hour)
    # `end_epoch` e' o INICIO da janela, como nos dados reais.
    starts = BASE + day_index * 86400 + hours * 3600
    ratio = sigmas[hours] / sigma_bar if prices_flat else np.ones(markets)

    rows = np.arange(markets)[:, None]
    walk = paths[rows, times]
    tau = (WINDOW - times).astype(float)
    quoted = np.clip(stats.norm.cdf(ratio[:, None] * walk / np.sqrt(tau)),
                     0.01, 0.99)
    up_wins = paths[:, WINDOW] > 0

    flip = np.zeros((markets, per_market), dtype=bool)
    flip[:, 1::2] = True                      # metade dos negocios do lado "Down"
    frame = pd.DataFrame({
        "condition_id": np.repeat([f"m{i}" for i in range(markets)], per_market),
        "outcome": np.where(flip.ravel(), "Down", "Up"),
        "price": np.where(flip.ravel(), 1.0 - quoted.ravel(), quoted.ravel()),
        "size": 10.0,
        "ts": (starts[:, None] + times).ravel(),
        "winner": np.repeat(np.where(up_wins, "Up", "Down"), per_market),
        "end_epoch": np.repeat(starts, per_market),
        "family": "btc5m",
    })
    return vs.derive(frame)


def _panel(amplitude: float = 3.0, days: int = 20, per_hour: int = 60,
           seed: int = 2) -> pd.DataFrame:
    """Painel de minutos com um perfil horario conhecido."""
    rng = np.random.default_rng(seed)
    factors = 1.0 + (amplitude - 1.0) * (np.arange(24) == 14)
    rows = []
    for day in range(days):
        for hour in range(24):
            rows.append(pd.DataFrame({
                "ts": pd.Timestamp("2026-01-01", tz="UTC")
                      + pd.Timedelta(days=day, hours=hour),
                "bucket": hour,
                "day": pd.Timestamp("2026-01-01", tz="UTC")
                       + pd.Timedelta(days=day),
                "abs_return": np.abs(rng.normal(0, factors[hour], per_hour)),
            }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Parte A -- o perfil
# ---------------------------------------------------------------------------

def test_o_perfil_recupera_a_forma_injectada():
    panel = _panel(amplitude=3.0)
    profile = vs.profile_of(panel)
    assert profile.idxmax() == 14
    # |N(0, 3)| tem media 3x a de |N(0, 1)|; o factor normalizado reflecte-o.
    assert profile[14] == pytest.approx(3.0, rel=0.1)


def test_a_amplitude_e_o_racio_entre_a_hora_maior_e_a_menor():
    profile = pd.Series({0: 0.5, 1: 1.0, 2: 2.0})
    assert vs.amplitude(profile) == pytest.approx(4.0)


def test_um_painel_sem_sazonalidade_da_amplitude_perto_de_um():
    panel = _panel(amplitude=1.0, per_hour=400)
    assert vs.amplitude(vs.profile_of(panel)) < 1.2


def test_o_score_distingue_um_perfil_certo_de_um_embaralhado():
    panel = _panel(amplitude=4.0, per_hour=200)
    profile = vs.profile_of(panel)
    real = vs.score_profile(panel, profile)
    media, alto = vs.shuffled_score(panel, profile, draws=20)
    assert real > 0.05
    assert abs(media) < 0.01
    assert real > alto


def test_o_chao_de_ruido_e_quase_toda_a_dispersao_se_o_perfil_for_constante():
    """Num painel onde o perfil e' identico todos os dias, a dispersao
    observada tem de ficar perto do chao simulado -- se o chao ficasse muito
    abaixo, estaria a subestimar o ruido e eu leria deriva onde nao ha'."""
    panel = _panel(amplitude=2.0, days=16, per_hour=50)
    observed = vs.dispersion_by_hour(panel)["dispersao"].median()
    floor = vs.dispersion_floor(panel, draws=12).median()
    assert observed == pytest.approx(floor, rel=0.25)


def test_o_chao_fica_abaixo_da_dispersao_quando_o_perfil_deriva():
    """E com deriva a mais, o chao tem de ficar claramente abaixo."""
    rng = np.random.default_rng(7)
    rows = []
    for day in range(16):
        peak = 8 if day % 2 == 0 else 18        # o perfil salta de hora
        for hour in range(24):
            factor = 3.0 if hour == peak else 1.0
            rows.append(pd.DataFrame({
                "bucket": hour,
                "day": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                "abs_return": np.abs(rng.normal(0, factor, 50)),
            }))
    panel = pd.concat(rows, ignore_index=True)
    observed = vs.dispersion_by_hour(panel)["dispersao"].max()
    floor = vs.dispersion_floor(panel, draws=12).max()
    assert observed > floor * 1.8


def test_o_oraculo_ganha_e_o_envelhecimento_fica_plano():
    panel = _panel(amplitude=3.0, days=24, per_hour=80)
    table = vs.profile_age(panel, lags=(0, 1, 7), window=5).set_index("atraso_dias")
    assert table.loc[0, "spearman"] >= table.loc[1, "spearman"]
    # O perfil e' constante por construcao: um perfil velho nao perde nada.
    assert table.loc[7, "spearman"] == pytest.approx(table.loc[1, "spearman"],
                                                     abs=0.03)


def test_mais_historico_nao_piora_o_perfil():
    panel = _panel(amplitude=3.0, days=24, per_hour=60)
    table = vs.history_needed(panel, spans=(1, 7)).set_index("dias_usados")
    assert table.loc[7, "spearman"] >= table.loc[1, "spearman"] - 0.02


def test_um_salto_entre_pedacos_nao_entra_como_retorno_de_um_minuto():
    """Os pedacos de 7 dias do yfinance juntam-se com descontinuidades. Se o
    salto entrasse como um retorno de um minuto, seria o maior |retorno| da
    amostra e cairia na hora da costura -- a mesma classe de erro do limite de
    sessao da ronda 17."""
    minutes = pd.date_range("2026-01-01", periods=10, freq="1min", tz="UTC")
    later = pd.date_range("2026-01-05", periods=10, freq="1min", tz="UTC")
    index = minutes.append(later)
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": np.r_[np.full(10, 100.0), np.full(10, 150.0)]},
                        index=index)
    panel = vs.minute_panel(bars)
    assert panel["abs_return"].max() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Parte B -- preco visto do lado "Up"
# ---------------------------------------------------------------------------

def test_o_preco_do_lado_up_nao_depende_de_quem_comprou():
    """Vender "Up" a 0,30 e comprar "Down" a 0,70 cotam o MESMO mercado. Se
    `p_up` dependesse do lado, metade da amostra entrava com o preco trocado --
    e nesta experiencia, ao contrario da ronda 22, o lado e' irrelevante."""
    frame = pd.DataFrame([
        {"condition_id": "a", "outcome": "Up", "price": 0.30, "size": 1.0,
         "ts": 1150, "winner": "Up", "end_epoch": 1000, "family": "btc5m"},
        {"condition_id": "a", "outcome": "Down", "price": 0.70, "size": 1.0,
         "ts": 1150, "winner": "Up", "end_epoch": 1000, "family": "btc5m"},
    ])
    out = vs.derive(frame)
    assert out["p_up"].tolist() == pytest.approx([0.30, 0.30])


def test_a_nitidez_e_positiva_quando_o_favorito_ganha():
    frame = pd.DataFrame([
        {"condition_id": "a", "outcome": "Up", "price": 0.80, "size": 1.0,
         "ts": 1150, "winner": "Up", "end_epoch": 1000, "family": "btc5m"},
        {"condition_id": "b", "outcome": "Up", "price": 0.20, "size": 1.0,
         "ts": 1150, "winner": "Down", "end_epoch": 1000, "family": "btc5m"},
    ])
    out = vs.derive(frame)
    # Favorito "Up" a 0,80 que ganha: +0,20. Azarao "Up" a 0,20 que perde: o
    # favorito era "Down" e ganhou, logo tambem +0,20. Somam-se, nao se anulam.
    assert out["nitidez"].tolist() == pytest.approx([0.20, 0.20])


def test_o_desvio_cru_anula_se_e_a_nitidez_nao():
    """Dois mercados onde o mercado acertou: um cotado "Up" a 0,90 que deu Up,
    outro cotado "Up" a 0,10 que deu Down. O desvio cru de um e' +0,10 e do
    outro -0,10, portanto somar desvios da' ZERO e esconde que o mercado
    acertou duas vezes. E' por isso que a estatistica desta ronda e' a nitidez
    e nao o desvio: so' a correccao de sinal faz os dois casos somarem."""
    frame = pd.DataFrame([
        {"condition_id": "a", "outcome": "Up", "price": 0.90, "size": 1.0,
         "ts": 1150, "winner": "Up", "end_epoch": 1000, "family": "btc5m"},
        {"condition_id": "b", "outcome": "Up", "price": 0.10, "size": 1.0,
         "ts": 1150, "winner": "Down", "end_epoch": 1000, "family": "btc5m"},
    ])
    out = vs.derive(frame)
    assert out["desvio"].sum() == pytest.approx(0.0)
    assert out["nitidez"].tolist() == pytest.approx([0.10, 0.10])


# ---------------------------------------------------------------------------
# Parte B -- o teste de cabeca
# ---------------------------------------------------------------------------

def test_um_mercado_que_ignora_a_sazonalidade_e_apanhado():
    """Mercado a cotar com volatilidade CONSTANTE num mundo onde ela varia por
    hora: nas horas volateis os favoritos ficam caros, logo a nitidez tem de
    cair com a volatilidade da hora."""
    frame = _synthetic(prices_flat=True, seed=11)
    profile = pd.Series(_hour_sigmas(), index=range(24))
    result = vs.hour_correlation(vs.by_hour(frame), profile)
    assert result["horas"] == 24
    assert result["spearman"] < -0.6
    assert result["p_spearman"] < 0.01


def test_um_mercado_que_a_preca_nao_da_sinal():
    """O controlo do teste anterior. O mesmo gerador, a mesma amostra, a mesma
    medida -- muda so' que o mercado usa a volatilidade certa de cada hora. Se
    isto tambem desse negativo, o sinal vinha da medida e nao do mercado."""
    frame = _synthetic(prices_flat=False, seed=11)
    profile = pd.Series(_hour_sigmas(), index=range(24))
    result = vs.hour_correlation(vs.by_hour(frame), profile)
    assert abs(result["spearman"]) < 0.45


def test_o_placebo_perto_de_meio_atenua_o_efeito():
    """Perto de 0,5 o preco quase nao contem informacao sobre sigma, porque
    Phi(0) = 0,5 para qualquer volatilidade. "Quase" e nao "nada": a banda vai
    de 0,44 a 0,56, logo sobra um resto de informacao e a fisica preve
    ATENUACAO, nao desaparecimento. O declive tem de encolher por um factor
    grande; exigir zero seria exigir mais do que a teoria diz.

    E a comparacao tem de ser de DECLIVES e nao da agitacao do perfil: a banda
    central tem seis vezes menos negocios, por isso o seu perfil horario e'
    mais agitado por ruido de amostra e uma comparacao de amplitudes cruas
    diria que o efeito la' e' quase igual."""
    frame = _synthetic(prices_flat=True, seed=12)
    profile = pd.Series(_hour_sigmas(), index=range(24))
    split = vs.placebo_split(frame, profile)
    assert split["informativo"]["declive"] < -0.10
    assert split["placebo"]["declive"] > split["informativo"]["declive"] * 0.6


# ---------------------------------------------------------------------------
# Parte B -- composicao e reamostragem
# ---------------------------------------------------------------------------

def test_a_padronizacao_apaga_uma_diferenca_que_e_so_de_mistura():
    """Duas horas com a MESMA nitidez dentro de cada celula mas misturas
    diferentes de preco. A media crua separa-as; a padronizada nao pode."""
    rows = []
    for hour, share in ((3, 0.9), (4, 0.1)):
        for position in range(400):
            central = position < int(400 * share)
            price = 0.50 if central else 0.80
            # Dentro de cada celula a nitidez e' a mesma nas duas horas.
            won = "Up" if (central or position % 2 == 0) else "Down"
            rows.append({"condition_id": f"m{hour}-{position}",
                         "outcome": "Up", "price": price, "size": 1.0,
                         "ts": BASE + hour * 3600 + 150,
                         "winner": won,
                         "end_epoch": BASE + hour * 3600,
                         "family": "btc5m"})
    frame = vs.derive(pd.DataFrame(rows))
    table = vs.by_hour(frame).set_index("hora")
    assert abs(table.loc[3, "cru"] - table.loc[4, "cru"]) > 0.05
    assert table.loc[3, "padronizado"] == pytest.approx(
        table.loc[4, "padronizado"], abs=0.02)


def test_a_padronizacao_usa_pesos_de_fora_quando_lhe_sao_dados():
    frame = _synthetic(prices_flat=True, days=8, per_hour=1, seed=13)
    weights = vs.cell_weights(frame)
    half = frame.iloc[: len(frame) // 2]
    own = vs.by_hour(half)["padronizado"].to_numpy()
    given = vs.by_hour(half, weights=weights)["padronizado"].to_numpy()
    assert not np.allclose(own, given, atol=1e-9, equal_nan=True)


def test_reamostrar_dias_da_intervalo_mais_largo_que_reamostrar_negocios():
    """A armadilha B. O erro que alguem cometeria de facto nao e' reamostrar as
    24 linhas da tabela -- e' reamostrar NEGOCIOS, tratando cada um como uma
    observacao independente, e so' depois agregar por hora. As 24 horas do mesmo
    dia partilham o regime desse dia, portanto esse intervalo sai estreito."""
    frame = _synthetic(prices_flat=True, days=12, per_hour=2, seed=14)
    profile = pd.Series(_hour_sigmas(), index=range(24))
    _, low, high = vs.day_bootstrap(frame, profile, draws=120, seed=3)

    weights = vs.cell_weights(frame)
    rng = np.random.default_rng(3)
    naive = []
    for _ in range(120):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        result = vs.hour_correlation(vs.by_hour(sample, weights=weights),
                                     profile)
        if result:
            naive.append(result["spearman"])
    naive_width = np.quantile(naive, 0.975) - np.quantile(naive, 0.025)
    assert (high - low) > naive_width * 1.5


def test_o_bootstrap_de_mercado_e_mais_largo_que_o_de_negocios():
    """A armadilha A da ronda 22, outra vez: 40 negocios partilham UMA
    resolucao, portanto o erro-padrao ingenuo de negocios subestima."""
    frame = _synthetic(prices_flat=True, days=10, per_hour=1, seed=15)
    _, low, high = vs.market_bootstrap(frame, draws=800)
    naive = frame["nitidez"].std() / np.sqrt(len(frame))
    assert (high - low) > 1.5 * 2 * 1.96 * naive


# ---------------------------------------------------------------------------
# Vol implicita e strike
# ---------------------------------------------------------------------------

def test_a_inversao_recupera_a_sigma_que_gerou_os_precos():
    """Gera-se o preco com a formula e inverte-se: sem ruido tem de voltar o
    mesmo sigma. Se nao voltasse, a magnitude em unidades de volatilidade nao
    queria dizer nada."""
    sigma, strike, tau = 0.0004, 100_000.0, 120.0
    deviations = np.array([-400.0, -200.0, 150.0, 300.0])
    z = deviations / (sigma * strike * np.sqrt(tau))
    prices = stats.norm.cdf(z)
    frame = pd.DataFrame({
        "condition_id": [f"m{i}" for i in range(4)],
        "outcome": "Up", "price": prices, "size": 1.0,
        "ts": int(BASE + WINDOW - tau), "winner": "Up",
        "end_epoch": BASE, "family": "btc5m"})
    out = vs.derive(frame)
    out["spot"] = strike + deviations
    out["strike"] = strike
    recovered = vs.implied_sigma(out)["sigma_implicita"]
    assert len(recovered) == 4
    assert recovered.to_numpy() == pytest.approx(np.full(4, sigma), rel=1e-6)


def test_o_strike_errado_e_apanhado_pela_concordancia_de_sinal():
    """Se `K` esta' certo, o sinal de `S_t - K` concorda com o sinal de
    `p - 0,5`. Com um `K` deslocado para longe, todos os negocios ficam do
    mesmo lado e a concordancia desce para a fraccao de favoritos "Up"."""
    frame = pd.DataFrame({
        "condition_id": [f"m{i}" for i in range(4)],
        "outcome": "Up", "price": [0.80, 0.75, 0.25, 0.20], "size": 1.0,
        "ts": BASE + 150, "winner": "Up",
        "end_epoch": BASE, "family": "btc5m"})
    out = vs.derive(frame)
    out["strike"] = 100.0
    out["spot"] = [101.0, 101.0, 99.0, 99.0]
    assert vs.strike_check(out)["concordancia"] == pytest.approx(1.0)
    out["strike"] = 50.0                      # strike errado, muito abaixo
    assert vs.strike_check(out)["concordancia"] == pytest.approx(0.5)


def test_ruido_no_subjacente_atenua_a_sigma_implicita():
    """A armadilha D nao e' hipotetica. Com `S_t` conhecido a inversao devolve
    o sigma certo; com ruido de resolucao de um minuto no numerador a mediana
    sai enviesada. O teste fixa a direccao do erro para que a leitura dos
    dados reais seja feita com ela em mente."""
    rng = np.random.default_rng(4)
    sigma, strike, tau, n = 0.0004, 100_000.0, 120.0, 4000
    deviations = rng.normal(0, sigma * strike * np.sqrt(tau), n)
    prices = np.clip(stats.norm.cdf(deviations / (sigma * strike * np.sqrt(tau))),
                     0.02, 0.98)
    frame = pd.DataFrame({
        "condition_id": [f"m{i}" for i in range(n)],
        "outcome": "Up", "price": prices, "size": 1.0,
        "ts": int(BASE + WINDOW - tau), "winner": "Up",
        "end_epoch": BASE, "family": "btc5m"})
    out = vs.derive(frame)
    out["strike"] = strike
    out["spot"] = strike + deviations
    clean = vs.implied_sigma(out)["sigma_implicita"].median()
    noisy_frame = out.copy()
    noisy_frame["spot"] = out["spot"] + rng.normal(0, 0.3 * sigma * strike
                                                   * np.sqrt(60), n)
    dirty = vs.implied_sigma(noisy_frame)["sigma_implicita"].median()
    assert clean == pytest.approx(sigma, rel=0.05)
    assert abs(dirty - sigma) > abs(clean - sigma)


def test_o_ruido_do_minuto_ACHATA_o_perfil_implicito():
    """A direccao da armadilha D no PERFIL, que e' a leitura de que depende a
    conclusao da ronda.

    O argumento e': o ruido de resolucao de um minuto entra no numerador de
    `(S_t - K) / (z * raiz(tau))` e enviesa o NIVEL de sigma, mas por igual em
    todas as horas, logo a FORMA sobrevive. Isso nao e' bem verdade, e e' por
    isso que vale a pena medi-lo: o ruido pesa MAIS onde o desvio verdadeiro e'
    pequeno, isto e' nas horas calmas, portanto levanta as horas calmas mais do
    que as agitadas e ACHATA o perfil. O declive medido do implicito sobre o
    realizado e' por isso um LIMITE INFERIOR -- um declive de 1,0 nos dados
    reais nao pode ser ruido a inflacionar a conclusao, so' a deprimi-la."""
    rng = np.random.default_rng(9)
    strike, n = 100_000.0, 24 * 1500
    hours = np.tile(np.arange(24), n // 24)
    factors = 1.0 + 1.4 * np.exp(-((hours - 14) ** 2) / 18.0)
    sigma = 4e-5 * factors                 # perfil verdadeiro, amplitude 2,4x
    tau = rng.integers(60, 300, n).astype(float)
    scale = sigma * strike * np.sqrt(tau)
    deviations = rng.normal(0, scale)
    prices = np.clip(stats.norm.cdf(deviations / scale), 0.02, 0.98)

    frame = vs.derive(pd.DataFrame({
        "condition_id": [f"m{i}" for i in range(n)], "outcome": "Up",
        "price": prices, "size": 1.0,
        "ts": BASE + hours * 3600 + (WINDOW - tau).astype(int),
        "winner": "Up", "end_epoch": BASE + hours * 3600,
        "family": "btc5m"}), step=WINDOW)
    frame["strike"] = strike
    frame["spot"] = strike + deviations
    limpo = vs.implied_profile(vs.implied_sigma(frame))

    sujo = frame.copy()
    sujo["spot"] = frame["spot"] + rng.normal(0, 4e-5 * strike * np.sqrt(60), n)
    achatado = vs.implied_profile(vs.implied_sigma(sujo))

    verdade = pd.Series(1.0 + 1.4 * np.exp(-((np.arange(24) - 14) ** 2) / 18.0),
                        index=range(24))
    verdade = verdade / verdade.mean()
    assert vs.amplitude(limpo) == pytest.approx(vs.amplitude(verdade), rel=0.15)
    assert vs.amplitude(achatado) < vs.amplitude(limpo)


def test_o_declive_implicito_sobre_realizado_da_um_quando_coincidem():
    """A escala da medida directa. Se a vol implicita for exactamente a
    realizada, o declive tem de ser 1,0; se for constante, 0,0. Sem este teste
    nao se sabe se "0,751" quer dizer tres quartos da sazonalidade ou outra
    coisa qualquer."""
    real = pd.Series(1.0 + 1.4 * np.exp(-((np.arange(24) - 14) ** 2) / 18.0),
                     index=range(24))
    real = real / real.mean()
    for escala, esperado in ((1.0, 1.0), (0.5, 0.5), (0.0, 0.0)):
        implicita = 1.0 + escala * (real - 1.0)
        sigma = pd.DataFrame({
            "hora": np.repeat(np.arange(24), 50),
            "sigma_implicita": np.repeat(implicita.to_numpy() * 4e-5, 50)})
        fit = vs.implied_vs_realised(sigma, real)
        assert fit["declive"] == pytest.approx(esperado, abs=0.02)


def test_a_juncao_usa_o_minuto_do_negocio_e_a_abertura_da_janela():
    index = pd.date_range("2026-01-01 00:00", periods=6, freq="1min", tz="UTC")
    bars = pd.DataFrame({"open": [10, 11, 12, 13, 14, 15],
                         "high": 20.0, "low": 1.0,
                         "close": [100, 101, 102, 103, 104, 105]}, index=index)
    start = int(index[0].timestamp())
    frame = vs.derive(pd.DataFrame([{
        "condition_id": "a", "outcome": "Up", "price": 0.7, "size": 1.0,
        "ts": start + 190,                     # cai no minuto 3
        "winner": "Up", "end_epoch": start, "family": "btc5m"}]))
    out = vs.join_underlying(frame, bars)
    assert out["spot"].iloc[0] == 103.0        # fecho do minuto 3
    assert out["strike"].iloc[0] == 10.0       # abertura do minuto 0


# ---------------------------------------------------------------------------
# Parte C -- persistencia e poder
# ---------------------------------------------------------------------------

def test_a_persistencia_e_alta_quando_o_perfil_e_o_mesmo_nas_duas_metades():
    frame = _synthetic(prices_flat=True, days=20, per_hour=3, seed=16)
    result = vs.persistence(frame)
    assert result["dias_total"] >= 6
    assert result["spearman"] > 0.5


def test_a_persistencia_cai_quando_o_perfil_nao_e_o_mesmo():
    """Dois conjuntos do MESMO tamanho, 24 dias cada. Num, o mercado usa
    volatilidade constante todos os dias, logo o perfil do erro repete-se. No
    outro, os doze primeiros dias sao assim e os doze seguintes tem o mercado a
    cotar bem cada hora -- metade com perfil, metade sem. A medida tem de
    separar os dois.

    Uma primeira versao comparava 12 dias contra 12+12, e o proprio baseline ja'
    estava em 0,12, nivel de ruido: nao havia margem para a comparacao dizer
    nada. Com amostras iguais e grandes a diferenca aparece.

    E O NUMERO QUE ESTE TESTE FIXA DE FACTO E' A VARIANCIA: sobre 8 sementes,
    com efeito real da' +0,551 com desvio-padrao 0,102, e sem efeito nenhum da'
    -0,005 com desvio-padrao 0,268. Uma unica estimativa de persistencia tem
    portanto um desvio-padrao de quase 0,27 sob a hipotese nula, e por isso o
    teste compara MEDIAS de varias sementes em vez de uma so'. E' tambem o
    numero com que se lerao os dados reais: la' so' ha' uma amostra, e qualquer
    valor dentro de +-0,5 nao distingue nada."""
    sementes = range(300, 305)
    puro = [vs.persistence(_synthetic(prices_flat=True, days=24, per_hour=3,
                                      seed=s))["spearman"] for s in sementes]
    misturado = []
    for s in sementes:
        primeira = _synthetic(prices_flat=True, days=12, per_hour=3, seed=s)
        segunda = _synthetic(prices_flat=False, days=12, per_hour=3, seed=s + 50)
        segunda = segunda.copy()
        segunda["dia"] = segunda["dia"] + pd.Timedelta(days=100)
        misturado.append(vs.persistence(
            pd.concat([primeira, segunda], ignore_index=True))["spearman"])

    assert np.mean(puro) > 0.4
    assert np.mean(misturado) < 0.25
    assert np.mean(puro) - np.mean(misturado) > 0.3


def test_o_minimo_detectavel_encolhe_com_a_raiz_dos_mercados():
    pequeno = _synthetic(prices_flat=True, days=10, per_hour=1, seed=19)
    grande = _synthetic(prices_flat=True, days=40, per_hour=1, seed=19)
    a = vs.detectable(pequeno)["mde_global"]
    b = vs.detectable(grande)["mde_global"]
    assert b < a
    assert b == pytest.approx(a / 2.0, rel=0.35)


def test_o_minimo_por_hora_e_maior_que_o_global():
    """Dividir por hora corta a amostra em 24. Sem este numero a' vista,
    qualquer hora individual parece significativa."""
    frame = _synthetic(prices_flat=True, days=20, per_hour=1, seed=20)
    power = vs.detectable(frame)
    assert power["mde_por_hora"] > power["mde_global"] * 4


def test_sem_dados_as_funcoes_devolvem_vazio_em_vez_de_rebentar():
    empty = pd.DataFrame()
    assert vs.derive(empty).empty
    assert vs.by_hour(empty).empty
    assert vs.hour_correlation(pd.DataFrame(), pd.Series(dtype=float)) == {}
    assert vs.persistence(empty) == {}
    assert vs.detectable(empty) == {}
    assert np.isnan(vs.market_bootstrap(empty)[0])
