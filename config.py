"""Configuração central do TradingML. Fonte única de verdade para watchlist,
features, parâmetros por horizonte e paths. Nada de dinheiro real em lado nenhum
deste projeto -- é uma simulação de trading para aprendizagem."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "tradingml.db"
INCOMING_DIR = DATA_DIR / "incoming"
ARCHIVE_DIR = INCOMING_DIR / "archive"
MODELS_DIR = DATA_DIR / "models"

# Watchlist: mega-caps EUA escolhidas por terem cobertura noticiosa diária densa,
# essencial para os booleanos de notícias variarem de facto em vez de ficarem
# quase sempre a False. Simulação em USD. Editável livremente.
WATCHLIST = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
    {"ticker": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Communication Services"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials"},
    {"ticker": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
]
TICKERS = [w["ticker"] for w in WATCHLIST]

# Universo alargado, usado APENAS pelo subsistema de padroes graficos.
#
# Porque e' separado da WATCHLIST: os dois sistemas escalam ao contrario um do
# outro. O sistema de noticias precisa de pesquisa profunda por empresa, feita
# por um agente Claude todos os dias uteis -- 40 empresas seria 5x o trabalho
# e o custo diario. O sistema de padroes nao usa noticias nenhumas (e'
# puramente geometrico) e o que lhe falta e' precisamente VOLUME DE DADOS: com
# 8 tickers a matriz de transicoes ficava a 1.69 observacoes por celula, perto
# de nada. Alargar aqui custa so' chamadas ao yfinance e CPU, zero uso de
# Claude.
#
# Estes tickers entram na tabela watchlist com active=0: existem para a
# deteccao de padroes, mas nao sao transaccionados nem avaliados em noticias.
PATTERN_ONLY = [
    # Technology
    {"ticker": "AVGO", "name": "Broadcom Inc.", "sector": "Technology"},
    {"ticker": "ORCL", "name": "Oracle Corp.", "sector": "Technology"},
    {"ticker": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "sector": "Technology"},
    {"ticker": "INTC", "name": "Intel Corp.", "sector": "Technology"},
    {"ticker": "CSCO", "name": "Cisco Systems Inc.", "sector": "Technology"},
    {"ticker": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
    {"ticker": "QCOM", "name": "Qualcomm Inc.", "sector": "Technology"},
    {"ticker": "TXN", "name": "Texas Instruments", "sector": "Technology"},
    # Communication Services
    {"ticker": "META", "name": "Meta Platforms Inc.", "sector": "Communication Services"},
    {"ticker": "NFLX", "name": "Netflix Inc.", "sector": "Communication Services"},
    {"ticker": "DIS", "name": "Walt Disney Co.", "sector": "Communication Services"},
    # Consumer Discretionary
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Discretionary"},
    {"ticker": "HD", "name": "Home Depot Inc.", "sector": "Consumer Discretionary"},
    {"ticker": "MCD", "name": "McDonald's Corp.", "sector": "Consumer Discretionary"},
    {"ticker": "NKE", "name": "Nike Inc.", "sector": "Consumer Discretionary"},
    # Consumer Staples
    {"ticker": "WMT", "name": "Walmart Inc.", "sector": "Consumer Staples"},
    {"ticker": "PG", "name": "Procter & Gamble Co.", "sector": "Consumer Staples"},
    {"ticker": "KO", "name": "Coca-Cola Co.", "sector": "Consumer Staples"},
    {"ticker": "PEP", "name": "PepsiCo Inc.", "sector": "Consumer Staples"},
    {"ticker": "COST", "name": "Costco Wholesale Corp.", "sector": "Consumer Staples"},
    # Financials
    {"ticker": "BAC", "name": "Bank of America Corp.", "sector": "Financials"},
    {"ticker": "WFC", "name": "Wells Fargo & Co.", "sector": "Financials"},
    {"ticker": "GS", "name": "Goldman Sachs Group", "sector": "Financials"},
    {"ticker": "MS", "name": "Morgan Stanley", "sector": "Financials"},
    {"ticker": "V", "name": "Visa Inc.", "sector": "Financials"},
    # Healthcare
    {"ticker": "UNH", "name": "UnitedHealth Group", "sector": "Healthcare"},
    {"ticker": "LLY", "name": "Eli Lilly & Co.", "sector": "Healthcare"},
    {"ticker": "PFE", "name": "Pfizer Inc.", "sector": "Healthcare"},
    {"ticker": "ABBV", "name": "AbbVie Inc.", "sector": "Healthcare"},
    {"ticker": "MRK", "name": "Merck & Co.", "sector": "Healthcare"},
    # Energy / Industrials
    {"ticker": "CVX", "name": "Chevron Corp.", "sector": "Energy"},
    {"ticker": "COP", "name": "ConocoPhillips", "sector": "Energy"},
    {"ticker": "CAT", "name": "Caterpillar Inc.", "sector": "Industrials"},
    {"ticker": "BA", "name": "Boeing Co.", "sector": "Industrials"},
]

# Referencia de mercado, para rotulos neutros ao mercado (retorno da accao
# menos retorno do indice na MESMA janela). Sem isto mede-se sobretudo a deriva
# geral das accoes: a 40 barras, 53.7% das janelas sao de subida so' porque o
# mercado subiu. Nao entra em PATTERN_TICKERS -- nao se detectam padroes nela,
# so' se usam os seus precos como denominador.
BENCHMARK = {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "sector": "Benchmark"}

# Tudo o que existe na tabela watchlist (os 8 transaccionados + os restantes).
ALL_INSTRUMENTS = WATCHLIST + PATTERN_ONLY + [BENCHMARK]
PATTERN_TICKERS = [w["ticker"] for w in WATCHLIST + PATTERN_ONLY]

CURRENCY_SYMBOL = "$"

# Perguntas booleanas diárias que a tarefa agendada (Claude) responde por ticker.
# Única lista de perguntas para os dois horizontes -- o que muda por horizonte é
# a JANELA de agregação (ver HORIZON_PARAMS.news_window_days), não as perguntas.
BOOLEAN_FEATURES = [
    "good_company_news",        # Houve noticia boa especifica desta empresa hoje?
    "bad_company_news",         # Houve noticia ma especifica desta empresa hoje?
    "peer_impact_news",         # Uma empresa do mesmo setor/concorrente teve noticia que a impacta?
    "sector_momentum_positive", # O setor em geral parece com momentum positivo hoje?
    "macro_event_today",        # Ha evento macro relevante hoje (juros, inflacao, etc.) que a afeta?
]

STARTING_CASH = 100_000.0
BOOTSTRAP_YEARS = 2

HORIZONS = ["SHORT", "LONG"]

# Colunas de technical_features (ja normalizadas/scale-free -- ver src/features.py)
# usadas por cada horizonte, e todos os outros parametros de decisao/simulacao.
HORIZON_PARAMS = {
    "SHORT": {
        "predict_ahead_days": 1,
        "confidence_threshold": 0.6,
        "stop_loss_pct": 0.10,
        "position_size_pct": 0.08,
        "retrain_frequency": "weekly",
        "technical_columns": [
            "ret_1d", "ret_5d", "sma5_ratio", "sma20_ratio",
            "sma_cross_short", "rsi14", "vol_20d", "volume_zscore_20d",
        ],
        "news_window_days": 1,
    },
    "LONG": {
        "predict_ahead_days": 20,
        "confidence_threshold": 0.6,
        "stop_loss_pct": 0.20,
        "position_size_pct": 0.08,
        "retrain_frequency": "monthly",
        "technical_columns": [
            "ret_20d", "ret_60d", "sma50_ratio", "sma200_ratio",
            "sma_cross_long", "rsi50", "vol_60d", "volume_zscore_20d",
        ],
        "news_window_days": 10,
    },
}

MIN_TRAIN_ROWS = 60  # minimo de linhas rotuladas antes de treinar um modelo real


# ---------------------------------------------------------------------------
# Reconhecimento de padroes graficos + previsao de sequencia de padroes
# ---------------------------------------------------------------------------
# Subsistema independente do de cima: em vez de prever direcao do preco a
# partir de features tabulares, deteta padroes classicos de analise tecnica e
# aprende com que probabilidade um padrao e' seguido de outro (cadeia de
# Markov de 1a ordem sobre o alfabeto de padroes).

# Timeframes suportados e quanto historico o yfinance da' para cada um.
# Cada timeframe treina a SUA propria matriz de transicoes -- a estatistica de
# que padrao segue qual nao tem de ser igual a 5 minutos e a 1 hora.
PATTERN_TIMEFRAMES = {
    "1h": {"yf_period": "730d", "yf_interval": "1h", "pivot_window": 5},
    "5m": {"yf_period": "60d", "yf_interval": "5m", "pivot_window": 3},
}
PATTERN_LIVE_TIMEFRAME = "5m"   # o que a pagina web usa por omissao

PATTERN_TYPES = [
    # continuacao / bullish
    "ASCENDING_TRIANGLE",
    "SYMMETRICAL_TRIANGLE",
    "BULL_FLAG",
    "PENNANT",
    "CUP_WITH_HANDLE",
    "RECTANGLE_BOTTOM",
    "FALLING_WEDGE",
    # continuacao / bearish
    "DESCENDING_TRIANGLE",
    "BEAR_FLAG",
    "INVERSE_CUP_WITH_HANDLE",
    "RECTANGLE_TOP",
    "RISING_WEDGE",
    # reversao
    "DOUBLE_BOTTOM",
    "DOUBLE_TOP",
    "DIAMOND_BOTTOM",
    "DIAMOND_TOP",
    "HEAD_AND_SHOULDERS_TOP",
    "INVERSE_HEAD_AND_SHOULDERS",
    "BROADENING_TOP",
    "BROADENING_BOTTOM",
]

PATTERN_BIAS = {
    "ASCENDING_TRIANGLE": "bullish",
    "SYMMETRICAL_TRIANGLE": "neutral",
    "BULL_FLAG": "bullish",
    "PENNANT": "neutral",
    "CUP_WITH_HANDLE": "bullish",
    "RECTANGLE_BOTTOM": "bullish",
    "DESCENDING_TRIANGLE": "bearish",
    "BEAR_FLAG": "bearish",
    "INVERSE_CUP_WITH_HANDLE": "bearish",
    "RECTANGLE_TOP": "bearish",
    "DOUBLE_BOTTOM": "bullish",
    "DOUBLE_TOP": "bearish",
    "DIAMOND_BOTTOM": "bullish",
    "DIAMOND_TOP": "bearish",
    "HEAD_AND_SHOULDERS_TOP": "bearish",
    "INVERSE_HEAD_AND_SHOULDERS": "bullish",
    # Cunhas: o vies e' CONTRARIO ao sentido da inclinacao -- uma cunha
    # ascendente e' bearish (a procura enfraquece enquanto o preco ainda sobe).
    "RISING_WEDGE": "bearish",
    "FALLING_WEDGE": "bullish",
    "BROADENING_TOP": "bearish",
    "BROADENING_BOTTOM": "bullish",
}

# Parametros de deteccao geometrica. Ver README (seccao "Parametros de
# deteccao") para a justificacao de cada valor -- foram escolhidos a olhar
# para quantos padroes cada combinacao produz de facto no historico real,
# nao por defeito arbitrario.
PATTERN_DETECTION = {
    "level_tolerance": 0.02,      # 2%: dois topos/fundos "ao mesmo nivel"
    "min_pattern_bars": 12,       # abaixo disto e' ruido, nao padrao
    "max_pattern_bars": 150,      # acima disto ja' nao e' uma formacao unica
    "min_r2": 0.70,               # qualidade minima do ajuste das trendlines
    # Zona morta deliberada entre os dois: um declive entre flat_slope_max e
    # min_slope nao e' nem "plano" nem "inclinado", e nao produz deteccao.
    # Sem esta separacao, declives ambiguos caem sempre no primeiro ramo do
    # if (triangulo ascendente) e nunca se detecta um simetrico.
    "flat_slope_max": 0.00035,    # |declive| por barra, normalizado ao preco
    "min_slope": 0.0009,          # declive minimo para contar como inclinado
    "cup_min_r2": 0.55,           # taca: parabola sobre precos reais e' mais ruidosa
    "flagpole_min_move": 0.03,    # 3% de movimento para haver "mastro"
    "flagpole_max_bars": 25,
    "flag_max_bars": 60,          # flags/pennants sao formacoes curtas
    "min_pattern_height": 0.015,  # amplitude minima (1.5%) para nao ser ruido
    "hs_head_margin": 0.015,      # cabeca tem de exceder os ombros em 1.5%
    "cup_min_depth": 0.04,        # profundidade minima da chavena
    "cup_max_depth": 0.35,
    "cup_handle_max_retrace": 0.5,  # pega nao pode desfazer >50% da chavena
    "min_quality": 0.55,          # score minimo para aceitar uma deteccao
}

# Modelo de sequencia (cadeia de Markov sobre padroes).
PATTERN_SEQUENCE = {
    "horizon_steps": 4,       # quantos padroes a' frente prever
    "beam_width": 5,          # caminhos mantidos vivos na beam search
    "smoothing_alpha": 0.5,   # prior de Jeffreys, aplicado a' distribuicao marginal
    # Forca do recuo para a marginal (interpolacao de Jelinek-Mercer): com
    # backoff_k observacoes vindas de um padrao, confia-se meio-a-meio na
    # estimativa condicional e na marginal. Ver src/patterns/sequence.py.
    "backoff_k": 10,
    "min_support": 3,         # abaixo de N observacoes, marcar como low-confidence

    # Usar o classificador contextual (src/patterns/classifier.py) no passo 1
    # da cadeia, em vez da cadeia de Markov?
    #
    # DESLIGADO com base em medicao, nao por preguica. Protocolo de 3
    # particoes (60/20/20, algoritmo e peso escolhidos na validacao, teste
    # intocado ate' ao fim), com 26 features incluindo qualidade da deteccao,
    # duracao, amplitude, volume, volatilidade previa e posicao na sessao:
    #
    #   1h: classificador 32.3% vs Markov 32.0%  (+0.3pp, erro-padrao 1.77pp)
    #   5m: classificador 14.9% vs Markov 16.0%  (-1.0pp, erro-padrao 1.81pp)
    #
    # Ambos dentro do ruido. O modelo ATRIBUI peso real ao contexto (48% da
    # massa dos coeficientes, com range_pct a ser o maior de todos), mas isso
    # nao se traduz em acertar mais -- o contexto e' preditivo de alguma
    # coisa, so' nao do TIPO do proximo padrao. Ligar isto so' acrescentaria
    # um artefacto joblib, uma cache e mais um modo de falha por zero ganho.
    #
    # Voltar a por a True quando houver bastante mais historico acumulado e o
    # `py src/patterns/classifier.py` mostrar um ganho acima de 2 erros-padrao.
    "use_classifier": False,
}


# ---------------------------------------------------------------------------
# Previsao de DIRECCAO depois de um padrao (alvo binario)
# ---------------------------------------------------------------------------
# Alvo diferente do de cima: em vez de "que padrao vem a seguir" (20 classes,
# sem vantagem mensuravel), pergunta-se "o preco sobe ou desce nas H barras a
# seguir a este padrao". Duas classes, quase equilibradas (51.6% de subidas a
# H=20), e os mesmos ~3400 exemplos deixam de se espalhar por 20 classes.
PATTERN_DIRECTION = {
    # SO' 1h. A 5m ha' 1911 padroes mas apenas 61 dias distintos de historico,
    # e como os 43 tickers se movem todos com o mercado no mesmo dia, a amostra
    # efectiva e' o numero de DIAS, nao de padroes. Com n=61 o erro-padrao e'
    # ~6.4pp: seria preciso uma vantagem de ~13pp para se detectar seja o que
    # for, o que nao existe em mercados. A 1h ha' 714 dias -> erro-padrao
    # ~1.9pp, e uma vantagem de 4pp ja' seria visivel.
    "timeframes": ["1h"],
    "horizons_bars": [5, 10, 20, 40],   # avaliados todos; o usado escolhe-se na validacao
    "train_fraction": 0.6,
    "validation_fraction": 0.2,
    "n_random_controls": 3000,          # instantes que NAO sao fim de padrao
    "min_train_rows": 300,
}
