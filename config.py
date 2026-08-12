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
