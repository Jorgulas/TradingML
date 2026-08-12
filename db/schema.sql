-- TradingML schema. Simulacao de trading (paper trading) -- nunca dinheiro real.
--
-- Camada de mercado/noticias (watchlist, prices, technical_features, news_features)
-- e partilhada pelos dois horizontes e calculada uma vez por dia.
-- Camada de decisao (outcomes, predictions, trades, positions, portfolio_state,
-- equity_curve, model_versions) tem `horizon` na chave: 'SHORT' ou 'LONG'.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS watchlist (
    ticker      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    sector      TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    added_date  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prices (
    ticker       TEXT NOT NULL REFERENCES watchlist(ticker),
    date         TEXT NOT NULL,
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL NOT NULL,
    volume       INTEGER,
    ingested_at  TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- date = dia a prever; todos os valores usam apenas dados ate' date-1
-- (rolling calculado e depois deslocado 1 sessao -- ver src/features.py).
-- Colunas normalizadas/scale-free de proposito (ratios e z-scores, nao precos
-- brutos) para que um unico modelo faca sentido pooled entre tickers.
CREATE TABLE IF NOT EXISTS technical_features (
    ticker              TEXT NOT NULL REFERENCES watchlist(ticker),
    date                TEXT NOT NULL,
    ret_1d              REAL,
    ret_5d              REAL,
    ret_20d             REAL,
    ret_60d             REAL,
    sma5_ratio          REAL,
    sma20_ratio         REAL,
    sma50_ratio         REAL,
    sma200_ratio        REAL,
    sma_cross_short     REAL,
    sma_cross_long      REAL,
    rsi14               REAL,
    rsi50               REAL,
    vol_20d             REAL,
    vol_60d             REAL,
    volume_zscore_20d   REAL,
    computed_at         TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- date = dia a que a avaliacao de noticias diz respeito. Escrita exclusiva da
-- tarefa agendada, via src/record_daily_features.py (nunca SQL direto do LLM).
CREATE TABLE IF NOT EXISTS news_features (
    ticker                      TEXT NOT NULL REFERENCES watchlist(ticker),
    date                        TEXT NOT NULL,
    good_company_news           INTEGER NOT NULL DEFAULT 0,
    bad_company_news            INTEGER NOT NULL DEFAULT 0,
    peer_impact_news            INTEGER NOT NULL DEFAULT 0,
    sector_momentum_positive    INTEGER NOT NULL DEFAULT 0,
    macro_event_today           INTEGER NOT NULL DEFAULT 0,
    notes                       TEXT,
    filled_by                   TEXT NOT NULL DEFAULT 'llm_agent',
    filled_at                   TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- Label por horizonte. direction fica NULL ate' target_close ser conhecido
-- (1 sessao depois para SHORT, 20 sessoes depois para LONG).
CREATE TABLE IF NOT EXISTS outcomes (
    ticker          TEXT NOT NULL REFERENCES watchlist(ticker),
    date            TEXT NOT NULL,
    horizon         TEXT NOT NULL CHECK (horizon IN ('SHORT', 'LONG')),
    ref_close       REAL NOT NULL,
    target_close    REAL,
    direction       INTEGER CHECK (direction IN (0, 1)),
    resolved_at     TEXT,
    PRIMARY KEY (ticker, date, horizon)
);

CREATE TABLE IF NOT EXISTS predictions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                TEXT NOT NULL REFERENCES watchlist(ticker),
    date                  TEXT NOT NULL,
    horizon               TEXT NOT NULL CHECK (horizon IN ('SHORT', 'LONG')),
    predicted_direction   INTEGER NOT NULL CHECK (predicted_direction IN (0, 1)),
    confidence            REAL NOT NULL,
    model_version         TEXT NOT NULL,
    feature_snapshot      TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    UNIQUE (ticker, date, horizon)
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL REFERENCES watchlist(ticker),
    date            TEXT NOT NULL,
    horizon         TEXT NOT NULL CHECK (horizon IN ('SHORT', 'LONG')),
    side            TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    qty             REAL NOT NULL,
    price           REAL NOT NULL,
    cash_after      REAL NOT NULL,
    reason          TEXT NOT NULL,
    prediction_id   INTEGER REFERENCES predictions(id),
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    ticker        TEXT NOT NULL REFERENCES watchlist(ticker),
    horizon       TEXT NOT NULL CHECK (horizon IN ('SHORT', 'LONG')),
    qty           REAL NOT NULL,
    avg_price     REAL NOT NULL,
    opened_date   TEXT NOT NULL,
    PRIMARY KEY (ticker, horizon)
);

-- Uma linha por horizonte (singleton por horizonte). Unico livro-razao de caixa.
CREATE TABLE IF NOT EXISTS portfolio_state (
    horizon      TEXT PRIMARY KEY CHECK (horizon IN ('SHORT', 'LONG')),
    cash         REAL NOT NULL,
    updated_at   TEXT NOT NULL
);

-- Uma linha por dia de mercado POR horizonte, mesmo sem trade nesse dia
-- (mark-to-market diario -- sem isto o grafico fica enganador).
CREATE TABLE IF NOT EXISTS equity_curve (
    date              TEXT NOT NULL,
    horizon           TEXT NOT NULL CHECK (horizon IN ('SHORT', 'LONG')),
    cash              REAL NOT NULL,
    positions_value   REAL NOT NULL,
    total_value       REAL NOT NULL,
    pnl_abs           REAL NOT NULL,
    pnl_pct           REAL NOT NULL,
    num_positions     INTEGER NOT NULL,
    PRIMARY KEY (date, horizon)
);

CREATE TABLE IF NOT EXISTS model_versions (
    version                         TEXT PRIMARY KEY,
    horizon                         TEXT NOT NULL CHECK (horizon IN ('SHORT', 'LONG')),
    trained_at                      TEXT NOT NULL,
    algorithm                       TEXT NOT NULL,
    hyperparams_json                TEXT NOT NULL,
    train_start_date                TEXT NOT NULL,
    train_end_date                  TEXT NOT NULL,
    n_train_rows                    INTEGER NOT NULL,
    cv_accuracy                     REAL,
    baseline_majority_accuracy      REAL,
    baseline_persistence_accuracy   REAL,
    notes                           TEXT
);

-- Visibilidade operacional do pipeline diario (nao e' por horizonte).
CREATE TABLE IF NOT EXISTS run_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         TEXT NOT NULL,
    stage        TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('OK', 'WARN', 'ERROR')),
    message      TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(date, horizon);
CREATE INDEX IF NOT EXISTS idx_trades_ticker_horizon ON trades(ticker, horizon);
CREATE INDEX IF NOT EXISTS idx_equity_curve_horizon ON equity_curve(horizon, date);
CREATE INDEX IF NOT EXISTS idx_outcomes_unresolved ON outcomes(horizon, direction);
