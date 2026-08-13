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


-- ===========================================================================
-- Subsistema de padroes graficos (independente do de previsao de direcao)
-- ===========================================================================

-- Barras intradiarias. ts em ISO8601 UTC. timeframe: '1h', '5m', ...
CREATE TABLE IF NOT EXISTS intraday_prices (
    ticker       TEXT NOT NULL REFERENCES watchlist(ticker),
    timeframe    TEXT NOT NULL,
    ts           TEXT NOT NULL,
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL NOT NULL,
    volume       INTEGER,
    ingested_at  TEXT NOT NULL,
    PRIMARY KEY (ticker, timeframe, ts)
);

-- Um padrao detectado no historico. confirmed_ts e' quando o padrao passou a
-- ser CONHECIVEL (o ultimo pivot so fica confirmado pivot_window barras
-- depois de acontecer) -- e' este o campo a usar para qualquer avaliacao
-- honesta, nunca end_ts.
CREATE TABLE IF NOT EXISTS detected_patterns (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT NOT NULL REFERENCES watchlist(ticker),
    timeframe      TEXT NOT NULL,
    pattern_type   TEXT NOT NULL,
    start_ts       TEXT NOT NULL,
    end_ts         TEXT NOT NULL,
    confirmed_ts   TEXT NOT NULL,
    start_idx      INTEGER NOT NULL,
    end_idx        INTEGER NOT NULL,
    quality        REAL NOT NULL,
    meta_json      TEXT,
    detected_at    TEXT NOT NULL,
    UNIQUE (ticker, timeframe, pattern_type, start_ts, end_ts)
);

-- Matriz de transicao aprendida: quantas vezes from_pattern foi seguido de
-- to_pattern. `count` e' o SUPORTE -- uma probabilidade de 60% assente em 2
-- observacoes nao vale nada e a UI tem de mostrar isso.
CREATE TABLE IF NOT EXISTS pattern_transitions (
    timeframe       TEXT NOT NULL,
    from_pattern    TEXT NOT NULL,
    to_pattern      TEXT NOT NULL,
    count           INTEGER NOT NULL,
    median_bars     REAL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (timeframe, from_pattern, to_pattern)
);

-- Snapshot de uma previsao encadeada de N passos feita num dado momento.
-- step=1..N; step_confidence e' P(este padrao | padrao anterior da cadeia),
-- cumulative_confidence e' o produto ao longo da cadeia toda ate' aqui.
CREATE TABLE IF NOT EXISTS pattern_forecasts (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                  TEXT NOT NULL REFERENCES watchlist(ticker),
    timeframe               TEXT NOT NULL,
    as_of_ts                TEXT NOT NULL,
    from_pattern            TEXT NOT NULL,
    step                    INTEGER NOT NULL,
    pattern_type            TEXT NOT NULL,
    step_confidence         REAL NOT NULL,
    cumulative_confidence   REAL NOT NULL,
    expected_bars           REAL,
    support                 INTEGER NOT NULL,
    created_at              TEXT NOT NULL,
    UNIQUE (ticker, timeframe, as_of_ts, step)
);

-- Classificador multiclasse contextual (passo 1 da cadeia de previsao).
-- Tabela separada de model_versions porque essa tem CHECK(horizon IN
-- ('SHORT','LONG')), que pertence ao outro subsistema.
-- Todas as accuracy aqui sao medidas no conjunto de TESTE, que nunca e' usado
-- para escolher nada (nem algoritmo nem peso do ensemble -- isso decide-se na
-- particao de validacao). Sem esta separacao os numeros ficam optimistas: com
-- split a dois, o ensemble a 5m aparentava +2.9pp que se revelaram +0.3pp.
CREATE TABLE IF NOT EXISTS pattern_model_versions (
    version                      TEXT PRIMARY KEY,
    timeframe                    TEXT NOT NULL,
    algorithm                    TEXT NOT NULL,
    trained_at                   TEXT NOT NULL,
    n_train                      INTEGER NOT NULL,
    n_validation                 INTEGER NOT NULL,
    n_test                       INTEGER NOT NULL,
    n_features                   INTEGER NOT NULL,
    accuracy                     REAL,
    top3_accuracy                REAL,
    markov_accuracy              REAL,
    markov_top3_accuracy         REAL,
    baseline_frequency_accuracy  REAL,
    ensemble_weight              REAL,
    ensemble_accuracy            REAL,
    standard_error               REAL,
    hyperparams_json             TEXT,
    feature_names_json           TEXT,
    notes                        TEXT
);

-- Rotulo de DIRECCAO depois de um padrao.
--
-- ref_close e' o fecho em CONFIRMED_TS, nao no fim do padrao. E' a distincao
-- que impede lookahead: o padrao acaba em end_ts, mas so' se SABE que existe
-- pivot_window barras depois, quando o ultimo pivot fica confirmado. Usar o
-- preco do fim do padrao seria negociar com informacao que ainda nao existia.
CREATE TABLE IF NOT EXISTS pattern_outcomes (
    ticker            TEXT NOT NULL REFERENCES watchlist(ticker),
    timeframe         TEXT NOT NULL,
    confirmed_ts      TEXT NOT NULL,
    horizon_bars      INTEGER NOT NULL,
    pattern_type      TEXT NOT NULL,
    ref_close         REAL NOT NULL,
    target_close      REAL NOT NULL,
    forward_return    REAL NOT NULL,
    benchmark_return  REAL,
    excess_return     REAL,
    direction         INTEGER NOT NULL CHECK (direction IN (0, 1)),
    excess_direction  INTEGER CHECK (excess_direction IN (0, 1)),
    computed_at       TEXT NOT NULL,
    PRIMARY KEY (ticker, timeframe, confirmed_ts, horizon_bars)
);

CREATE TABLE IF NOT EXISTS pattern_direction_models (
    version                  TEXT PRIMARY KEY,
    timeframe                TEXT NOT NULL,
    horizon_bars             INTEGER NOT NULL,
    market_neutral           INTEGER NOT NULL,
    algorithm                TEXT NOT NULL,
    trained_at               TEXT NOT NULL,
    n_train                  INTEGER NOT NULL,
    n_validation             INTEGER NOT NULL,
    n_test                   INTEGER NOT NULL,
    n_effective_days         INTEGER NOT NULL,
    accuracy                 REAL,
    auc                      REAL,
    log_loss                 REAL,
    baseline_majority        REAL,
    random_control_accuracy  REAL,
    mean_return_when_up      REAL,
    mean_return_when_down    REAL,
    standard_error           REAL,
    significant              INTEGER,
    selected                 INTEGER NOT NULL DEFAULT 0,
    notes                    TEXT
);

CREATE INDEX IF NOT EXISTS idx_intraday_ticker_tf ON intraday_prices(ticker, timeframe, ts);
CREATE INDEX IF NOT EXISTS idx_pattern_outcomes ON pattern_outcomes(timeframe, horizon_bars, confirmed_ts);
CREATE INDEX IF NOT EXISTS idx_patterns_ticker_tf ON detected_patterns(ticker, timeframe, confirmed_ts);
CREATE INDEX IF NOT EXISTS idx_forecasts_lookup ON pattern_forecasts(ticker, timeframe, as_of_ts);
