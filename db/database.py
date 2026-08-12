"""Ligacao SQLite (WAL, foreign keys) + inicializacao/seed idempotentes."""

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection(db_path=None) -> sqlite3.Connection:
    """db_path=None usa a BD real do projeto (config.DB_PATH). Testes passam
    ':memory:' ou um ficheiro temporario para isolamento completo."""
    if db_path is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        db_path = str(config.DB_PATH)
    elif db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    if db_path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()


def seed_watchlist(conn: sqlite3.Connection) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for item in config.WATCHLIST:
        conn.execute(
            """INSERT INTO watchlist (ticker, name, sector, active, added_date)
               VALUES (?, ?, ?, 1, ?)
               ON CONFLICT(ticker) DO UPDATE SET name = excluded.name, sector = excluded.sector""",
            (item["ticker"], item["name"], item["sector"], now),
        )
    conn.commit()


def seed_portfolio_state(conn: sqlite3.Connection) -> None:
    """Cria a linha de carteira por horizonte se ainda nao existir.
    NUNCA reescreve cash de uma carteira ja existente (perderia o historico
    simulado) -- por isso ON CONFLICT DO NOTHING, nao DO UPDATE."""
    now = datetime.now(timezone.utc).isoformat()
    for horizon in config.HORIZONS:
        conn.execute(
            """INSERT INTO portfolio_state (horizon, cash, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(horizon) DO NOTHING""",
            (horizon, config.STARTING_CASH, now),
        )
    conn.commit()


def log_run(conn: sqlite3.Connection, date: str, stage: str, status: str, message: str = "") -> None:
    if status not in ("OK", "WARN", "ERROR"):
        raise ValueError(f"invalid run_log status: {status}")
    conn.execute(
        "INSERT INTO run_log (date, stage, status, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (date, stage, status, message, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


if __name__ == "__main__":
    connection = get_connection()
    seed_watchlist(connection)
    seed_portfolio_state(connection)
    tables = [r[0] for r in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    print("DB initialized at", config.DB_PATH)
    print("Tables:", ", ".join(tables))
    connection.close()
