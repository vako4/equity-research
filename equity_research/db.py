import sqlite3
from contextlib import closing
from pathlib import Path

from equity_research.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS universe (
    ticker       TEXT PRIMARY KEY,
    company_name TEXT,
    sector       TEXT,
    industry     TEXT,
    added_date   DATE NOT NULL,
    removed_date DATE,
    is_active    INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS prices_daily (
    ticker    TEXT NOT NULL,
    date      DATE NOT NULL,
    open      REAL,
    high      REAL,
    low       REAL,
    adj_close REAL,
    volume    INTEGER,
    currency  TEXT NOT NULL DEFAULT 'USD',
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker) REFERENCES universe(ticker)
);

CREATE TABLE IF NOT EXISTS financials_annual (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT NOT NULL,
    fiscal_period_end   DATE NOT NULL,
    report_date         DATE,
    currency            TEXT NOT NULL DEFAULT 'USD',
    revenue             REAL,
    gross_profit        REAL,
    operating_income    REAL,
    net_income          REAL,
    eps_diluted         REAL,
    shares_diluted      REAL,
    total_assets        REAL,
    total_equity        REAL,
    total_debt          REAL,
    cash                REAL,
    operating_cash_flow REAL,
    capex               REAL,
    free_cash_flow      REAL,
    ingested_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, fiscal_period_end),
    FOREIGN KEY (ticker) REFERENCES universe(ticker)
);

CREATE TABLE IF NOT EXISTS financials_quarterly (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT NOT NULL,
    fiscal_period_end   DATE NOT NULL,
    report_date         DATE,
    currency            TEXT NOT NULL DEFAULT 'USD',
    revenue             REAL,
    gross_profit        REAL,
    operating_income    REAL,
    net_income          REAL,
    eps_diluted         REAL,
    shares_diluted      REAL,
    total_assets        REAL,
    total_equity        REAL,
    total_debt          REAL,
    cash                REAL,
    operating_cash_flow REAL,
    capex               REAL,
    free_cash_flow      REAL,
    ingested_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, fiscal_period_end),
    FOREIGN KEY (ticker) REFERENCES universe(ticker)
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT,
    data_type     TEXT,
    status        TEXT,
    rows_upserted INTEGER,
    error_msg     TEXT,
    ingested_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cross-sectional factor queries filter on date/period columns; these indexes
-- prevent full table scans on the two largest tables.
CREATE INDEX IF NOT EXISTS idx_prices_date
    ON prices_daily(date);

CREATE INDEX IF NOT EXISTS idx_financials_annual_period
    ON financials_annual(fiscal_period_end);

CREATE INDEX IF NOT EXISTS idx_financials_quarterly_period
    ON financials_quarterly(fiscal_period_end);
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    # WAL allows concurrent readers alongside one writer. It persists in the DB
    # file after the first set, but re-applying it per-connection is harmless
    # and guards against older SQLite builds where mode inheritance isn't guaranteed.
    conn.execute("PRAGMA journal_mode=WAL")
    # Foreign key enforcement is OFF by default in SQLite and does NOT persist in
    # the file — it must be enabled on every connection. Without this, FK
    # constraints (e.g. prices_daily.ticker → universe.ticker) are silently ignored.
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(get_connection(db_path)) as conn:
        conn.executescript(_SCHEMA)


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {DB_PATH}")
