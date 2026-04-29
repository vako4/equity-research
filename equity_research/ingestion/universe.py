import sqlite3
from contextlib import closing, nullcontext
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from equity_research.config import SP500_WIKI_URL, UNIVERSE_SNAPSHOT_DIR
from equity_research.db import get_connection

_REQUIRED_COLS = {"Symbol", "Security", "GICS Sector"}
_EXPECTED_MIN_ROWS = 400


def _find_sp500_table() -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "equity-research/0.1 (https://github.com/vako4/equity-research; "
            "valerian.meipariani@iset.ge) python-requests"
        )
    }
    resp = requests.get(SP500_WIKI_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    for tbl in tables:
        if _REQUIRED_COLS.issubset(tbl.columns) and len(tbl) >= _EXPECTED_MIN_ROWS:
            df = tbl.rename(columns={
                "Symbol":            "ticker",
                "Security":          "company_name",
                "GICS Sector":       "sector",
                "GICS Sub-Industry": "industry",
            })
            # Wikipedia uses dots (BRK.B); yfinance expects dashes (BRK-B).
            df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
            return df[["ticker", "company_name", "sector", "industry"]]
    raise RuntimeError(
        f"S&P 500 constituent table not found on Wikipedia. "
        f"Expected columns {_REQUIRED_COLS} and >= {_EXPECTED_MIN_ROWS} rows. "
        "Page structure may have changed."
    )


def save_snapshot(df: pd.DataFrame, snapshot_dir: Path = UNIVERSE_SNAPSHOT_DIR) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"sp500_{date.today().isoformat()}.csv"
    df.to_csv(path, index=False)
    return path


def upsert_universe(df: pd.DataFrame, conn: sqlite3.Connection) -> int:
    today = date.today().isoformat()
    rows = [
        (row.ticker, row.company_name, row.sector, row.industry, today)
        for row in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO universe (ticker, company_name, sector, industry, added_date, is_active)
        VALUES (?, ?, ?, ?, ?, 1)
        ON CONFLICT(ticker) DO UPDATE SET
            company_name = excluded.company_name,
            sector       = excluded.sector,
            industry     = excluded.industry,
            is_active    = 1,
            removed_date = NULL
        """,
        rows,
    )
    return len(rows)


def _mark_removals(current_tickers: list[str], conn: sqlite3.Connection) -> int:
    """Set is_active=0 on tickers that were active but absent from today's fetch."""
    placeholders = ",".join("?" * len(current_tickers))
    today = date.today().isoformat()
    cursor = conn.execute(
        f"UPDATE universe SET is_active = 0, removed_date = ? "
        f"WHERE is_active = 1 AND ticker NOT IN ({placeholders})",
        [today, *current_tickers],
    )
    return cursor.rowcount


def _write_log(
    conn: sqlite3.Connection,
    status: str,
    rows_upserted: int | None = None,
    error_msg: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO ingestion_log (ticker, data_type, status, rows_upserted, error_msg)
        VALUES (NULL, 'universe', ?, ?, ?)
        """,
        (status, rows_upserted, error_msg),
    )
    conn.commit()


def _log_failure(conn: sqlite3.Connection, error_msg: str) -> None:
    """Write a failure row to ingestion_log. Falls back to a fresh connection if needed."""
    try:
        _write_log(conn, "failed", error_msg=error_msg)
    except Exception:
        # conn may be in a broken state after rollback; open a fresh one just for this write.
        try:
            with closing(get_connection()) as fallback:
                _write_log(fallback, "failed", error_msg=error_msg)
        except Exception:
            pass  # never let a logging failure mask the original exception


def refresh_universe(
    conn: sqlite3.Connection | None = None,
    snapshot_dir: Path = UNIVERSE_SNAPSHOT_DIR,
) -> dict:
    """Scrape Wikipedia, save CSV snapshot, upsert to DB, mark removals, log result."""
    cm = closing(get_connection()) if conn is None else nullcontext(conn)
    with cm as active_conn:
        try:
            df = _find_sp500_table()
            snapshot_path = save_snapshot(df, snapshot_dir)
            rows_upserted = upsert_universe(df, active_conn)
            removed = _mark_removals(df["ticker"].tolist(), active_conn)
            active_conn.commit()
            _write_log(active_conn, "success", rows_upserted=rows_upserted)
            return {
                "snapshot_path": str(snapshot_path),
                "rows_upserted": rows_upserted,
                "removed_count": removed,
                "tickers": df["ticker"].tolist(),
            }
        except Exception as exc:
            try:
                active_conn.rollback()
            except Exception:
                pass
            _log_failure(active_conn, str(exc))
            raise
