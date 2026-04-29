import sqlite3
import time
from contextlib import closing, nullcontext
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from equity_research.config import (
    DEFAULT_CURRENCY,
    PRICE_BATCH_SIZE,
    PRICE_BATCH_SLEEP_SECONDS,
    PRICES_YEARS_BACK,
)
from equity_research.db import get_connection


def _active_tickers(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT ticker FROM universe WHERE is_active = 1").fetchall()
    return [r[0] for r in rows]


def _date_range(years_back: int) -> tuple[str, str]:
    end = date.today()
    # 365.25 corrects for leap years: 12 * 365.25 = 4383 days covers exactly 12 calendar years.
    start = end - timedelta(days=int(years_back * 365.25))
    return start.isoformat(), end.isoformat()


def _download_batch(
    tickers: list[str], start: str, end: str
) -> tuple[dict[str, pd.DataFrame], str | None]:
    """
    Download one batch with one retry on exception or empty result.
    Returns ({ticker: df}, error_msg). error_msg is None on any non-empty result;
    set to the actual failure reason + "after 2 attempts" when both attempts fail.
    """
    last_error: str | None = None

    for attempt in range(2):
        if attempt > 0:
            # Empty result or exception on first attempt — likely rate-limited.
            time.sleep(PRICE_BATCH_SLEEP_SECONDS * 5)

        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            last_error = str(exc)
            continue

        if raw.empty:
            last_error = "yfinance returned empty DataFrame (possible rate limit)"
            continue

        result: dict[str, pd.DataFrame] = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    df = raw.xs(ticker, level=1, axis=1).dropna(how="all")
                    if not df.empty:
                        result[ticker] = df
                except KeyError:
                    pass
        else:
            # Single-ticker download returns flat columns.
            if len(tickers) == 1:
                df = raw.dropna(how="all")
                if not df.empty:
                    result[tickers[0]] = df

        return result, None

    return {}, f"{last_error} (after 2 attempts)"


def _float(val) -> float | None:
    return float(val) if pd.notna(val) else None


def _int(val) -> int | None:
    return int(val) if pd.notna(val) else None


def _upsert_prices(ticker: str, df: pd.DataFrame, conn: sqlite3.Connection) -> int:
    # Columns selected explicitly to guard against yfinance reordering.
    # itertuples(name=None) returns plain tuples (index, Open, High, Low, Close, Volume)
    # — ~10x faster than iterrows(), which allocates a full Series per row.
    rows = [
        (
            ticker,
            idx.date().isoformat(),
            _float(o), _float(h), _float(l), _float(c),
            _int(v),
            DEFAULT_CURRENCY,
        )
        for idx, o, h, l, c, v
        in df[["Open", "High", "Low", "Close", "Volume"]].itertuples(name=None)
    ]
    conn.executemany(
        """
        INSERT INTO prices_daily (ticker, date, open, high, low, adj_close, volume, currency)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            open      = excluded.open,
            high      = excluded.high,
            low       = excluded.low,
            adj_close = excluded.adj_close,
            volume    = excluded.volume
        """,
        rows,
    )
    return len(rows)


def _log(
    conn: sqlite3.Connection,
    ticker: str,
    status: str,
    rows_upserted: int | None = None,
    error_msg: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO ingestion_log (ticker, data_type, status, rows_upserted, error_msg)
        VALUES (?, 'prices', ?, ?, ?)
        """,
        (ticker, status, rows_upserted, error_msg),
    )


def ingest_prices(
    tickers: list[str] | None = None,
    years_back: int = PRICES_YEARS_BACK,
    conn: sqlite3.Connection | None = None,
) -> dict:
    """Pull and persist adjusted OHLCV for active universe tickers.

    # TODO: per-ticker currency lookup needed for international universe expansion.
    """
    cm = closing(get_connection()) if conn is None else nullcontext(conn)
    with cm as active_conn:
        if tickers is None:
            tickers = _active_tickers(active_conn)

        start, end = _date_range(years_back)
        stats = {"success": 0, "failed": 0, "total_rows": 0, "failed_tickers": []}

        try:
            for batch_start in range(0, len(tickers), PRICE_BATCH_SIZE):
                batch = tickers[batch_start : batch_start + PRICE_BATCH_SIZE]
                downloaded, batch_error = _download_batch(batch, start, end)

                for ticker in batch:
                    if ticker not in downloaded:
                        err = batch_error or "not returned by yfinance"
                        stats["failed"] += 1
                        stats["failed_tickers"].append(ticker)
                        _log(active_conn, ticker, "failed", error_msg=err)
                    else:
                        try:
                            n = _upsert_prices(ticker, downloaded[ticker], active_conn)
                            stats["success"] += 1
                            stats["total_rows"] += n
                            _log(active_conn, ticker, "success", rows_upserted=n)
                        except Exception as exc:
                            stats["failed"] += 1
                            stats["failed_tickers"].append(ticker)
                            _log(active_conn, ticker, "failed", error_msg=str(exc))

                active_conn.commit()

                if batch_start + PRICE_BATCH_SIZE < len(tickers):
                    time.sleep(PRICE_BATCH_SLEEP_SECONDS)
        finally:
            # Commits any buffered log writes from the current (partial) batch
            # before the exception propagates. No-op on clean exit.
            active_conn.commit()

        return stats
