import sqlite3
import time
from contextlib import closing, nullcontext
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from equity_research.config import (
    DEFAULT_CURRENCY,
    FUNDAMENTALS_SLEEP_SECONDS,
    REPORT_LAG_ANNUAL_DAYS,
    REPORT_LAG_QUARTERLY_DAYS,
)
from equity_research.db import get_connection

# yfinance row label → our column name. First match wins where labels alias.
_INCOME_MAP: dict[str, str] = {
    "Total Revenue":          "revenue",
    "Gross Profit":           "gross_profit",
    "Operating Income":       "operating_income",
    "Net Income":             "net_income",
    "Diluted EPS":            "eps_diluted",
    "Diluted Average Shares": "shares_diluted",
}
_BALANCE_MAP: dict[str, str] = {
    "Total Assets":                                     "total_assets",
    "Stockholders Equity":                              "total_equity",
    "Total Equity Gross Minority Interest":             "total_equity",
    "Total Debt":                                       "total_debt",
    "Cash And Cash Equivalents":                        "cash",
    "Cash Cash Equivalents And Short Term Investments": "cash",
}
_CASHFLOW_MAP: dict[str, str] = {
    "Operating Cash Flow": "operating_cash_flow",
    "Capital Expenditure": "capex",
    "Free Cash Flow":      "free_cash_flow",
}

_OUR_COLS = [
    "revenue", "gross_profit", "operating_income", "net_income",
    "eps_diluted", "shares_diluted", "total_assets", "total_equity",
    "total_debt", "cash", "operating_cash_flow", "capex", "free_cash_flow",
]


def _closest_col(
    df: pd.DataFrame | None, target: pd.Timestamp, max_days: int = 45
) -> pd.Timestamp | None:
    """Return the column in df nearest to target within max_days, or None.

    Guards against non-DatetimeIndex columns by attempting pd.to_datetime
    coercion. Uses per-element total_seconds() rather than TimedeltaIndex.abs(),
    which is absent in some pandas 2.x builds. Returns None on any arithmetic
    failure (e.g. tz-awareness mismatch between inc and bal/cf columns).
    """
    if df is None or df.empty:
        return None
    cols = df.columns
    if not isinstance(cols, pd.DatetimeIndex):
        try:
            cols = pd.to_datetime(cols)
        except Exception:
            return None
    try:
        diffs_sec = [abs((c - target).total_seconds()) for c in cols]
    except Exception:
        return None
    min_sec = min(diffs_sec)
    return cols[diffs_sec.index(min_sec)] if min_sec / 86400 <= max_days else None


def _extract_period(
    stmt_cols: list[tuple[pd.DataFrame | None, pd.Timestamp | None]],
    maps: list[dict[str, str]],
) -> dict[str, float | None]:
    """
    Extract all mapped fields from (statement, column) pairs.
    First match wins — guards against duplicate labels across maps.
    """
    result: dict[str, float | None] = {}
    for (stmt, col), field_map in zip(stmt_cols, maps):
        if stmt is None or col is None or stmt.empty or col not in stmt.columns:
            continue
        series = stmt[col]
        for yf_label, our_col in field_map.items():
            if yf_label in series.index and our_col not in result:
                val = series[yf_label]
                result[our_col] = float(val) if pd.notna(val) else None
    return result


def _upsert_period(
    ticker: str,
    period_end: date,
    lag_days: int,
    fields: dict[str, float | None],
    table: str,
    conn: sqlite3.Connection,
) -> None:
    assert table in {"financials_annual", "financials_quarterly"}, (
        f"unexpected table name: {table!r}"
    )
    report_date = (period_end + timedelta(days=lag_days)).isoformat()
    cols = ["ticker", "fiscal_period_end", "report_date", "currency"] + _OUR_COLS
    vals = [ticker, period_end.isoformat(), report_date, DEFAULT_CURRENCY] + [
        fields.get(c) for c in _OUR_COLS
    ]
    placeholders = ", ".join(["?"] * len(cols))
    # ingested_at uses CURRENT_TIMESTAMP on INSERT (schema default) and is
    # explicitly refreshed on UPDATE so it always reflects the last ingest time.
    update_set = ", ".join(
        f"{c} = excluded.{c}" for c in cols if c not in ("ticker", "fiscal_period_end")
    ) + ", ingested_at = CURRENT_TIMESTAMP"
    conn.execute(
        f"""
        INSERT INTO {table} ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(ticker, fiscal_period_end) DO UPDATE SET {update_set}
        """,
        vals,
    )


def _fetch_statements(ticker: str) -> dict[str, pd.DataFrame | None]:
    """Fetch all six statement DataFrames. Falls back to legacy attribute names
    for income statement only — balance_sheet and cashflow have no legacy alias."""
    t = yf.Ticker(ticker)

    def _get(attr: str, fallback: str | None = None) -> pd.DataFrame | None:
        df = getattr(t, attr, None)
        if (df is None or (isinstance(df, pd.DataFrame) and df.empty)) and fallback is not None:
            df = getattr(t, fallback, None)
        return df if isinstance(df, pd.DataFrame) and not df.empty else None

    return {
        "inc_a": _get("income_stmt",            "financials"),
        "inc_q": _get("quarterly_income_stmt",  "quarterly_financials"),
        "bal_a": _get("balance_sheet"),
        "bal_q": _get("quarterly_balance_sheet"),
        "cf_a":  _get("cashflow"),
        "cf_q":  _get("quarterly_cashflow"),
    }


def _process_periods(
    inc: pd.DataFrame | None,
    bal: pd.DataFrame | None,
    cf: pd.DataFrame | None,
    ticker: str,
    lag_days: int,
    table: str,
    conn: sqlite3.Connection,
) -> int:
    """Upsert all periods for one grain (annual or quarterly). Returns row count.

    Periods where every extracted field is None are silently skipped — they
    indicate a column with no recognisable labels and would produce junk rows.
    """
    if inc is None:
        return 0
    count = 0
    for col in inc.columns:
        fields = _extract_period(
            [(inc, col),
             (bal, _closest_col(bal, col)),
             (cf,  _closest_col(cf,  col))],
            [_INCOME_MAP, _BALANCE_MAP, _CASHFLOW_MAP],
        )
        if not any(v is not None for v in fields.values()):
            continue  # no recognisable labels matched; skip rather than persist nulls
        _upsert_period(ticker, col.date(), lag_days, fields, table, conn)
        count += 1
    return count


def _ingest_one(ticker: str, conn: sqlite3.Connection) -> dict:
    """
    Fetch and upsert annual + quarterly fundamentals for one ticker.

    Returns a dict with four keys:
      annual_rows / quarterly_rows : int on success, None on failure
      annual_error / quarterly_error: str on failure, None on success

    Statement fetch failure propagates to both grains — there is nothing
    to process if we can't reach yfinance.
    """
    try:
        stmts = _fetch_statements(ticker)
    except Exception as exc:
        err = str(exc)
        return {
            "annual_rows": None, "quarterly_rows": None,
            "annual_error": err, "quarterly_error": err,
        }

    result = {
        "annual_rows": None, "quarterly_rows": None,
        "annual_error": None, "quarterly_error": None,
    }

    try:
        n = _process_periods(
            stmts["inc_a"], stmts["bal_a"], stmts["cf_a"],
            ticker, REPORT_LAG_ANNUAL_DAYS, "financials_annual", conn,
        )
        if n == 0:
            raise ValueError("no annual statement data returned by yfinance")
        result["annual_rows"] = n
    except Exception as exc:
        result["annual_error"] = str(exc)

    try:
        n = _process_periods(
            stmts["inc_q"], stmts["bal_q"], stmts["cf_q"],
            ticker, REPORT_LAG_QUARTERLY_DAYS, "financials_quarterly", conn,
        )
        if n == 0:
            raise ValueError("no quarterly statement data returned by yfinance")
        result["quarterly_rows"] = n
    except Exception as exc:
        result["quarterly_error"] = str(exc)

    return result


def _log(
    conn: sqlite3.Connection,
    ticker: str,
    data_type: str,
    status: str,
    rows_upserted: int | None = None,
    error_msg: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO ingestion_log (ticker, data_type, status, rows_upserted, error_msg)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ticker, data_type, status, rows_upserted, error_msg),
    )


def ingest_fundamentals(
    tickers: list[str] | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict:
    """Pull and persist annual + quarterly fundamentals for active universe tickers."""
    cm = closing(get_connection()) if conn is None else nullcontext(conn)
    with cm as active_conn:
        if tickers is None:
            rows = active_conn.execute(
                "SELECT ticker FROM universe WHERE is_active = 1"
            ).fetchall()
            tickers = [r[0] for r in rows]

        stats = {
            "annual_success": 0, "annual_failed": 0, "annual_rows": 0,
            "quarterly_success": 0, "quarterly_failed": 0, "quarterly_rows": 0,
            "failed_tickers": [],   # tickers where BOTH annual and quarterly failed
        }

        try:
            for i, ticker in enumerate(tickers):
                result = _ingest_one(ticker, active_conn)

                if result["annual_rows"] is not None:
                    stats["annual_success"] += 1
                    stats["annual_rows"] += result["annual_rows"]
                    _log(active_conn, ticker, "fundamentals_annual", "success",
                         rows_upserted=result["annual_rows"])
                else:
                    stats["annual_failed"] += 1
                    _log(active_conn, ticker, "fundamentals_annual", "failed",
                         error_msg=result["annual_error"])

                if result["quarterly_rows"] is not None:
                    stats["quarterly_success"] += 1
                    stats["quarterly_rows"] += result["quarterly_rows"]
                    _log(active_conn, ticker, "fundamentals_quarterly", "success",
                         rows_upserted=result["quarterly_rows"])
                else:
                    stats["quarterly_failed"] += 1
                    _log(active_conn, ticker, "fundamentals_quarterly", "failed",
                         error_msg=result["quarterly_error"])

                if result["annual_rows"] is None and result["quarterly_rows"] is None:
                    stats["failed_tickers"].append(ticker)

                active_conn.commit()

                if i < len(tickers) - 1:
                    time.sleep(FUNDAMENTALS_SLEEP_SECONDS)
        finally:
            active_conn.commit()

        return stats
