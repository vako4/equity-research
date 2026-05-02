"""
Ticker-agnostic ratio computation for tearsheets and cross-sectional factor work.

All functions read from the local DB and return DataFrames — no rendering logic here.
Price data: prices_daily.adj_close (split- and dividend-adjusted as of last ingest).
Fundamental data: financials_quarterly / financials_annual joined on report_date.

Column definitions (applies wherever these fields are used):
  total_debt   -- yfinance "Total Debt": short-term + long-term obligations combined.
                  Excludes operating leases unless yfinance folds them in (varies).
  cash         -- "Cash And Cash Equivalents" if available, else
                  "Cash Cash Equivalents And Short Term Investments".
                  Composition therefore varies by ticker and period.
  total_equity -- "Stockholders Equity" or "Total Equity Gross Minority Interest"
                  (first match). May include minority interest.
  shares_diluted -- diluted weighted-average shares from the income statement.
"""

import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd

from equity_research.db import get_connection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_prices(conn: sqlite3.Connection, ticker: str) -> pd.Series:
    """adj_close as a float Series indexed by DatetimeIndex, sorted ascending."""
    rows = conn.execute(
        "SELECT date, adj_close FROM prices_daily WHERE ticker = ? ORDER BY date",
        (ticker,),
    ).fetchall()
    if not rows:
        return pd.Series(dtype=float, name=ticker)
    return pd.Series(
        [r[1] for r in rows],
        index=pd.to_datetime([r[0] for r in rows]),
        name=ticker,
        dtype=float,
    )


def _annualized(p_end: float, p_start: float, days: int) -> float:
    """Geometric annualized return. NaN-safe: returns NaN when either price is
    NaN (NaN > 0 is False), non-positive, or days is zero."""
    if not (p_start > 0 and p_end > 0 and days > 0):
        return float("nan")
    return (p_end / p_start) ** (365.25 / days) - 1


def _horizon_start(series: pd.Series, target: pd.Timestamp) -> pd.Timestamp | None:
    """Last available trading date at or before target. None if no data exists."""
    candidates = series.index[series.index <= target]
    return candidates[-1] if len(candidates) > 0 else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def returns_table(ticker: str, benchmark: str = "SPY") -> pd.DataFrame:
    """
    Annualized geometric returns for standard horizons.

    Source: prices_daily.adj_close for both ticker and benchmark.
    Annualization: (P_end / P_start) ** (365.25 / calendar_days) - 1.

    Horizon endpoint handling:
      - Each period start snaps to the last available trading day at or
        before T - N years (calendar). Both ticker and benchmark are
        annualized over the identical window (snapped to ticker's start
        date) so that excess = same-period outperformance.
      - Inception uses the ticker's first available price date as start.
        If days_since_inception < 365, the Inception row is NaN for all
        columns — annualizing a sub-year return is misleading.
      - A row is NaN whenever either series lacks data for that horizon.

    Index:   ["1Y", "3Y", "5Y", "10Y", "Inception"] (string, fixed order)
    Columns: ticker_ann (float), bench_ann (float), excess (float)
             All values are decimal fractions (0.12 = 12%).
    """
    with closing(get_connection()) as conn:
        prices_t = _load_prices(conn, ticker)
        prices_b = _load_prices(conn, benchmark)

    if prices_t.empty:
        raise ValueError(f"No price data for {ticker!r}. Run ingest_prices first.")
    if prices_b.empty:
        raise ValueError(
            f"No price data for benchmark {benchmark!r}. Run ingest_benchmarks first."
        )

    end_date = min(prices_t.index[-1], prices_b.index[-1])
    p_end_t = float(prices_t.asof(end_date))
    p_end_b = float(prices_b.asof(end_date))

    rows: list[dict] = []

    for label, years in [("1Y", 1), ("3Y", 3), ("5Y", 5), ("10Y", 10)]:
        target = end_date - pd.DateOffset(years=years)
        snap_t = _horizon_start(prices_t, target)

        if snap_t is None:
            rows.append({"ticker_ann": np.nan, "bench_ann": np.nan, "excess": np.nan})
            continue

        days = (end_date - snap_t).days
        t_ann = _annualized(p_end_t, float(prices_t[snap_t]), days)
        b_ann = _annualized(p_end_b, float(prices_b.asof(snap_t)), days)
        excess = t_ann - b_ann if np.isfinite(t_ann) and np.isfinite(b_ann) else np.nan
        rows.append({"ticker_ann": t_ann, "bench_ann": b_ann, "excess": excess})

    # Inception row
    inception = prices_t.index[0]
    days_since = (end_date - inception).days

    if days_since < 365:
        rows.append({"ticker_ann": np.nan, "bench_ann": np.nan, "excess": np.nan})
    else:
        t_ann = _annualized(p_end_t, float(prices_t[inception]), days_since)
        b_ann = _annualized(p_end_b, float(prices_b.asof(inception)), days_since)
        excess = t_ann - b_ann if np.isfinite(t_ann) and np.isfinite(b_ann) else np.nan
        rows.append({"ticker_ann": t_ann, "bench_ann": b_ann, "excess": excess})

    return pd.DataFrame(
        rows,
        index=pd.Index(["1Y", "3Y", "5Y", "10Y", "Inception"], name="period"),
    )


def drawdown(ticker: str) -> pd.DataFrame:
    """
    Rolling drawdown from peak using prices_daily.adj_close.

    Formula: (adj_close - cumulative_max) / cumulative_max
    Values are ≤ 0 (e.g. -0.31 means -31% below the prior peak).
    NaN until the first valid adj_close price.

    Index:   DatetimeIndex, daily
    Columns: drawdown (float)
    """
    with closing(get_connection()) as conn:
        prices = _load_prices(conn, ticker)
    if prices.empty:
        raise ValueError(f"No price data for {ticker!r}. Run ingest_prices first.")
    cum_max = prices.cummax()
    dd = (prices - cum_max) / cum_max
    return pd.DataFrame({"drawdown": dd})


def valuation_ratios(ticker: str) -> pd.DataFrame:
    """
    Daily valuation ratios joined from adj_close × carried-forward quarterly fundamentals.

    Computation:
      pe_ttm  = (adj_close × shares_diluted) / net_income_ttm
      pb      = (adj_close × shares_diluted) / total_equity
      ev_ebit = (adj_close × shares_diluted + total_debt - cash) / operating_income_ttm

    TTM (trailing twelve months): rolling 4-quarter sum of net_income and
    operating_income computed at each report_date BEFORE the merge_asof join.
    Requires exactly 4 non-NaN quarters; NaN when fewer quarters are available.

    Join: pd.merge_asof(direction='backward') — each trading day carries forward
    the most recent quarterly report as of that date. Ratios are recomputed daily
    from the current adj_close; the fundamental inputs are carried forward as-is.

    Staleness: ratios use the most recently REPORTED TTM, not the most recently
    COMPLETED TTM. report_date ≈ fiscal_period_end + 45 days, so the most recent
    ~90 days of ratios reflect TTM ending ~one quarter ago, not yesterday. This
    is correct point-in-time behavior for backtest realism.

    EBIT proxy: operating_income is used as EBIT — slight deviation from textbook
    EBIT (excludes interest income on cash, immaterial for most names but
    non-trivial for cash-rich firms like AAPL).

    NaN conditions:
      - Before the first report_date with >= 4 quarters of history.
      - When TTM denominator (net_income_ttm or operating_income_ttm) is <= 0.
      - When total_equity <= 0: P/B is NaN when book equity is non-positive. This
        is a deliberate choice — negative book equity from buybacks is valid (HD,
        MCD, SBUX), but the ratio loses economic meaning. Phase 3 cross-sectional
        work will need to handle this — either treating negatives as NaN (current),
        flipping signs, or capping. For now: NaN, document, defer.
      - When shares_diluted is NULL (market cap cannot be computed).

    Note: EV/EBITDA is not computed — D&A is not in the schema. operating_income
    is used as the EBIT proxy (Revenue - COGS - operating expenses including D&A).

    Column definitions:
      total_debt     -- yfinance "Total Debt": short-term + long-term obligations.
                        Excludes operating leases unless yfinance folds them in (varies).
      cash           -- "Cash And Cash Equivalents" if available, else
                        "Cash Cash Equivalents And Short Term Investments".
      total_equity   -- "Stockholders Equity" or "Total Equity Gross Minority Interest"
                        (first match). May include minority interest.
      shares_diluted -- diluted weighted-average shares from the income statement.

    Index:   DatetimeIndex, daily (trading days only)
    Columns: pe_ttm (float), pb (float), ev_ebit (float)
             All ratios are dimensionless multiples (e.g. 33.4 means 33.4x).
    """
    with closing(get_connection()) as conn:
        prices = _load_prices(conn, ticker)
        rows = conn.execute(
            """
            SELECT report_date, net_income, operating_income,
                   shares_diluted, total_equity, total_debt, cash
            FROM financials_quarterly
            WHERE ticker = ? AND report_date IS NOT NULL
            ORDER BY report_date
            """,
            (ticker,),
        ).fetchall()

    if prices.empty:
        raise ValueError(f"No price data for {ticker!r}. Run ingest_prices first.")
    if not rows:
        raise ValueError(
            f"No quarterly fundamentals for {ticker!r}. Run ingest_fundamentals first."
        )

    fund = pd.DataFrame(rows, columns=[
        "report_date", "net_income", "operating_income",
        "shares_diluted", "total_equity", "total_debt", "cash",
    ])
    fund["report_date"] = pd.to_datetime(fund["report_date"])
    fund = fund.sort_values("report_date").reset_index(drop=True)

    # TTM: rolling 4-quarter sum of flow items at each report_date, before merge.
    # min_periods=4 kept explicitly: documents the boundary condition (not an accident).
    fund["net_income_ttm"] = fund["net_income"].rolling(4, min_periods=4).sum()
    fund["op_income_ttm"] = fund["operating_income"].rolling(4, min_periods=4).sum()
    fund = fund.drop(columns=["net_income", "operating_income"])

    # For each trading day, carry forward the most recent quarterly report
    prices_df = prices.rename("adj_close").to_frame()
    merged = pd.merge_asof(
        prices_df,
        fund.set_index("report_date"),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    market_cap = merged["adj_close"] * merged["shares_diluted"]

    # P/E TTM: NaN when earnings are negative, zero, or absent
    pe = (market_cap / merged["net_income_ttm"]).where(merged["net_income_ttm"] > 0)

    # P/B: NaN when book equity is non-positive (see NaN conditions in docstring)
    pb = (market_cap / merged["total_equity"]).where(merged["total_equity"] > 0)

    # EV/EBIT TTM: NULL debt/cash treated as 0 (best-effort; see column definitions)
    ev = market_cap + merged["total_debt"].fillna(0) - merged["cash"].fillna(0)
    ev_ebit = (ev / merged["op_income_ttm"]).where(merged["op_income_ttm"] > 0)

    return pd.DataFrame(
        {"pe_ttm": pe, "pb": pb, "ev_ebit": ev_ebit},
        index=merged.index,
    )


def quality_ratios(ticker: str) -> pd.DataFrame:
    """
    Quality and solvency ratios at each report_date from annual and quarterly filings.

    Two source DataFrames are built independently (one from financials_annual, one from
    financials_quarterly), then concatenated with a 'frequency' discriminator column.

    TTM treatment:
      - Quarterly: rolling 4-quarter sum for net_income and operating_income
        (min_periods=4). NaN before 4 quarters of history accumulate.
      - Annual: the annual flow figure IS the TTM by definition — no rolling needed.

    gross_margin uses single-period gross_profit / revenue (not TTM) because margin
    is a rate, not a level.

    NULL total_debt is treated as 0 throughout (fillna(0) applied to both cap_employed
    and debt_equity). For US-listed equities, NULL debt in yfinance almost always
    means zero debt, not missing data.

    NaN conditions:
      - total_equity <= 0: ROE, ROCE, debt_equity all NaN.
      - cap_employed (equity + debt) <= 0: ROCE NaN.
      - revenue <= 0: gross_margin NaN.
      - net_income_ttm / op_income_ttm NaN (< 4 quarters of quarterly history).
      - No forward-fill — gaps at the report_date level are informative.

    Index:   DatetimeIndex at report_date, sorted ascending (annual + quarterly mixed).
             NOT guaranteed unique — annual and quarterly reports for the same fiscal
             period can share a date when filed together (e.g. Q4 + FY same day).
             Downstream consumers should filter by 'frequency' before scalar lookup.
    Columns: frequency (str: 'annual'|'quarterly'),
             roe (float), roce (float) — decimal fractions (0.30 = 30%),
             gross_margin (float) — decimal fraction (0.43 = 43%),
             debt_equity (float) — dimensionless ratio (1.5 = 1.5x).
    """
    _COLS = [
        "report_date", "revenue", "gross_profit", "net_income",
        "operating_income", "total_equity", "total_debt",
    ]
    _SQL = """
        SELECT report_date, revenue, gross_profit, net_income, operating_income,
               total_equity, total_debt
        FROM {table}
        WHERE ticker = ? AND report_date IS NOT NULL
        ORDER BY report_date
    """

    with closing(get_connection()) as conn:
        annual_rows = conn.execute(_SQL.format(table="financials_annual"), (ticker,)).fetchall()
        quarterly_rows = conn.execute(_SQL.format(table="financials_quarterly"), (ticker,)).fetchall()

    def _ratios(rows: list, frequency: str, ttm_rolling: bool) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=_COLS)
        df["report_date"] = pd.to_datetime(df["report_date"])
        df = df.sort_values("report_date").reset_index(drop=True)

        if ttm_rolling:
            net_inc = df["net_income"].rolling(4, min_periods=4).sum()
            op_inc = df["operating_income"].rolling(4, min_periods=4).sum()
        else:
            net_inc = df["net_income"]
            op_inc = df["operating_income"]

        df = df.set_index("report_date")
        net_inc.index = df.index
        op_inc.index = df.index

        eq = df["total_equity"]
        debt = df["total_debt"].fillna(0)
        cap_employed = eq + debt

        roe = (net_inc / eq).where(eq > 0)
        roce = (op_inc / cap_employed).where((eq > 0) & (cap_employed > 0))
        gross_margin = (df["gross_profit"] / df["revenue"]).where(df["revenue"] > 0)
        debt_equity = (debt / eq).where(eq > 0)

        return pd.DataFrame({
            "frequency": frequency,
            "roe": roe,
            "roce": roce,
            "gross_margin": gross_margin,
            "debt_equity": debt_equity,
        })

    ann = _ratios(annual_rows, "annual", ttm_rolling=False)
    qrt = _ratios(quarterly_rows, "quarterly", ttm_rolling=True)

    parts = [p for p in (ann, qrt) if not p.empty]
    if not parts:
        raise ValueError(f"No fundamentals for {ticker!r}. Run ingest_fundamentals first.")

    return pd.concat(parts).sort_index()


def price_history(ticker: str, benchmark: str = "SPY") -> pd.DataFrame:
    """
    Indexed price history for ticker and benchmark, normalized to 100 at first shared date.

    Source: prices_daily.adj_close for both series.
    Join: outer — all trading dates present in either series.
    Forward-fill: up to 3 calendar days to bridge holiday mismatches between markets.
    Leading rows before both series have started are dropped.

    Index:   DatetimeIndex, daily (trading days only)
    Columns: ticker (float), benchmark (float)
             Both equal 100.0 on the first date where both series have data.
    """
    with closing(get_connection()) as conn:
        prices_t = _load_prices(conn, ticker)
        prices_b = _load_prices(conn, benchmark)

    if prices_t.empty:
        raise ValueError(f"No price data for {ticker!r}. Run ingest_prices first.")
    if prices_b.empty:
        raise ValueError(
            f"No price data for benchmark {benchmark!r}. Run ingest_benchmarks first."
        )

    merged = pd.concat(
        [prices_t.rename(ticker), prices_b.rename(benchmark)],
        axis=1,
        join="outer",
    )
    merged.sort_index(inplace=True)

    # Reindex to full calendar before ffill so limit=3 counts calendar days, not
    # trading-day positions. Handles cross-market holiday mismatches (e.g. an LSE
    # benchmark vs US ticker in a future phase); a no-op for same-calendar pairs
    # like US equities vs SPY.
    trading_dates = merged.index
    cal_idx = pd.date_range(trading_dates[0], trading_dates[-1], freq="D")
    merged = merged.reindex(cal_idx).ffill(limit=3).reindex(trading_dates)

    # Drop leading rows before both series have data
    first_valid = max(
        merged[ticker].first_valid_index(),
        merged[benchmark].first_valid_index(),
    )
    merged = merged.loc[first_valid:].copy()

    # Normalize to 100 at first shared date
    merged[ticker] = merged[ticker] / merged[ticker].iloc[0] * 100
    merged[benchmark] = merged[benchmark] / merged[benchmark].iloc[0] * 100

    return merged
