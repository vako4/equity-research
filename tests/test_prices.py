from datetime import date

import pandas as pd

from equity_research.ingestion.prices import _date_range, _upsert_prices


def _price_df(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open":   [150.0] * len(dates),
            "High":   [155.0] * len(dates),
            "Low":    [149.0] * len(dates),
            "Close":  [153.0] * len(dates),
            "Volume": [1_000_000] * len(dates),
        },
        index=pd.to_datetime(dates),
    )


# Test 2
def test_price_upsert_idempotent(conn_with_ticker):
    conn = conn_with_ticker
    df = _price_df(["2024-01-02", "2024-01-03"])

    _upsert_prices("AAPL", df, conn)
    conn.commit()
    _upsert_prices("AAPL", df, conn)
    conn.commit()

    rows = conn.execute("SELECT COUNT(*) FROM prices_daily WHERE ticker = 'AAPL'").fetchone()[0]
    assert rows == 2


# Test 7
def test_date_range_leap_year_span():
    """12-year window uses 365.25 days/year — result must not fall short by a year."""
    # Anchor after a leap day so a plain 365*12 multiplier would land 3 days short.
    fixed_today = date(2024, 3, 1)
    start_str, end_str = _date_range(12, today=fixed_today)
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    assert end == fixed_today
    assert (end - start).days == int(12 * 365.25)  # 4383, not 4380
