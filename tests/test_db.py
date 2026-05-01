import sqlite3

import pandas as pd
import pytest

from equity_research.ingestion.prices import _upsert_prices


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


# Test 1
def test_fk_rejects_unknown_ticker(conn):
    """Inserting a price row for a ticker absent from universe raises IntegrityError."""
    with pytest.raises(sqlite3.IntegrityError):
        _upsert_prices("UNKNOWN", _price_df(["2024-01-02"]), conn)


# Test 9
def test_active_universe_both_directions(conn):
    """Active-universe query includes is_active=1 tickers and excludes is_active=0."""
    conn.executemany(
        "INSERT INTO universe (ticker, added_date, is_active) VALUES (?, '2020-01-01', ?)",
        [("AAPL", 1), ("MSFT", 1), ("REMOVED", 0)],
    )
    conn.commit()

    active = {
        r[0] for r in conn.execute("SELECT ticker FROM universe WHERE is_active = 1")
    }
    assert "AAPL" in active       # present and active — must appear
    assert "MSFT" in active
    assert "REMOVED" not in active  # explicitly removed — must not appear
