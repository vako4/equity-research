import sqlite3

import pytest

from equity_research.db import init_db


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys=ON")
    init_db(conn=c)
    yield c
    c.close()


@pytest.fixture
def conn_with_ticker(conn):
    conn.execute(
        """
        INSERT INTO universe (ticker, company_name, sector, industry, added_date, is_active)
        VALUES ('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics', '2020-01-01', 1)
        """
    )
    conn.commit()
    yield conn
