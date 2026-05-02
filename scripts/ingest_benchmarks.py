"""
Load benchmark ETF prices (SPY, QQQ, IWM) into prices_daily.

These tickers are not S&P 500 constituents and are not in the standard
universe. They are inserted into universe with is_benchmark=1, is_active=0
so they satisfy the prices_daily FK but remain invisible to any query
filtering WHERE is_active = 1.

Safe to re-run: universe rows are upserted, prices are upserted.
"""

import logging
import sys
from contextlib import closing
from datetime import date

from equity_research.config import PRICES_YEARS_BACK
from equity_research.db import get_connection
from equity_research.ingestion.prices import _date_range, ingest_prices

BENCHMARKS = [
    ("SPY", "SPDR S&P 500 ETF Trust"),
    ("QQQ", "Invesco QQQ Trust"),
    ("IWM", "iShares Russell 2000 ETF"),
]

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def main() -> None:
    tickers = [t for t, _ in BENCHMARKS]
    log.info("Upserting benchmark tickers into universe: %s", tickers)

    try:
        with closing(get_connection()) as conn:
            conn.executemany(
                """
                INSERT INTO universe (ticker, company_name, added_date, is_active, is_benchmark)
                VALUES (?, ?, ?, 0, 1)
                ON CONFLICT(ticker) DO UPDATE SET
                    company_name = excluded.company_name,
                    is_benchmark = 1
                """,
                [(ticker, name, date.today().isoformat()) for ticker, name in BENCHMARKS],
            )
            conn.commit()
    except Exception as exc:
        log.error("Failed to upsert benchmark tickers: %s", exc)
        sys.exit(1)

    log.info("Fetching %d years of price history for benchmarks", PRICES_YEARS_BACK)

    try:
        start, end = _date_range(PRICES_YEARS_BACK)
        stats = ingest_prices(tickers=tickers, start_date=start, end_date=end)
    except Exception as exc:
        log.error("Price ingestion failed: %s", exc)
        sys.exit(1)

    log.info(
        "Done | succeeded=%d failed=%d total_rows=%d",
        stats["success"], stats["failed"], stats["total_rows"],
    )

    if stats["failed_tickers"]:
        log.warning("Failed: %s", ", ".join(stats["failed_tickers"]))
        sys.exit(1)


if __name__ == "__main__":
    main()
