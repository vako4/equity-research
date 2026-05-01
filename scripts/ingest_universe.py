"""
Refresh the S&P 500 universe from Wikipedia and persist to DB.

Scrapes the constituent table, saves a CSV snapshot, upserts active tickers,
and soft-deletes removed tickers (is_active=0, removed_date stamped). Rows are
never hard-deleted — historical membership is preserved for point-in-time
universe construction in backtests.

Usage
-----
    python scripts/ingest_universe.py           # full refresh
    python scripts/ingest_universe.py --dry-run # show config, no network call

Exit codes
----------
    0  success — DB updated, snapshot written
    1  refresh failed — network error, Wikipedia parse failure, or DB write error
    import errors exit non-zero before main() runs (Python default, not via sys.exit)
"""

import argparse
import logging
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

from equity_research.config import DB_PATH, SP500_WIKI_URL, UNIVERSE_SNAPSHOT_DIR
from equity_research.db import get_connection
from equity_research.ingestion.universe import refresh_universe

LOG_DIR = Path(__file__).parent.parent / "logs"

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configuration and exit without fetching data",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run - no data will be fetched")
        print(f"  Source:       {SP500_WIKI_URL}")
        print(f"  Snapshot dir: {UNIVERSE_SNAPSHOT_DIR}")
        print(f"  DB:           {DB_PATH}")
        sys.exit(0)

    log.info("Starting universe refresh | source=%s", SP500_WIKI_URL)

    try:
        with closing(get_connection()) as conn:
            stats = refresh_universe(conn=conn)

            removed_tickers: list[str] = []
            if stats["removed_count"] > 0:
                rows = conn.execute(
                    "SELECT ticker FROM universe WHERE is_active = 0 AND removed_date = ?",
                    (date.today().isoformat(),),
                ).fetchall()
                removed_tickers = [r[0] for r in rows]
    except Exception as exc:
        log.error("Universe refresh failed: %s", exc)
        sys.exit(1)

    log.info(
        "Done | upserted=%d removed=%d snapshot=%s",
        stats["rows_upserted"], stats["removed_count"], stats["snapshot_path"],
    )

    if removed_tickers:
        log.warning(
            "S&P 500 reconstitution - removed tickers (%d): %s",
            len(removed_tickers), ", ".join(sorted(removed_tickers)),
        )


if __name__ == "__main__":
    main()
