"""
Ingest adjusted OHLCV prices for the active S&P 500 universe.

Usage
-----
    python scripts/ingest_prices.py                  # full run, active universe
    python scripts/ingest_prices.py --tickers AAPL MSFT
    python scripts/ingest_prices.py --years-back 5
    python scripts/ingest_prices.py --dry-run        # plan preview, no network calls

Resume behavior
---------------
Re-running after a partial failure is safe: every insert is an upsert, so
already-ingested rows are overwritten with identical data rather than
duplicated. The tradeoff is that successfully ingested tickers are re-fetched
in full — there is no --skip-existing flag. For a 500-ticker universe this
adds ~10-20 minutes of redundant network time; acceptable given how rarely
partial failures occur in practice. If you need to re-run only the failed
subset, pass their tickers explicitly via --tickers (the list is written to
logs/failed_tickers_<timestamp>.txt after any run with failures).
"""

import argparse
import logging
import math
import sys
from contextlib import closing
from pathlib import Path

from equity_research.config import PRICE_BATCH_SIZE, PRICES_YEARS_BACK
from equity_research.db import get_connection
from equity_research.ingestion.prices import _date_range, ingest_prices

LOG_DIR = Path(__file__).parent.parent / "logs"

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def _resolve_tickers(explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT ticker FROM universe WHERE is_active = 1 ORDER BY ticker"
        ).fetchall()
    return [r[0] for r in rows]


def _write_failed(tickers: list[str], start: str, end: str) -> Path:
    from datetime import datetime, timezone

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOG_DIR / f"failed_tickers_{ts}.txt"
    lines = [f"# failed tickers | date range {start} to {end}", *tickers]
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Explicit ticker list (default: active universe)",
    )
    parser.add_argument(
        "--years-back", type=int, default=PRICES_YEARS_BACK, metavar="N",
        help=f"Lookback window in years (default: {PRICES_YEARS_BACK})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and exit without fetching data",
    )
    args = parser.parse_args()

    try:
        tickers = _resolve_tickers(args.tickers)
    except Exception as exc:
        log.error("Failed to resolve ticker list: %s", exc)
        sys.exit(2)

    # Compute the date range once so dry-run preview and live run are identical.
    start, end = _date_range(args.years_back)
    source = "explicit list" if args.tickers else "active universe"
    batch_count = math.ceil(len(tickers) / PRICE_BATCH_SIZE) if tickers else 0

    if args.dry_run:
        print("Dry run - no data will be fetched")
        print(f"  Source:      {source}")
        print(f"  Tickers:     {len(tickers)}")
        print(f"  Date range:  {start} to {end}")
        print(f"  Batches:     {batch_count} (batch size {PRICE_BATCH_SIZE})")
        sys.exit(0)

    log.info(
        "Starting price ingestion | source=%s tickers=%d date_range=%s:%s batches=%d",
        source, len(tickers), start, end, batch_count,
    )

    try:
        stats = ingest_prices(tickers=tickers, start_date=start, end_date=end)
    except Exception as exc:
        log.error("Ingestion aborted with unhandled exception: %s", exc)
        sys.exit(2)

    log.info(
        "Done | succeeded=%d failed=%d total_rows=%d",
        stats["success"], stats["failed"], stats["total_rows"],
    )

    if stats["failed_tickers"]:
        log.warning(
            "Failed tickers (%d): %s",
            len(stats["failed_tickers"]), ", ".join(stats["failed_tickers"]),
        )
        try:
            path = _write_failed(stats["failed_tickers"], start, end)
            log.warning("Failed ticker list written to %s", path)
        except Exception as exc:
            log.error("Could not write failed ticker file: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
