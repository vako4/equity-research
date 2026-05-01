"""
Ingest annual and quarterly fundamentals for the active S&P 500 universe.

Usage
-----
    python scripts/ingest_fundamentals.py                  # full run, active universe
    python scripts/ingest_fundamentals.py --tickers AAPL MSFT
    python scripts/ingest_fundamentals.py --dry-run        # plan preview, no network calls

Resume behavior
---------------
Re-running after a partial failure is safe: every insert is an upsert, so
already-ingested rows are overwritten with identical data rather than
duplicated. The tradeoff is that successfully ingested tickers are re-fetched
in full - there is no --skip-existing flag. For a 500-ticker universe a full
run takes ~8-10 minutes; re-running only the failed subset via --tickers is
usually the right call. The failure file written to logs/ lists every problem
ticker with a tag indicating which grain(s) failed (see below).

Failure tags in logs/failed_tickers_fundamentals_<timestamp>.txt
-----------------------------------------------------------------
  both              annual and quarterly both failed
  partial:annual    annual failed, quarterly succeeded
  partial:quarterly quarterly failed, annual succeeded
"""

import argparse
import logging
import math
import sys
from contextlib import closing
from pathlib import Path

from equity_research.config import FUNDAMENTALS_SLEEP_SECONDS
from equity_research.db import get_connection
from equity_research.ingestion.fundamentals import ingest_fundamentals

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


def _write_failures(stats: dict) -> Path:
    from datetime import datetime, timezone

    both_set = set(stats["failed_tickers"])
    annual_fail_set = set(stats["annual_failed_tickers"])
    quarterly_fail_set = set(stats["quarterly_failed_tickers"])
    all_failed = annual_fail_set | quarterly_fail_set

    tagged: list[tuple[str, str]] = []
    for ticker in sorted(all_failed):
        if ticker in both_set:
            tag = "both"
        elif ticker in annual_fail_set:
            tag = "partial:annual"
        else:
            tag = "partial:quarterly"
        tagged.append((ticker, tag))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOG_DIR / f"failed_tickers_fundamentals_{ts}.txt"

    col_width = max(len(t) for t, _ in tagged) + 2
    lines = [
        "# tag key: both=both grains failed | partial:annual=annual failed only | partial:quarterly=quarterly failed only",
        *[f"{ticker:<{col_width}}{tag}" for ticker, tag in tagged],
    ]
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
        "--dry-run", action="store_true",
        help="Print plan and exit without fetching data",
    )
    args = parser.parse_args()

    try:
        tickers = _resolve_tickers(args.tickers)
    except Exception as exc:
        log.error("Failed to resolve ticker list: %s", exc)
        sys.exit(2)

    source = "explicit list" if args.tickers else "active universe"
    est_minutes = math.ceil(len(tickers) * FUNDAMENTALS_SLEEP_SECONDS / 60)

    if args.dry_run:
        print("Dry run - no data will be fetched")
        print(f"  Source:           {source}")
        print(f"  Tickers:          {len(tickers)}")
        print(f"  Est. runtime:     ~{est_minutes} min ({FUNDAMENTALS_SLEEP_SECONDS}s sleep between tickers)")
        sys.exit(0)

    log.info(
        "Starting fundamentals ingestion | source=%s tickers=%d est_runtime=~%dmin",
        source, len(tickers), est_minutes,
    )

    try:
        stats = ingest_fundamentals(tickers=tickers)
    except Exception as exc:
        log.error("Ingestion aborted with unhandled exception: %s", exc)
        sys.exit(2)

    log.info(
        "Annual:    succeeded=%d failed=%d rows=%d",
        stats["annual_success"], stats["annual_failed"], stats["annual_rows"],
    )
    log.info(
        "Quarterly: succeeded=%d failed=%d rows=%d",
        stats["quarterly_success"], stats["quarterly_failed"], stats["quarterly_rows"],
    )

    any_failure = bool(stats["annual_failed_tickers"] or stats["quarterly_failed_tickers"])

    if any_failure:
        n_both = len(stats["failed_tickers"])
        n_partial = len(set(stats["annual_failed_tickers"]) | set(stats["quarterly_failed_tickers"])) - n_both
        log.warning("Failures: %d both-grain, %d partial", n_both, n_partial)
        try:
            path = _write_failures(stats)
            log.warning("Failure details written to %s", path)
        except Exception as exc:
            log.error("Could not write failure file: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
