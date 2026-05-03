"""Run the Magic Formula screen and print results to stdout."""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from equity_research.analytics.screens import magic_formula


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Magic Formula screen.")
    parser.add_argument(
        "--as-of-date",
        default=str(date.today()),
        help="Screen date in YYYY-MM-DD format (default: today).",
    )
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    as_of_date = pd.Timestamp(args.as_of_date)
    result, stats = magic_formula(as_of_date, top_n=args.top_n)

    print(f"Magic Formula screen — {as_of_date.date()}")
    print(f"  Input tickers:       {stats['input_count']}")
    print(f"  After sector filter: {stats['post_sector_filter']}")
    print(f"  After NaN drop:      {stats['post_nan_drop']}")
    print(f"  Ranked (top {args.top_n}):      {stats['final_count']}")
    print()

    pd.set_option("display.max_rows", 50)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(result.to_string())


if __name__ == "__main__":
    main()
