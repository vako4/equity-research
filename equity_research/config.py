from pathlib import Path

# config.py lives at equity_research/config.py; two .parent calls reach the project root.
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "equity.db"
UNIVERSE_SNAPSHOT_DIR = DATA_DIR / "universe_snapshots"

# Schema defaults — currency and point-in-time filing lag estimates.
DEFAULT_CURRENCY = "USD"
REPORT_LAG_ANNUAL_DAYS = 90
REPORT_LAG_QUARTERLY_DAYS = 45

# Pull 12 years so tickers with minor data gaps still clear the 10-year target.
# Survivorship bias note: history is pulled for the *current* S&P 500 universe,
# so earlier years contain only companies that survived to today. See
# docs/data_limitations.md for full discussion.
PRICES_YEARS_BACK = 12

# Wikipedia S&P 500 constituent table.
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# yfinance batch download: tickers per request and sleep between batches.
# Starting values — increase PRICE_BATCH_SLEEP_SECONDS if ingestion produces HTTP 429 errors.
PRICE_BATCH_SIZE = 100
PRICE_BATCH_SLEEP_SECONDS = 2.0

# Fundamentals are fetched one ticker at a time.
# Starting value — increase if ingestion produces HTTP 429 errors.
FUNDAMENTALS_SLEEP_SECONDS = 1.0
