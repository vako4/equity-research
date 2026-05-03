# Data Limitations

## Survivorship Bias

Universe reflects current S&P 500 membership only. Companies removed before today's first scrape are absent from all tables. Any backtest using this data overstates returns by excluding delisted and degraded companies.

TODO Phase 4: source point-in-time constituent lists before any multi-date backtest.

## Look-Ahead Bias (Partial Mitigation)

Cross-statement alignment uses a 45-day tolerance window; balance sheet columns more than 45 days from the income statement date are rejected. Fiscal period end dates are stored as reported. What we *don't* control: yfinance does not expose original filing dates, so `report_date` is estimated (period end + 90/45 days). Factors built on this data may embed mild look-ahead in the first weeks after a period closes.

Future work: replace estimated `report_date` with actual SEC EDGAR filing dates.

## Retroactive Price Adjustment

`adj_close` reflects split and dividend adjustments as of the fetch date. Re-running `ingest_prices` after a stock split silently overwrites history with newly adjusted prices. The series is internally consistent at any point in time but not stable across re-runs.

For backtest reproducibility, snapshot the prices table after ingestion and archive it alongside the strategy code.

## Fundamentals Coverage

yfinance returns approximately 4 years of annual periods and 5 quarters of quarterly data. This is shallow for factor research requiring 10+ years of history. Extending coverage requires a paid data source.

Concrete examples from the current run: BRK-B has an all-NULL 2021 annual row (yfinance returns the row but cannot populate it); GEV (April 2024 spinoff from GE) has annual rows back to 2022 derived from the parent's consolidated statements, which may not reflect GEV as a standalone entity.

## Universe Construction

Constituent list scraped from Wikipedia, not from an authoritative index provider. Wikipedia is generally accurate and updated promptly, but is not suitable for production-grade backtesting where precise reconstitution dates matter.
