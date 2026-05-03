"""Compare Magic Formula and FCF Quality screens at a given date."""
import pandas as pd

from equity_research.analytics.screens import fcf_quality_screen, magic_formula

AS_OF = pd.Timestamp("2026-05-03")
TOP_N = 30

mf_result,  mf_stats  = magic_formula(AS_OF, top_n=TOP_N)
fcf_result, fcf_stats = fcf_quality_screen(AS_OF, top_n=TOP_N)

pd.set_option("display.max_rows", 50)
pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.width", 120)

# --- Stats ---
print(f"Screen comparison — {AS_OF.date()}\n")
print(f"{'':30s} {'Magic Formula':>15} {'FCF Quality':>12}")
print(f"{'Input tickers':30s} {mf_stats['input_count']:>15} {fcf_stats['input_count']:>12}")
print(f"{'After sector filter':30s} {mf_stats['post_sector_filter']:>15} {fcf_stats['post_sector_filter']:>12}")
print(f"{'After NaN drop':30s} {mf_stats['post_nan_drop']:>15} {fcf_stats['post_nan_drop']:>12}")
print(f"{'Final (top 30)':30s} {mf_stats['final_count']:>15} {fcf_stats['final_count']:>12}")

# --- Magic Formula top 30 ---
print("\n=== MAGIC FORMULA TOP 30 ===")
print(mf_result.to_string())

# --- FCF Quality top 30 ---
print("\n=== FCF QUALITY SCREEN TOP 30 ===")
print(fcf_result.to_string())

# --- Overlap ---
mf_tickers  = set(mf_result.index)
fcf_tickers = set(fcf_result.index)
both        = sorted(mf_tickers & fcf_tickers)
only_mf     = sorted(mf_tickers - fcf_tickers)
only_fcf    = sorted(fcf_tickers - mf_tickers)

print("\n=== OVERLAP ANALYSIS ===")
print(f"Overlap: {len(both)} / {TOP_N} tickers ({len(both)/TOP_N*100:.1f}%)")
print(f"\nIn both ({len(both)}):              {both}")
print(f"Unique to Magic Formula ({len(only_mf)}): {only_mf}")
print(f"Unique to FCF Quality   ({len(only_fcf)}): {only_fcf}")
