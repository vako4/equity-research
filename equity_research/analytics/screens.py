from __future__ import annotations

from contextlib import closing

import pandas as pd

from equity_research.db import get_connection
from equity_research.analytics import cross_section, factors

# Sectors where EV-based and capital-efficiency ratios lose comparability
_EXCLUDED_SECTORS = {"Financials", "Real Estate"}


def magic_formula(
    as_of_date: pd.Timestamp, top_n: int = 30
) -> tuple[pd.DataFrame, dict[str, int]]:
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT ticker, sector FROM universe WHERE is_active = 1 AND is_benchmark = 0"
        ).fetchall()

    universe = pd.DataFrame(rows, columns=["ticker", "sector"]).set_index("ticker")
    tickers = universe.index.tolist()

    val_df = cross_section.compute(tickers, as_of_date, factors.valuation_factor)
    qual_df = cross_section.compute(tickers, as_of_date, factors.quality_factor)

    combined = val_df.join(qual_df)
    combined = combined.join(universe["sector"])

    # Exclude Financials and Real Estate: EV/EBIT and ROCE are not comparable
    # across these sectors due to leverage-as-product and asset-revaluation mechanics
    sector_filtered = combined[~combined["sector"].isin(_EXCLUDED_SECTORS)]

    # dropna(subset=[...]) drops a row if EITHER factor is NaN, not just when both are
    ranked = sector_filtered.dropna(subset=["ev_ebit", "roce"])

    ranked = ranked.copy()
    ranked["ev_ebit_rank"] = ranked["ev_ebit"].rank(ascending=True)
    ranked["roce_rank"] = ranked["roce"].rank(ascending=False)
    ranked["combined_rank"] = ranked["ev_ebit_rank"] + ranked["roce_rank"]
    ranked = ranked.sort_values("combined_rank")

    stats = {
        "input_count": len(tickers),
        "post_sector_filter": len(sector_filtered),
        "post_nan_drop": len(ranked),
        "final_count": min(top_n, len(ranked)),
    }

    return ranked[
        ["sector", "ev_ebit", "roce", "ev_ebit_rank", "roce_rank", "combined_rank"]
    ].head(top_n), stats


def fcf_quality_screen(
    as_of_date: pd.Timestamp, top_n: int = 30
) -> tuple[pd.DataFrame, dict[str, int]]:
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT ticker, sector FROM universe WHERE is_active = 1 AND is_benchmark = 0"
        ).fetchall()

    universe = pd.DataFrame(rows, columns=["ticker", "sector"]).set_index("ticker")
    tickers = universe.index.tolist()

    fcf_df = cross_section.compute(tickers, as_of_date, factors.fcf_yield_factor)
    qual_df = cross_section.compute(tickers, as_of_date, factors.quality_factor)

    # Join only the roce column from qual_df — other quality factors not needed here
    combined = fcf_df.join(qual_df[["roce"]])
    combined = combined.join(universe["sector"])

    # Exclude Financials and Real Estate: EV-based and capital-efficiency ratios
    # are not comparable across these sectors due to leverage-as-product and
    # asset-revaluation mechanics
    sector_filtered = combined[~combined["sector"].isin(_EXCLUDED_SECTORS)]

    # dropna drops row if EITHER factor is NaN, not just when both are
    eligible = sector_filtered.dropna(subset=["fcf_yield", "roce"])

    # Rank within the full eligible universe before applying top_n cutoff,
    # so ranks reflect standing in the universe, not just within the returned slice
    eligible = eligible.copy()
    eligible["fcf_yield_rank"] = eligible["fcf_yield"].rank(ascending=False)
    eligible["roce_rank"] = eligible["roce"].rank(ascending=False)
    eligible["combined_rank"] = eligible["fcf_yield_rank"] + eligible["roce_rank"]

    result = eligible.sort_values("combined_rank").head(top_n)

    stats = {
        "input_count": len(tickers),
        "post_sector_filter": len(sector_filtered),
        "post_nan_drop": len(eligible),
        "final_count": len(result),
    }

    return result[
        ["sector", "fcf_yield", "roce", "fcf_yield_rank", "roce_rank", "combined_rank"]
    ], stats
