# Phase 3 Planning: Cross-Sectional Screens

**Status:** Decision doc — lock before any screens.py work begins.  
**Scope:** Phase 3 (factor screens) through the seam into Phase 4 (backtesting).

---

## 1. Output Shape for Factor Screens

**Decision:** DataFrames throughout the analytics layer. HTML rendering is a separate, thin wrapper applied at report-generation time.

This preserves the Phase 2 invariant: analytics modules return DataFrames, rendering is never their concern.

**Screen logic lives in:** `equity_research/analytics/screens.py` (new file).  
`cross_section.py` owns data assembly; `screens.py` owns filtering and ranking logic applied to the assembled DataFrame.

Typical screen output shape — one row per ticker, one column per factor:

```
ticker | pe_ttm | ev_ebitda | roe | roce | ... | sector
```

Returned as `pd.DataFrame` indexed by ticker. Callers decide whether to render as HTML, write to CSV, or feed into a backtest loop.

---

## 2. Cross-Sectional Layer API

**Module:** `equity_research/analytics/cross_section.py` (new file).

**Primary signature:**

```
compute(
    tickers: list[str],
    as_of_date: pd.Timestamp,
    ratio_fn: Callable[[str, pd.Timestamp], pd.Series],
) -> pd.DataFrame   # wide: tickers × factors
```

**`ratio_fn` return type — `pd.Series`, not `dict[str, float]`.**  
Phase 2 `ratios.py` functions (`valuation_ratios`, `quality_ratios`) return time-indexed DataFrames. The wrapper's job is to select the point-in-time row for `as_of_date` (via `merge_asof` or `.loc`) and return it as a `pd.Series` with factor names as the index. Coercion from DataFrame → Series happens inside the wrapper, not inside `compute`. `compute` concatenates one Series per ticker into the wide output DataFrame and never touches DataFrame-shaped intermediates.

**`ratio_fn` wrappers live in:** `equity_research/analytics/factors.py` (new file).  
This module contains thin, per-factor-group wrapper functions — one calling into `valuation_ratios`, one into `quality_ratios`, etc. Each accepts `(ticker, as_of_date)` and returns a `pd.Series`. `cross_section.compute` imports nothing from `ratios.py`; it operates only on the `ratio_fn` it is passed. Callers compose factor groups by passing a wrapper from `factors.py`, or by passing a lambda that combines multiple wrappers.

**Other decisions:**

- One ticker at a time, not batch. `ratios.py` functions are per-ticker; `concurrent.futures` parallelism can be layered on later without changing the `compute` signature.
- **Phase 4 reuse:** `compute` is called once per rebalance date. The rebalance loop is external — that is by design.

---

## 3. Sector / Edge-Case Policy

**Decision:** `ratios.py` computes ratios universally, no sector awareness. Sector filters are applied in `screens.py` after the wide DataFrame is assembled. Do not bake sector logic into `ratios.py`.

**Edge cases to handle explicitly:**

| Case | Where handled | Policy |
|---|---|---|
| Negative book equity (ROCE denominator ≤ 0) | `factors.py` wrapper | Return `NaN`. A screen ranking on ROCE should not surface companies with negative denominators as outliers — a negative denominator produces a number that sorts into "top quintile" for the wrong reason, so NaN-and-exclude is the correct ranking semantic. |
| Financials / REITs on EV-based ratios (EV/EBITDA, EV/FCF) | `screens.py` filter | Exclude by GICS sector code before ranking |
| Truncated TTM for recent listings (<4 quarters available) | `factors.py` wrapper | Return `NaN` if TTM cannot be fully constructed |
| NULL `total_debt` convention | Already in `data_limitations.md` | Treat NULL as zero debt (existing behavior) |

NaN propagates cleanly through ranking. Each screen must document its NaN-drop policy (e.g., require N of M factors non-null before including a ticker in results).

---

## 4. `as_of_date` as First-Class Parameter

**Decision:** Every cross-sectional call takes `as_of_date` explicitly. No implicit `datetime.today()` anywhere in the analytics layer.

`as_of_date` flows as:

```
compute(tickers, as_of_date, ratio_fn)
  → ratio_fn(ticker, as_of_date)          # once per ticker, in factors.py
    → ratios.py function → DataFrame
    → point-in-time slice → pd.Series     # coercion inside factors.py wrapper
```

The `merge_asof` discipline from Phase 2 (point-in-time fundamentals lookup) is applied per ticker — ~500 merge operations for one screen date. Acceptable for Phase 3. In Phase 4 this runs once per rebalance date, where caching will matter.

**Why this matters now:** `as_of_date` is the parameter that becomes the outer loop variable in Phase 4. Making it an explicit argument — rather than defaulting to today — keeps every screen reproducible by date and avoids a retrofit when the backtest loop is wired up.

**Phase 3 rendering entry point:**

```
build_screen(as_of_date: pd.Timestamp, screen_fn: Callable[[pd.DataFrame], pd.DataFrame]) -> None
```

Calls `compute`, applies `screen_fn`, writes HTML. No business logic in the entry point.
