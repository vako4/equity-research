"""
Build a self-contained single-stock tearsheet HTML file.

Output lands in reports/<TICKER>_<YYYY-MM-DD>.html.
No network required to open — plotly.js is inlined (~3-4 MB).

Usage
-----
    python scripts/build_tearsheet.py --ticker AAPL
    python scripts/build_tearsheet.py --ticker AAPL --benchmark QQQ
    python scripts/build_tearsheet.py --ticker AAPL --output reports/custom.html
"""

import argparse
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Environment, FileSystemLoader

from equity_research.analytics.ratios import (
    drawdown, fundamentals_snapshot, price_history,
    quality_ratios, returns_table, valuation_ratios,
)
from equity_research.db import get_connection

REPORTS_DIR = Path(__file__).parent.parent / "reports"
TEMPLATES_DIR = Path(__file__).parent.parent / "equity_research" / "templates"

HORIZON_LABELS = ["1Y", "3Y", "5Y", "10Y"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val: float) -> str:
    """Decimal return → coloured percentage string. Dash for NaN."""
    if pd.isna(val):
        return "—"
    cls = "pos" if val >= 0 else "neg"
    return f'<span class="{cls}">{val:+.2%}</span>'


# ---------------------------------------------------------------------------
# Price history section builder
# ---------------------------------------------------------------------------

def _price_chart_html(df: pd.DataFrame, ticker: str, benchmark: str) -> str:
    """Log-scale indexed price chart, both series normalized to 100 at first shared date.
    INLINE RULE: this is the topmost chart in template render order — it carries
    include_plotlyjs="inline" to embed the single plotly.js bundle. All other
    chart helpers must use include_plotlyjs=False. Reordering charts in the
    template requires moving "inline" to whichever helper renders first."""
    fig = go.Figure([
        go.Scatter(
            x=df.index.tolist(),
            y=df[ticker].round(2).tolist(),
            name=ticker,
            line=dict(color="#0d6efd", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>",
        ),
        go.Scatter(
            x=df.index.tolist(),
            y=df[benchmark].round(2).tolist(),
            name=benchmark,
            line=dict(color="#adb5bd", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>",
        ),
    ])
    fig.update_layout(
        yaxis_type="log",
        yaxis_title="Indexed price (log scale, base = 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=40, b=0),
        height=360,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                  size=12, color="#212529"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#dee2e6")
    return fig.to_html(full_html=False, include_plotlyjs="inline")


# ---------------------------------------------------------------------------
# Returns section builders
# ---------------------------------------------------------------------------

def _returns_table_html(df: pd.DataFrame, ticker: str, benchmark: str) -> str:
    header = (
        f"<tr><th>Period</th><th>{ticker}</th>"
        f"<th>{benchmark}</th><th>Excess</th></tr>"
    )
    rows = "".join(
        f"<tr><td>{period}</td>"
        f"<td>{_fmt(row['ticker_ann'])}</td>"
        f"<td>{_fmt(row['bench_ann'])}</td>"
        f"<td>{_fmt(row['excess'])}</td></tr>"
        for period, row in df.iterrows()
    )
    note = (
        "<p class='note'>Annualized geometric returns. "
        "Inception shown as NaN when &lt;1 year of history. "
        "Source: prices_daily.adj_close.</p>"
    )
    return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>{note}"


def _returns_chart_html(df: pd.DataFrame, ticker: str, benchmark: str) -> str:
    """Grouped bar chart of annualized returns for the four standard horizons.
    Inception excluded — its scale can dwarf the others on short histories."""
    plot_df = df.loc[HORIZON_LABELS].copy()
    # Drop horizons where either series is NaN (e.g. ticker too young for 10Y).
    plot_df = plot_df.dropna(subset=["ticker_ann", "bench_ann"])

    fig = go.Figure([
        go.Bar(
            name=ticker,
            x=plot_df.index.tolist(),
            y=(plot_df["ticker_ann"] * 100).round(2).tolist(),
            marker_color="#0d6efd",
        ),
        go.Bar(
            name=benchmark,
            x=plot_df.index.tolist(),
            y=(plot_df["bench_ann"] * 100).round(2).tolist(),
            marker_color="#adb5bd",
        ),
    ])
    fig.update_layout(
        barmode="group",
        yaxis_title="Annualized Return (%)",
        yaxis_ticksuffix="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=40, b=0),
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                  size=12, color="#212529"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#dee2e6", zeroline=True, zerolinecolor="#adb5bd")

    # include_plotlyjs=False — bundle already inlined by _price_chart_html above.
    # See INLINE RULE comment in _price_chart_html.
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# Drawdown section builder
# ---------------------------------------------------------------------------

def _drawdown_chart_html(ticker: str) -> str:
    """Filled area chart of rolling drawdown from peak.
    include_plotlyjs=False — bundle already inlined by _price_chart_html."""
    df = drawdown(ticker)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index.tolist(),
        y=(df["drawdown"] * 100).round(2).tolist(),
        fill="tozeroy",
        fillcolor="rgba(220, 53, 69, 0.12)",
        line=dict(color="#dc3545", width=1),
        name="Drawdown",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        yaxis_title="Drawdown (%)",
        yaxis_ticksuffix="%",
        margin=dict(l=0, r=0, t=10, b=0),
        height=240,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                  size=12, color="#212529"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#dee2e6", zeroline=True, zerolinecolor="#adb5bd")
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# Valuation ratios section builder
# ---------------------------------------------------------------------------

_RATIO_PANELS = [
    ("pe_ttm",  "P/E (TTM)"),
    ("pb",      "P/B"),
    ("ev_ebit", "EV/EBIT (TTM)"),
]


def _valuation_chart_html(ticker: str) -> str:
    """Three stacked subplots — one per ratio — with independent y-axes.
    Each series is dropna()'d independently so P/B (1q history) renders its
    full range while P/E and EV/EBIT (4q TTM) start later.
    include_plotlyjs=False — bundle already inlined by _price_chart_html."""
    df = valuation_ratios(ticker)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[label for _, label in _RATIO_PANELS],
    )

    for i, (col, label) in enumerate(_RATIO_PANELS, start=1):
        series = df[col].dropna()
        fig.add_trace(
            go.Scatter(
                x=series.index.tolist(),
                y=series.round(2).tolist(),
                line=dict(color="#0d6efd", width=1.5),
                hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}x<extra></extra>",
                showlegend=False,
            ),
            row=i, col=1,
        )
        fig.update_yaxes(
            title_text=label,
            title_standoff=10,
            gridcolor="#dee2e6",
            row=i, col=1,
        )

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            size=12,
            color="#212529",
        ),
    )
    fig.update_xaxes(showgrid=False)
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# Fundamentals snapshot table builder
# ---------------------------------------------------------------------------

_SNAPSHOT_METRICS = [
    ("revenue",          "Revenue"),
    ("gross_profit",     "Gross Profit"),
    ("operating_income", "Op. Income"),
    ("net_income",       "Net Income"),
    ("eps_diluted",      "EPS (dil.)"),
]


def _snapshot_table_html(df: pd.DataFrame) -> str:
    """HTML table of the 4 most recent quarters with inline QoQ/YoY annotations.

    df: 5-row DataFrame from fundamentals_snapshot(), most-recent-first (descending).
    pct_change(-1) on descending data = (row[i] - row[i+1]) / row[i+1] = QoQ.
    pct_change(-4) on descending data = (row[i] - row[i+4]) / row[i+4] = YoY.
    NaN annotations are suppressed — cells show only what's available."""
    qoq = df.pct_change(periods=-1)
    yoy = df.pct_change(periods=-4)
    display = df.iloc[:4]

    def _fmt_val(col: str, val: float) -> str:
        if pd.isna(val):
            return "—"
        if col == "eps_diluted":
            return f"${val:.2f}"
        return f"${val / 1e9:.1f}B"

    def _fmt_pct(val: float, label: str) -> str:
        if pd.isna(val) or not np.isfinite(val):
            return ""
        cls = "pos" if val >= 0 else "neg"
        sign = "+" if val >= 0 else ""
        return f'<br><span class="note {cls}">{sign}{val:.1%} {label}</span>'

    header = (
        "<tr><th>Period</th>"
        + "".join(f"<th>{lbl}</th>" for _, lbl in _SNAPSHOT_METRICS)
        + "</tr>"
    )
    rows_html = ""
    for idx in display.index:
        period = idx.strftime("%b %Y")
        cells = f"<td>{period}</td>"
        for col, _ in _SNAPSHOT_METRICS:
            val_str = _fmt_val(col, display.loc[idx, col])
            qoq_ann = _fmt_pct(qoq.loc[idx, col], "QoQ")
            yoy_ann = _fmt_pct(yoy.loc[idx, col], "YoY")
            cells += f"<td>{val_str}{qoq_ann}{yoy_ann}</td>"
        rows_html += f"<tr>{cells}</tr>"

    note = (
        "<p class='note'>Revenue, Gross Profit, Op. Income, Net Income in USD billions. "
        "EPS = net_income / shares_diluted (raw diluted shares). "
        "QoQ and YoY vs same-source quarterly filings.</p>"
    )
    return f"<table><thead>{header}</thead><tbody>{rows_html}</tbody></table>{note}"


# ---------------------------------------------------------------------------
# Quality ratios section builder
# ---------------------------------------------------------------------------

_QUALITY_PANELS = [
    ("roe",          "ROE (TTM)"),
    ("roce",         "ROCE (TTM)"),
    ("gross_margin", "Gross Margin"),
    ("debt_equity",  "Debt / Equity"),
]


def _quality_chart_html(ticker: str) -> str:
    """4 stacked subplots, one per quality metric, annual + quarterly overlaid.
    Annual: solid width=2; quarterly: dashed width=1.5. Same #0d6efd colour.
    Legend appears once (first panel only) via legendgroup.
    include_plotlyjs=False — bundle already inlined by _price_chart_html."""
    df = quality_ratios(ticker)
    ann = df[df["frequency"] == "annual"]
    qrt = df[df["frequency"] == "quarterly"]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[label for _, label in _QUALITY_PANELS],
    )

    for i, (col, _) in enumerate(_QUALITY_PANELS, start=1):
        first_panel = (i == 1)

        ann_s = ann[col].dropna()
        qrt_s = qrt[col].dropna()

        fig.add_trace(
            go.Scatter(
                x=ann_s.index.tolist(),
                y=ann_s.round(4).tolist(),
                name="Annual",
                legendgroup="annual",
                showlegend=first_panel,
                mode="lines+markers",
                line=dict(color="#0d6efd", width=2, dash="solid"),
                hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
            ),
            row=i, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=qrt_s.index.tolist(),
                y=qrt_s.round(4).tolist(),
                name="Quarterly",
                legendgroup="quarterly",
                showlegend=first_panel,
                mode="lines+markers",
                line=dict(color="#0d6efd", width=1.5, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
            ),
            row=i, col=1,
        )
        fig.update_yaxes(
            title_standoff=10,
            gridcolor="#dee2e6",
            row=i, col=1,
        )

    fig.update_layout(
        height=760,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            size=12,
            color="#212529",
        ),
    )
    fig.update_xaxes(showgrid=False)
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build(ticker: str, benchmark: str, output: Path) -> None:
    with closing(get_connection()) as conn:
        uni_row = conn.execute(
            "SELECT company_name, sector, industry FROM universe WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        price_row = conn.execute(
            "SELECT adj_close FROM prices_daily WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        n_annual = conn.execute(
            "SELECT COUNT(*) FROM financials_annual WHERE ticker = ?", (ticker,)
        ).fetchone()[0]
        n_quarterly = conn.execute(
            "SELECT COUNT(*) FROM financials_quarterly WHERE ticker = ?", (ticker,)
        ).fetchone()[0]

    company_name = uni_row[0] if uni_row else ""
    sector = uni_row[1] if uni_row else "—"
    industry = uni_row[2] if uni_row else "—"
    current_price = f"${price_row[0]:,.2f}" if price_row else "—"

    ph = price_history(ticker, benchmark=benchmark)
    df = returns_table(ticker, benchmark=benchmark)

    price_chart = _price_chart_html(ph, ticker, benchmark)
    table_html = _returns_table_html(df, ticker, benchmark)
    returns_chart = _returns_chart_html(df, ticker, benchmark)
    dd_chart = _drawdown_chart_html(ticker)
    val_chart = _valuation_chart_html(ticker)
    quality_chart = _quality_chart_html(ticker)
    snapshot_html = _snapshot_table_html(fundamentals_snapshot(ticker))

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    template = env.get_template("tearsheet.html")
    html = template.render(
        ticker=ticker,
        benchmark=benchmark,
        run_date=date.today().isoformat(),
        company_name=company_name,
        sector=sector,
        industry=industry,
        current_price=current_price,
        n_annual=n_annual,
        n_quarterly=n_quarterly,
        price_chart_html=price_chart,
        returns_table_html=table_html,
        returns_chart_html=returns_chart,
        drawdown_chart_html=dd_chart,
        valuation_chart_html=val_chart,
        quality_chart_html=quality_chart,
        snapshot_table_html=snapshot_html,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker", required=True, metavar="TICKER")
    parser.add_argument("--benchmark", default="SPY", metavar="TICKER")
    parser.add_argument("--output", metavar="PATH", default=None)
    args = parser.parse_args()

    ticker = args.ticker.upper()
    benchmark = args.benchmark.upper()
    output = (
        Path(args.output)
        if args.output
        else REPORTS_DIR / f"{ticker}_{date.today().isoformat()}.html"
    )

    try:
        build(ticker, benchmark, output)
        print(f"Written: {output}  ({output.stat().st_size / 1_000_000:.1f} MB)")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
