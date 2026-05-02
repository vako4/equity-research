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
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

from equity_research.analytics.ratios import drawdown, price_history, returns_table

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
# Orchestration
# ---------------------------------------------------------------------------

def build(ticker: str, benchmark: str, output: Path) -> None:
    ph = price_history(ticker, benchmark=benchmark)
    df = returns_table(ticker, benchmark=benchmark)

    price_chart = _price_chart_html(ph, ticker, benchmark)
    table_html = _returns_table_html(df, ticker, benchmark)
    returns_chart = _returns_chart_html(df, ticker, benchmark)
    dd_chart = _drawdown_chart_html(ticker)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    template = env.get_template("tearsheet.html")
    html = template.render(
        ticker=ticker,
        benchmark=benchmark,
        run_date=date.today().isoformat(),
        price_chart_html=price_chart,
        returns_table_html=table_html,
        returns_chart_html=returns_chart,
        drawdown_chart_html=dd_chart,
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
