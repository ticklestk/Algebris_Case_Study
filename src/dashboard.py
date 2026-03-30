"""Interactive Plotly Dash dashboard: Truflation vs Official CPI.

Designed for a Portfolio Manager audience — clean, information-dense,
and focused on actionable insight.

Features:
    1. Main chart: YoY % inflation — Truflation vs CPI on same scale
    2. Spread chart: Truflation − CPI divergence
    3. Rolling correlation window
    4. Date-range slider for zooming
    5. Shaded NBER recession periods
    6. Summary statistics cards
"""

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots

from src.utils import load_parquet


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Chart builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Finance-appropriate colour palette
COLORS = {
    "cpi": "#1f77b4",           # Muted blue
    "truflation": "#ff7f0e",    # Orange
    "spread": "#2ca02c",        # Green
    "recession": "rgba(200, 200, 200, 0.3)",  # Light grey shading
    "bg": "#fafafa",
    "grid": "#e5e5e5",
}


def build_main_chart(aligned_df, truflation_daily=None, recession_df=None):
    """Build the primary YoY % change comparison chart.

    Args:
        aligned_df: Monthly aligned DataFrame with 'cpi_yoy' and 'truflation_yoy_monthly'.
        truflation_daily: Optional daily Truflation for granular view.
        recession_df: Optional recession indicator for shading.

    Returns:
        Plotly Figure.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("YoY Inflation Rate (%)", "Truflation − CPI Spread (pp)"),
    )

    # ── Recession shading ─────────────────────────────────────────
    if recession_df is not None:
        _add_recession_bands(fig, recession_df, rows=[1, 2])

    # ── Main chart: YoY lines ─────────────────────────────────────
    valid = aligned_df.dropna(subset=["cpi_yoy", "truflation_yoy_monthly"], how="all")

    fig.add_trace(
        go.Scatter(
            x=valid.index,
            y=valid["cpi_yoy"],
            name="Official CPI (BLS/FRED)",
            line=dict(color=COLORS["cpi"], width=2.5),
            hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>CPI</extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=valid.index,
            y=valid["truflation_yoy_monthly"],
            name="Truflation Index",
            line=dict(color=COLORS["truflation"], width=2, dash="dot"),
            hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>Truflation</extra>",
        ),
        row=1, col=1,
    )

    # ── Spread chart ──────────────────────────────────────────────
    spread = valid["truflation_yoy_monthly"] - valid["cpi_yoy"]
    fig.add_trace(
        go.Bar(
            x=valid.index,
            y=spread,
            name="Spread (Truflation − CPI)",
            marker_color=[
                COLORS["spread"] if v >= 0 else "#d62728" for v in spread.values
            ],
            hovertemplate="%{x|%b %Y}: %{y:+.2f}pp<extra>Spread</extra>",
        ),
        row=2, col=1,
    )

    # Zero line on spread
    fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────
    fig.update_layout(
        height=700,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=60),
        hovermode="x unified",
        title=dict(
            text="US Inflation: Official CPI vs Truflation Alternative Data",
            font=dict(size=16),
        ),
    )
    fig.update_yaxes(title_text="YoY %", row=1, col=1)
    fig.update_yaxes(title_text="pp", row=2, col=1)

    return fig


def build_correlation_chart(aligned_df, window: int = 12):
    """Build rolling correlation chart between Truflation and CPI YoY.

    Args:
        aligned_df: Monthly aligned DataFrame.
        window: Rolling window in months.

    Returns:
        Plotly Figure.
    """
    valid = aligned_df.dropna(subset=["cpi_yoy", "truflation_yoy_monthly"])
    rolling_corr = (
        valid["cpi_yoy"]
        .rolling(window=window)
        .corr(valid["truflation_yoy_monthly"])
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr.values,
            fill="tozeroy",
            line=dict(color=COLORS["cpi"], width=2),
            hovertemplate="%{x|%b %Y}: ρ = %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        height=300,
        template="plotly_white",
        title=f"Rolling {window}-Month Correlation",
        yaxis=dict(title="Pearson ρ", range=[-1, 1]),
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def _add_recession_bands(fig, recession_df, rows=None):
    """Add grey shaded rectangles for NBER recession periods."""
    if rows is None:
        rows = [1]

    in_recession = False
    start = None

    for date, row in recession_df.iterrows():
        if row["recession"] == 1 and not in_recession:
            start = date
            in_recession = True
        elif row["recession"] == 0 and in_recession:
            for r in rows:
                fig.add_vrect(
                    x0=start, x1=date,
                    fillcolor=COLORS["recession"],
                    layer="below", line_width=0,
                    row=r, col=1,
                )
            in_recession = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Summary statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_summary_stats(aligned_df) -> dict:
    """Compute headline statistics for the dashboard cards."""
    valid = aligned_df.dropna(subset=["cpi_yoy", "truflation_yoy_monthly"])
    if valid.empty:
        return {}

    corr = valid["cpi_yoy"].corr(valid["truflation_yoy_monthly"])
    mae = (valid["cpi_yoy"] - valid["truflation_yoy_monthly"]).abs().mean()
    latest_cpi = valid["cpi_yoy"].dropna().iloc[-1] if not valid["cpi_yoy"].dropna().empty else None
    latest_truf = valid["truflation_yoy_monthly"].dropna().iloc[-1]

    return {
        "correlation": f"{corr:.3f}",
        "mae": f"{mae:.2f}pp",
        "latest_cpi_yoy": f"{latest_cpi:.2f}%" if latest_cpi else "N/A",
        "latest_truflation_yoy": f"{latest_truf:.2f}%",
        "latest_spread": f"{(latest_truf - latest_cpi):+.2f}pp" if latest_cpi else "N/A",
        "n_months": str(len(valid)),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dash app layout
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        title="Truflation vs CPI — Algebris Analytics",
    )

    # Load data
    aligned = load_parquet("aligned_monthly")
    recession = load_parquet("recession_raw", subdir="raw")
    stats = compute_summary_stats(aligned)

    # Build charts
    main_fig = build_main_chart(aligned, recession_df=recession)
    corr_fig = build_correlation_chart(aligned)

    # ── Layout ────────────────────────────────────────────────────
    app.layout = dbc.Container([
        # Header
        dbc.Row(
            dbc.Col(html.H2(
                "Truflation vs Official CPI — Alternative Inflation Data Assessment",
                className="text-center my-3",
            )),
        ),

        # Summary cards
        dbc.Row([
            _stat_card("Overall Correlation", stats.get("correlation", "—")),
            _stat_card("Mean Abs. Error", stats.get("mae", "—")),
            _stat_card("Latest CPI YoY", stats.get("latest_cpi_yoy", "—")),
            _stat_card("Latest Truflation YoY", stats.get("latest_truflation_yoy", "—")),
            _stat_card("Current Spread", stats.get("latest_spread", "—")),
        ], className="mb-3"),

        # Main chart with date range slider
        dbc.Row(dbc.Col(dcc.Graph(
            id="main-chart",
            figure=main_fig,
            config={"displayModeBar": True, "scrollZoom": True},
        ))),

        # Correlation chart
        dbc.Row(dbc.Col(dcc.Graph(
            id="corr-chart",
            figure=corr_fig,
        )), className="mt-3"),

        # Footer
        dbc.Row(dbc.Col(html.P(
            "Data sources: FRED (CPI-U, USREC), Truflation API. "
            "Prepared for Algebris Data Analytics Case Study.",
            className="text-muted text-center mt-4 mb-2",
            style={"fontSize": "0.85em"},
        ))),

    ], fluid=True)

    return app


def _stat_card(title: str, value: str) -> dbc.Col:
    """Create a summary statistic card."""
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.P(title, className="text-muted mb-1", style={"fontSize": "0.8em"}),
                html.H5(value, className="mb-0"),
            ]),
            className="text-center shadow-sm",
        ),
        width=True,  # Equal-width columns
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_dashboard(debug: bool = False, port: int = 8050):
    """Launch the dashboard server."""
    app = create_app()
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    run_dashboard(debug=True)
