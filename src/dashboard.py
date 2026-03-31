"""Interactive Plotly Dash dashboard: Truflation vs Official CPI.

Designed for a Portfolio Manager audience — clean, information-dense,
and focused on actionable insight.

Sections:
    1. Summary stat cards
    2. Main comparison: CPI vs Truflation YoY (2010–present)
    3. Correlation analysis: rolling correlation + scatter plot
    4. Nowcast: walk-forward model vs actual
"""

import pandas as pd
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html
from plotly.subplots import make_subplots

from src.config import settings
from src.utils import load_parquet
from src.nowcast import run_full_analysis


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Colour palette
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COLORS = {
    "cpi":       "#1f77b4",               # Muted blue
    "truflation":"#ff7f0e",               # Orange
    "spread_pos":"rgba(44,160,44,0.7)",   # Green
    "spread_neg":"rgba(214,39,40,0.7)",   # Red
    "recession": "rgba(200,200,200,0.35)",
    "zero_line": "rgba(100,100,100,0.5)",
}

CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Arial, sans-serif", size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# Truflation data starts 2010 — default view shows only the comparison window
TRUF_START = "2010-01-01"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Chart builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_main_chart(aligned_df, recession_df=None):
    """CPI vs Truflation YoY % with spread panel. Defaulted to 2010–present."""
    valid = aligned_df.dropna(subset=["cpi_yoy", "truflation_yoy_monthly"], how="all")
    overlap = aligned_df.dropna(subset=["cpi_yoy", "truflation_yoy_monthly"])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32],
        subplot_titles=("YoY Inflation Rate (%)", "Truflation − CPI Spread (pp)"),
    )

    # Recession shading (both panels)
    if recession_df is not None:
        _add_recession_bands(fig, recession_df, rows=[1, 2])

    # CPI line
    fig.add_trace(go.Scatter(
        x=valid.index, y=valid["cpi_yoy"],
        name="Official CPI (FRED)",
        line=dict(color=COLORS["cpi"], width=2.5),
        hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>CPI</extra>",
    ), row=1, col=1)

    # Truflation line
    fig.add_trace(go.Scatter(
        x=valid.index, y=valid["truflation_yoy_monthly"],
        name="Truflation Index",
        line=dict(color=COLORS["truflation"], width=2, dash="dot"),
        hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>Truflation</extra>",
    ), row=1, col=1)

    # Spread bars — use Scatter with fill for reliable date axis
    spread = overlap["truflation_yoy_monthly"] - overlap["cpi_yoy"]
    colors = [COLORS["spread_pos"] if v >= 0 else COLORS["spread_neg"] for v in spread]
    fig.add_trace(go.Bar(
        x=overlap.index, y=spread,
        name="Spread (Truflation − CPI)",
        marker_color=colors,
        hovertemplate="%{x|%b %Y}: %{y:+.2f}pp<extra>Spread</extra>",
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["zero_line"], row=2, col=1)

    fig.update_layout(
        **CHART_LAYOUT,
        height=600,
        title=dict(text="US Inflation: Official CPI vs Truflation Alternative Data", font=dict(size=15)),
        xaxis=dict(
            type="date",
            range=[TRUF_START, valid.index.max().strftime("%Y-%m-%d")],
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#f0f0f0",
                activecolor="#1f77b4",
            ),
            rangeslider=dict(visible=True, thickness=0.05),
        ),
        xaxis2=dict(type="date"),
        bargap=0.1,
    )
    fig.update_yaxes(title_text="YoY %", row=1, col=1, ticksuffix="%")
    fig.update_yaxes(title_text="pp", row=2, col=1, ticksuffix="pp")

    return fig


def build_correlation_chart(aligned_df, window=12):
    """Rolling 12-month Pearson correlation between Truflation and CPI YoY."""
    valid = aligned_df.dropna(subset=["cpi_yoy", "truflation_yoy_monthly"])
    rolling_corr = (
        valid["cpi_yoy"]
        .rolling(window=window)
        .corr(valid["truflation_yoy_monthly"])
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_corr.index, y=rolling_corr.values,
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.15)",
        line=dict(color=COLORS["cpi"], width=2),
        hovertemplate="%{x|%b %Y}: ρ = %{y:.3f}<extra></extra>",
        name="12-month rolling ρ",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["zero_line"])

    # Annotate the overall correlation
    overall = valid["cpi_yoy"].corr(valid["truflation_yoy_monthly"])
    fig.add_hline(
        y=overall, line_dash="dot", line_color=COLORS["truflation"],
        annotation_text=f"Overall ρ = {overall:.3f}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        **CHART_LAYOUT,
        height=300,
        title=f"Rolling {window}-Month Pearson Correlation — Truflation vs CPI YoY",
        yaxis=dict(title="Pearson ρ", range=[-1, 1]),
        showlegend=False,
    )
    return fig


def build_scatter_chart(aligned_df):
    """Scatter plot of Truflation YoY vs CPI YoY with OLS regression line."""
    valid = aligned_df.dropna(subset=["cpi_yoy", "truflation_yoy_monthly"]).copy()
    valid["year"] = valid.index.year

    fig = px.scatter(
        valid,
        x="truflation_yoy_monthly",
        y="cpi_yoy",
        color="year",
        color_continuous_scale="Blues",
        trendline="ols",
        labels={
            "truflation_yoy_monthly": "Truflation YoY %",
            "cpi_yoy": "Official CPI YoY %",
            "year": "Year",
        },
        hover_data={"year": True},
        template="plotly_white",
    )
    fig.update_traces(
        marker=dict(size=6, opacity=0.7),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        **{k: v for k, v in CHART_LAYOUT.items() if k != "hovermode"},
        hovermode="closest",
        height=380,
        title="Truflation vs CPI YoY — Scatter with OLS Trend",
    )
    return fig


def build_nowcast_chart():
    """Walk-forward nowcast vs actual CPI — loaded from saved CSV."""
    try:
        wf = pd.read_csv(settings.processed_data_dir / "walk_forward_results.csv", index_col="date", parse_dates=True)
    except FileNotFoundError:
        return go.Figure().update_layout(title="Nowcast data not found — run the pipeline first")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wf.index, y=wf["actual_cpi_yoy"],
        name="Actual CPI YoY",
        line=dict(color=COLORS["cpi"], width=2.5),
        hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>Actual CPI</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=wf.index, y=wf["model_prediction"],
        name="Model Nowcast (Truflation-based)",
        line=dict(color=COLORS["truflation"], width=2, dash="dot"),
        hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>Model</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=wf.index, y=wf["baseline_prediction"],
        name="Naive Baseline (persistence)",
        line=dict(color="grey", width=1.5, dash="dash"),
        hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>Baseline</extra>",
    ))

    # Error metrics as annotation
    model_mae  = (wf["actual_cpi_yoy"] - wf["model_prediction"]).abs().mean()
    base_mae   = (wf["actual_cpi_yoy"] - wf["baseline_prediction"]).abs().mean()
    improvement = (1 - model_mae / base_mae) * 100
    fig.add_annotation(
        text=f"Model MAE: {model_mae:.3f}pp  |  Baseline MAE: {base_mae:.3f}pp  |  Improvement: {improvement:+.1f}%",
        xref="paper", yref="paper", x=0.01, y=0.97,
        showarrow=False, align="left",
        font=dict(size=12, color="#555"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#ddd", borderwidth=1,
    )

    fig.update_layout(
        **CHART_LAYOUT,
        height=400,
        title="Walk-Forward Nowcast: Model vs Naive Baseline vs Actual CPI YoY",
        yaxis=dict(title="YoY %", ticksuffix="%"),
    )
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _add_recession_bands(fig, recession_df, rows=None):
    """Add NBER grey shading for each recession period."""
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


def compute_summary_stats(aligned_df) -> dict:
    """Compute headline statistics for the dashboard stat cards."""
    valid = aligned_df.dropna(subset=["cpi_yoy", "truflation_yoy_monthly"])
    if valid.empty:
        return {}
    corr        = valid["cpi_yoy"].corr(valid["truflation_yoy_monthly"])
    mae         = (valid["cpi_yoy"] - valid["truflation_yoy_monthly"]).abs().mean()
    latest_cpi  = valid["cpi_yoy"].iloc[-1]
    latest_truf = valid["truflation_yoy_monthly"].iloc[-1]
    return {
        "correlation":          f"{corr:.3f}",
        "mae":                  f"{mae:.2f}pp",
        "latest_cpi_yoy":       f"{latest_cpi:.2f}%",
        "latest_truflation_yoy":f"{latest_truf:.2f}%",
        "latest_spread":        f"{(latest_truf - latest_cpi):+.2f}pp",
        "n_months":             str(len(valid)),
    }


def _stat_card(title, value, subtitle=""):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.P(title, className="text-muted mb-1", style={"fontSize": "0.78em", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        html.H4(value, className="mb-0 fw-bold"),
        html.P(subtitle, className="text-muted mb-0", style={"fontSize": "0.75em"}) if subtitle else None,
    ]), className="text-center shadow-sm h-100 border-0"))


def _section_header(title, description):
    return dbc.Row(dbc.Col([
        html.H5(title, className="mb-1 mt-4 fw-semibold"),
        html.P(description, className="text-muted mb-2", style={"fontSize": "0.9em"}),
        html.Hr(className="mt-1 mb-3"),
    ]))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  App
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        title="Truflation vs CPI — Algebris Analytics",
    )

    # Load data
    aligned  = load_parquet("aligned_monthly")
    recession = load_parquet("recession_raw", subdir="raw")
    stats    = compute_summary_stats(aligned)

    # Run analysis and build charts
    analysis    = run_full_analysis(aligned)
    commentary  = analysis["commentary"]

    main_fig    = build_main_chart(aligned, recession_df=recession)
    corr_fig    = build_correlation_chart(aligned)
    scatter_fig = build_scatter_chart(aligned)
    nowcast_fig = build_nowcast_chart()

    overlap_months = stats.get("n_months", "—")

    app.layout = dbc.Container([

        # ── Header ────────────────────────────────────────────────
        dbc.Row(dbc.Col([
            html.H2("Truflation vs Official CPI", className="text-center mb-1 mt-3 fw-bold"),
            html.P(
                "Alternative Inflation Data Assessment — Algebris Data Analytics Case Study",
                className="text-center text-muted mb-0",
                style={"fontSize": "0.95em"},
            ),
            html.P(
                "Evaluates whether Truflation's real-time daily inflation index can serve as a "
                "reliable proxy for official CPI when government data releases are delayed.",
                className="text-center text-muted mt-1 mb-3",
                style={"fontSize": "0.85em", "maxWidth": "700px", "margin": "0 auto"},
            ),
        ])),

        html.Hr(className="mb-3"),

        # ── Stat cards ────────────────────────────────────────────
        dbc.Row([
            _stat_card("Overall Correlation",    stats.get("correlation", "—"),    f"Over {overlap_months} months"),
            _stat_card("Mean Abs. Error",        stats.get("mae", "—"),            "Truflation vs CPI"),
            _stat_card("Latest CPI YoY",         stats.get("latest_cpi_yoy", "—"), "Official (FRED)"),
            _stat_card("Latest Truflation YoY",  stats.get("latest_truflation_yoy","—"), "Real-time estimate"),
            _stat_card("Current Spread",         stats.get("latest_spread", "—"),  "Truflation minus CPI"),
        ], className="mb-4 g-3"),

        # ── Section 1: Main comparison ────────────────────────────
        _section_header(
            "1. CPI vs Truflation — YoY Inflation Rate",
            "Both series shown as year-on-year % change. Grey shading = NBER recession periods. "
            "The spread panel (bottom) highlights divergence — positive spread suggests Truflation "
            "is running ahead of official CPI, historically a leading indicator of upside surprises.",
        ),
        dbc.Row(dbc.Col(dcc.Graph(
            id="main-chart",
            figure=main_fig,
            config={"displayModeBar": True, "scrollZoom": False, "modeBarButtonsToRemove": ["lasso2d","select2d"]},
        ))),

        # ── Section 2: Correlation analysis ──────────────────────
        _section_header(
            "2. Correlation Analysis",
            "Left: Rolling 12-month Pearson correlation — how consistently does Truflation track CPI "
            "over time? Dips near zero occurred around COVID (2020) when real-time spending data "
            "diverged from BLS survey methodology. "
            "Right: Scatter plot with OLS trend — each point is one month, colour-coded by year.",
        ),
        dbc.Row([
            dbc.Col(dcc.Graph(id="corr-chart", figure=corr_fig), md=7),
            dbc.Col(dcc.Graph(id="scatter-chart", figure=scatter_fig), md=5),
        ], className="align-items-center"),

        # ── Section 3: Nowcasting ─────────────────────────────────
        _section_header(
            "3. Nowcasting — Can Truflation Predict the Next CPI Print?",
            "Walk-forward expanding-window validation: the model is trained on data up to month t−1 "
            "and predicts CPI at month t. Features: Truflation YoY, CPI lag-1, CPI lag-2, lagged "
            "Truflation−CPI spread. Baseline = naive persistence (predict next CPI = last CPI). "
            "No lookahead bias by construction.",
        ),
        dbc.Row(dbc.Col(dcc.Graph(id="nowcast-chart", figure=nowcast_fig))),

        # ── Section 4: PM Commentary ─────────────────────────────
        _section_header(
            "4. Portfolio Manager Commentary",
            "Written assessment based on the latest available data. "
            "Refreshed each time the pipeline runs.",
        ),
        dbc.Row(dbc.Col(
            dbc.Card(dbc.CardBody(
                dcc.Markdown(
                    commentary,
                    style={"fontSize": "0.92em", "lineHeight": "1.7"},
                )
            ), className="shadow-sm border-0"),
            className="mb-4",
        )),

        # ── Footer ────────────────────────────────────────────────
        html.Hr(className="mt-4"),
        dbc.Row(dbc.Col(html.P(
            "Data sources: FRED (CPIAUCSL, USREC) · Truflation API (truflation.com) · "
            "NBER recession dates. Prepared for Algebris Data Analytics Case Study.",
            className="text-muted text-center mb-3",
            style={"fontSize": "0.8em"},
        ))),

    ], fluid=True, style={"maxWidth": "1400px"})

    return app


def run_dashboard(debug=False, port=None):
    """Launch the dashboard server."""
    import os
    app = create_app()
    app.run(debug=debug, port=int(os.environ.get("PORT", port or 8050)))


if __name__ == "__main__":
    run_dashboard(debug=True)
