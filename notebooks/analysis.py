# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Algebris Data Analytics Case Study
# ## Truflation as a CPI Proxy: Pipeline, Dashboard & Nowcasting
#
# **Objective**: Evaluate whether Truflation's high-frequency alternative inflation
# index can serve as a reliable proxy for the official US CPI during data blackout
# periods (e.g. government shutdowns).
#
# **Structure**:
# 1. Data Engineering — Ingest Truflation + FRED CPI via API
# 2. Data Visualisation — Interactive Plotly Dash dashboard
# 3. Data Analysis — Correlation, Granger causality, nowcasting model

# %% [markdown]
# ---
# ## 1. Data Engineering
#
# The pipeline fetches data from two sources:
# - **Truflation API**: Daily real-time inflation index
# - **FRED API**: Monthly CPI-U (All Items, Seasonally Adjusted)
#
# Design principles:
# - Pydantic validation on API responses (catch schema changes early)
# - Retry logic with exponential backoff for transient failures
# - Modular: each stage is independently runnable and testable

# %%
# Setup
import sys
sys.path.insert(0, "..")

from src.config import settings
from src.utils import setup_logging

setup_logging("INFO")
settings.ensure_dirs()

# %%
# 1a. Ingest data from APIs
from src.data_ingestion import ingest_all

raw_data = ingest_all()

print(f"Truflation: {len(raw_data['truflation'])} daily observations")
print(f"CPI:        {len(raw_data['cpi'])} monthly observations")
print(f"Recession:  {len(raw_data['recession'])} observations")

# %%
# 1b. Quick look at raw data
raw_data["truflation"].head(10)

# %%
raw_data["cpi"].tail(10)

# %% [markdown]
# ---
# ## 1c. Transform: YoY % Change & Alignment
#
# Key decisions:
# - Truflation (daily) → YoY via 365-day lag at native frequency
# - Truflation → resampled to monthly (mean) for CPI comparison
# - CPI (monthly) → YoY via 12-month lag
# - Outer join preserves dates where only one series has data

# %%
from src.transformations import transform_all

transformed = transform_all(raw_data)

aligned = transformed["aligned"]
print(f"Aligned monthly dataset: {len(aligned)} rows")
aligned.tail(10)

# %% [markdown]
# ---
# ## 2. Data Visualisation
#
# The interactive dashboard (see `src/dashboard.py`) can be launched via:
# ```bash
# python run_pipeline.py --dashboard
# ```
#
# Below we show the key charts inline for this notebook submission.

# %%
import plotly.io as pio
from src.dashboard import build_main_chart, build_correlation_chart, compute_summary_stats

pio.renderers.default = "notebook"

# Load recession data for shading
recession = raw_data["recession"]

# Summary statistics
stats = compute_summary_stats(aligned)
print("Summary Statistics:")
for k, v in stats.items():
    print(f"  {k}: {v}")

# %%
# Main chart: CPI vs Truflation YoY with recession shading
fig = build_main_chart(aligned, recession_df=recession)
fig.show()

# %%
# Rolling correlation
corr_fig = build_correlation_chart(aligned, window=12)
corr_fig.show()

# %% [markdown]
# ---
# ## 3. Data Analysis & Nowcasting
#
# ### Approach
# 1. **Correlation analysis**: Contemporaneous + lead-lag (does Truflation lead CPI?)
# 2. **Granger causality**: Formal test — does past Truflation help predict CPI
#    beyond what past CPI alone explains?
# 3. **Nowcasting model**: Linear regression with walk-forward expanding-window CV
#    - Features: current Truflation YoY, CPI lag-1, CPI lag-2, spread lag-1
#    - Baseline: naive persistence (CPI(t) = CPI(t-1))
#    - No lookahead bias by construction

# %%
from src.nowcast import run_full_analysis

results = run_full_analysis(aligned)

# %%
# Lead-lag correlation profile
import plotly.express as px

lead_lag = results["correlation"].lead_lag_series
fig = px.bar(
    x=lead_lag.index,
    y=lead_lag.values,
    labels={"x": "Truflation Lead (months)", "y": "Correlation with CPI YoY"},
    title="Lead-Lag Correlation: Truflation → CPI",
)
fig.show()

# %%
# Walk-forward nowcast results
wf = results["nowcast"].walk_forward_results

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=wf.index, y=wf["actual_cpi_yoy"], name="Actual CPI YoY", line=dict(width=2)))
fig.add_trace(go.Scatter(x=wf.index, y=wf["model_prediction"], name="Model Nowcast", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=wf.index, y=wf["baseline_prediction"], name="Naive Baseline", line=dict(dash="dash", color="grey")))
fig.update_layout(
    title="Walk-Forward Nowcast vs Actual CPI YoY",
    yaxis_title="YoY %",
    template="plotly_white",
    height=400,
)
fig.show()

# %%
# Error metrics comparison
print(f"Model MAE:    {results['nowcast'].model_mae:.4f} pp")
print(f"Baseline MAE: {results['nowcast'].baseline_mae:.4f} pp")
print(f"Improvement:  {results['nowcast'].mae_improvement_pct:+.1f}%")
print(f"Model RMSE:   {results['nowcast'].model_rmse:.4f} pp")
print(f"Baseline RMSE:{results['nowcast'].baseline_rmse:.4f} pp")

# %%
# Model coefficients (explainability)
print("\nModel Coefficients:")
for feature, coef in results["nowcast"].model_coefficients.items():
    print(f"  {feature}: {coef:.4f}")

# %% [markdown]
# ---
# ## PM Commentary

# %%
print(results["commentary"])

# %% [markdown]
# ---
# ## AI Tools Disclosure
#
# Claude (Anthropic) was used to:
# 1. Scaffold the project structure and boilerplate (config, retry logic, CLI)
# 2. Draft docstrings and type annotations
# 3. Review the walk-forward validation logic for lookahead bias
#
# All analytical decisions (feature selection, model choice, commentary framing)
# were made by the candidate. The code was reviewed, tested, and adapted
# to the specific Truflation API response format by the candidate.
