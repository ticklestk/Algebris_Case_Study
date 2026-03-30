"""Nowcasting & analysis: can Truflation predict the next CPI print?

This module performs:
1. Statistical analysis (correlation, lead-lag, Granger causality)
2. A simple nowcasting model with walk-forward validation
3. PM-ready commentary generation

Design principles:
- No lookahead bias: strict expanding-window time-series CV
- Baseline comparison: every model is benchmarked vs naive persistence
- Explainability: coefficients, error metrics, and confidence intervals
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import grangercausalitytests

from src.utils import save_csv


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data classes for structured results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CorrelationResult:
    """Results from correlation and lead-lag analysis."""
    contemporaneous_corr: float
    best_lead_months: int
    best_lead_corr: float
    lead_lag_series: pd.Series  # Correlation at each lag


@dataclass
class GrangerResult:
    """Results from Granger causality test."""
    optimal_lag: int
    f_statistic: float
    p_value: float
    is_significant: bool  # At 5% level


@dataclass
class NowcastResult:
    """Results from the nowcasting model."""
    model_mae: float
    model_rmse: float
    baseline_mae: float          # Naive persistence baseline
    baseline_rmse: float
    mae_improvement_pct: float   # % improvement over baseline
    latest_nowcast: float        # Nowcast for next CPI print
    latest_nowcast_date: str     # Which CPI month this predicts
    model_coefficients: dict     # For explainability
    walk_forward_results: pd.DataFrame  # Full CV results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. Correlation & Lead-Lag Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyse_correlation(aligned_df: pd.DataFrame, max_lead: int = 6) -> CorrelationResult:
    """Compute contemporaneous and lead-lag correlations.

    Tests whether Truflation at time t correlates with CPI at time t+k
    for k = 0, 1, ..., max_lead months.

    Args:
        aligned_df: Monthly aligned DataFrame.
        max_lead: Maximum number of months to test Truflation leading CPI.

    Returns:
        CorrelationResult with the best lead and full lead-lag series.
    """
    valid = aligned_df.dropna(subset=["cpi_yoy", "truflation_yoy_monthly"])
    cpi = valid["cpi_yoy"]
    truf = valid["truflation_yoy_monthly"]

    # Contemporaneous
    contemp_corr = cpi.corr(truf)
    logger.info(f"Contemporaneous correlation: {contemp_corr:.4f}")

    # Lead-lag: does Truflation at t predict CPI at t+k?
    lead_corrs = {}
    for k in range(0, max_lead + 1):
        # Shift Truflation back by k months (i.e., Truflation leads CPI by k months)
        shifted = truf.shift(k)
        overlap = pd.concat([cpi, shifted], axis=1).dropna()
        if len(overlap) > 10:
            lead_corrs[k] = overlap.iloc[:, 0].corr(overlap.iloc[:, 1])

    lead_series = pd.Series(lead_corrs, name="correlation")
    lead_series.index.name = "lead_months"

    best_lead = max(lead_corrs, key=lead_corrs.get)
    best_corr = lead_corrs[best_lead]
    logger.info(f"Best lead: {best_lead} months (ρ = {best_corr:.4f})")

    return CorrelationResult(
        contemporaneous_corr=contemp_corr,
        best_lead_months=best_lead,
        best_lead_corr=best_corr,
        lead_lag_series=lead_series,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. Granger Causality Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_granger_causality(
    aligned_df: pd.DataFrame,
    max_lag: int = 6,
) -> GrangerResult:
    """Test whether Truflation Granger-causes CPI YoY.

    Granger causality tests whether past values of Truflation help
    predict CPI beyond what past CPI alone can explain.

    Args:
        aligned_df: Monthly aligned DataFrame.
        max_lag: Maximum lag order to test.

    Returns:
        GrangerResult with the most significant lag.
    """
    valid = aligned_df[["cpi_yoy", "truflation_yoy_monthly"]].dropna()

    if len(valid) < max_lag + 10:
        logger.warning("Insufficient data for Granger causality test")
        return GrangerResult(0, 0.0, 1.0, False)

    # grangercausalitytests expects [effect, cause] column order
    test_data = valid[["cpi_yoy", "truflation_yoy_monthly"]]

    try:
        results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
    except Exception as e:
        logger.error(f"Granger causality test failed: {e}")
        return GrangerResult(0, 0.0, 1.0, False)

    # Find the lag with the lowest p-value (F-test)
    best_lag = None
    best_p = 1.0
    best_f = 0.0

    for lag, result in results.items():
        f_test = result[0]["ssr_ftest"]
        p_val = f_test[1]
        f_stat = f_test[0]
        if p_val < best_p:
            best_p = p_val
            best_f = f_stat
            best_lag = lag

    is_sig = best_p < 0.05
    logger.info(
        f"Granger causality: lag={best_lag}, F={best_f:.2f}, "
        f"p={best_p:.4f} ({'significant' if is_sig else 'not significant'})"
    )

    return GrangerResult(
        optimal_lag=best_lag,
        f_statistic=best_f,
        p_value=best_p,
        is_significant=is_sig,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. Walk-Forward Nowcasting Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_nowcast_model(
    aligned_df: pd.DataFrame,
    min_train_months: int = 24,
) -> NowcastResult:
    """Build a nowcasting model using walk-forward expanding-window CV.

    Model: CPI_YoY(t) = β₀ + β₁ · Truflation_YoY(t) + β₂ · CPI_YoY(t-1) + ε

    Using CPI_YoY(t-1) as a feature captures the autoregressive component.
    The question is whether Truflation adds predictive power on top of that.

    Baseline: naive persistence — CPI_YoY(t) = CPI_YoY(t-1).

    Walk-forward validation:
        - Train on months [0, ..., t-1], predict month t
        - Expand window and repeat
        - No lookahead bias by construction

    Args:
        aligned_df: Monthly aligned DataFrame.
        min_train_months: Minimum months before first prediction.

    Returns:
        NowcastResult with metrics, latest nowcast, and full CV results.
    """
    valid = aligned_df[["cpi_yoy", "truflation_yoy_monthly"]].dropna()

    # Feature engineering (no lookahead: only past values)
    features_df = pd.DataFrame(index=valid.index)
    features_df["truflation_yoy"] = valid["truflation_yoy_monthly"]
    features_df["cpi_yoy_lag1"] = valid["cpi_yoy"].shift(1)  # Previous month CPI
    features_df["cpi_yoy_lag2"] = valid["cpi_yoy"].shift(2)  # 2-month lag
    features_df["truf_cpi_spread_lag1"] = (
        valid["truflation_yoy_monthly"].shift(1) - valid["cpi_yoy"].shift(1)
    )
    features_df["target"] = valid["cpi_yoy"]

    features_df = features_df.dropna()

    if len(features_df) < min_train_months + 6:
        logger.warning("Insufficient data for walk-forward validation")
        raise ValueError("Not enough overlapping data for nowcasting")

    feature_cols = ["truflation_yoy", "cpi_yoy_lag1", "cpi_yoy_lag2", "truf_cpi_spread_lag1"]

    # ── Walk-forward expanding-window CV ──────────────────────────
    predictions = []
    actuals = []
    dates = []
    baseline_preds = []  # Naive persistence: CPI(t-1)

    for t in range(min_train_months, len(features_df)):
        train = features_df.iloc[:t]
        test_row = features_df.iloc[t]

        X_train = train[feature_cols].values
        y_train = train["target"].values
        X_test = test_row[feature_cols].values.reshape(1, -1)

        model = LinearRegression()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]
        actual = test_row["target"]
        baseline = test_row["cpi_yoy_lag1"]  # Naive persistence

        predictions.append(pred)
        actuals.append(actual)
        dates.append(features_df.index[t])
        baseline_preds.append(baseline)

    # ── Metrics ───────────────────────────────────────────────────
    preds_arr = np.array(predictions)
    actuals_arr = np.array(actuals)
    baseline_arr = np.array(baseline_preds)

    model_mae = mean_absolute_error(actuals_arr, preds_arr)
    model_rmse = np.sqrt(mean_squared_error(actuals_arr, preds_arr))
    baseline_mae = mean_absolute_error(actuals_arr, baseline_arr)
    baseline_rmse = np.sqrt(mean_squared_error(actuals_arr, baseline_arr))
    improvement = (1 - model_mae / baseline_mae) * 100

    logger.info(f"Model MAE: {model_mae:.4f}pp | Baseline MAE: {baseline_mae:.4f}pp")
    logger.info(f"Improvement over baseline: {improvement:+.1f}%")

    # ── Latest nowcast ────────────────────────────────────────────
    # Fit on ALL available data, predict next month
    X_all = features_df[feature_cols].values
    y_all = features_df["target"].values
    final_model = LinearRegression()
    final_model.fit(X_all, y_all)

    # Latest available features for next-month prediction
    latest_truf = valid["truflation_yoy_monthly"].iloc[-1]
    latest_cpi = valid["cpi_yoy"].iloc[-1]
    prev_cpi = valid["cpi_yoy"].iloc[-2]
    latest_spread = valid["truflation_yoy_monthly"].iloc[-1] - valid["cpi_yoy"].iloc[-1]

    next_features = np.array([[latest_truf, latest_cpi, prev_cpi, latest_spread]])
    nowcast = final_model.predict(next_features)[0]

    # Approximate next CPI publication month
    last_date = valid.index[-1]
    next_month = last_date + pd.DateOffset(months=1)

    coefficients = dict(zip(
        ["intercept"] + feature_cols,
        [final_model.intercept_] + list(final_model.coef_),
    ))

    logger.info(f"Nowcast for {next_month.strftime('%b %Y')}: {nowcast:.2f}% YoY")
    logger.info(f"Model coefficients: {coefficients}")

    # ── Walk-forward results DataFrame ────────────────────────────
    wf_results = pd.DataFrame({
        "date": dates,
        "actual_cpi_yoy": actuals,
        "model_prediction": predictions,
        "baseline_prediction": baseline_preds,
        "model_error": preds_arr - actuals_arr,
        "baseline_error": baseline_arr - actuals_arr,
    }).set_index("date")

    save_csv(wf_results, "walk_forward_results")

    return NowcastResult(
        model_mae=model_mae,
        model_rmse=model_rmse,
        baseline_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        mae_improvement_pct=improvement,
        latest_nowcast=nowcast,
        latest_nowcast_date=next_month.strftime("%B %Y"),
        model_coefficients=coefficients,
        walk_forward_results=wf_results,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. PM-Ready Commentary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_commentary(
    corr_result: CorrelationResult,
    granger_result: GrangerResult,
    nowcast_result: NowcastResult,
    aligned_df: pd.DataFrame,
) -> str:
    """Generate a brief written commentary for the Portfolio Manager.

    Style: concise, actionable, hedge-fund-ready. No fluff.
    """
    valid = aligned_df.dropna(subset=["cpi_yoy", "truflation_yoy_monthly"])
    latest_truf = valid["truflation_yoy_monthly"].iloc[-1]
    latest_cpi = valid["cpi_yoy"].iloc[-1]
    latest_date = valid.index[-1].strftime("%B %Y")

    commentary = f"""
## Truflation as CPI Proxy — Assessment Summary

### Key Finding
Truflation's daily inflation index shows a **{corr_result.contemporaneous_corr:.2f} contemporaneous
correlation** with official CPI YoY, with the strongest signal at a **{corr_result.best_lead_months}-month
lead** (ρ = {corr_result.best_lead_corr:.3f}).

### Granger Causality
{"Truflation **Granger-causes** CPI at the " + str(granger_result.optimal_lag) + "-month lag "
 "(F = " + f"{granger_result.f_statistic:.2f}" + ", p = " + f"{granger_result.p_value:.4f}" + "), "
 "confirming it contains predictive information beyond CPI's own history."
 if granger_result.is_significant else
 "Granger causality test was **not significant** at the 5% level "
 "(p = " + f"{granger_result.p_value:.4f}" + "), suggesting Truflation may not add "
 "incremental predictive power beyond CPI's autoregressive behaviour."}

### Nowcast
Based on the latest Truflation reading, the model nowcasts **{nowcast_result.latest_nowcast_date}
CPI YoY at {nowcast_result.latest_nowcast:.2f}%**, compared to the last official print of
{latest_cpi:.2f}% ({latest_date}).

- Model MAE: **{nowcast_result.model_mae:.3f}pp** vs baseline (naive persistence): {nowcast_result.baseline_mae:.3f}pp
- Improvement over baseline: **{nowcast_result.mae_improvement_pct:+.1f}%**

### Current Signal
As of the latest data, Truflation reads **{latest_truf:.2f}% YoY** vs CPI at **{latest_cpi:.2f}%**,
a spread of **{latest_truf - latest_cpi:+.2f}pp**. {"This suggests upside risk to the next CPI print."
if latest_truf > latest_cpi else "This suggests downside risk to the next CPI print."}

### Caveats
1. Truflation's methodology differs from BLS (real-time spending vs survey-based basket weights).
2. The correlation may weaken during structural breaks or policy regime changes.
3. Truflation has a shorter track record — robustness improves as more data accumulates.
4. The linear model is deliberately simple for interpretability; nonlinear approaches may capture
   additional signal but at the cost of explainability.

### Recommendation
Truflation is a **useful directional indicator** during data blackout periods (e.g. government
shutdowns). It should be used as one input among several, not as a direct CPI substitute.
The spread between Truflation and CPI can serve as an early warning for inflation surprises.
"""
    return commentary.strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Unified analysis entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_full_analysis(aligned_df: pd.DataFrame) -> dict:
    """Run the complete analysis pipeline.

    Returns:
        Dict with 'correlation', 'granger', 'nowcast', and 'commentary'.
    """
    logger.info("=" * 60)
    logger.info("Starting analysis pipeline")
    logger.info("=" * 60)

    corr = analyse_correlation(aligned_df)
    granger = test_granger_causality(aligned_df)
    nowcast = build_nowcast_model(aligned_df)
    commentary = generate_commentary(corr, granger, nowcast, aligned_df)

    logger.info("Analysis complete ✓")
    return {
        "correlation": corr,
        "granger": granger,
        "nowcast": nowcast,
        "commentary": commentary,
    }
