"""Tests for data transformation functions.

Validates YoY calculations and alignment logic using synthetic data,
ensuring correctness independent of API availability.
"""

import numpy as np
import pandas as pd
import pytest

from src.transformations import (
    align_monthly,
    compute_yoy_pct_change,
    resample_truflation_monthly,
    transform_cpi,
)


@pytest.fixture
def synthetic_monthly_cpi() -> pd.DataFrame:
    """Create synthetic monthly CPI data (24 months, ~2% annual inflation)."""
    dates = pd.date_range("2022-01-01", periods=24, freq="MS")
    # Start at 300, grow at ~0.17% per month ≈ 2% per year
    values = 300 * (1.0017 ** np.arange(24))
    return pd.DataFrame({"cpi": values}, index=dates)


@pytest.fixture
def synthetic_daily_truflation() -> pd.DataFrame:
    """Create synthetic daily Truflation index (2 years)."""
    dates = pd.date_range("2022-01-01", periods=730, freq="D")
    # Similar growth rate with some daily noise
    base = 150 * (1 + 0.02 / 365) ** np.arange(730)
    noise = np.random.default_rng(42).normal(0, 0.1, 730)
    return pd.DataFrame({"truflation_index": base + noise}, index=dates)


class TestYoYCalculation:
    """Tests for compute_yoy_pct_change."""

    def test_monthly_yoy_returns_correct_values(self, synthetic_monthly_cpi):
        result = compute_yoy_pct_change(synthetic_monthly_cpi, "cpi", periods=12)
        assert "cpi_yoy" in result.columns
        # First 12 months should be NaN (no year-ago comparison)
        assert result["cpi_yoy"].iloc[:12].isna().all()
        # Remaining should be ~2% (our synthetic growth rate)
        valid = result["cpi_yoy"].dropna()
        assert all(1.5 < v < 2.5 for v in valid), f"YoY values outside expected range: {valid.values}"

    def test_custom_output_column_name(self, synthetic_monthly_cpi):
        result = compute_yoy_pct_change(
            synthetic_monthly_cpi, "cpi", periods=12, output_col="my_yoy"
        )
        assert "my_yoy" in result.columns


class TestCPITransformation:
    """Tests for transform_cpi."""

    def test_adds_yoy_column(self, synthetic_monthly_cpi):
        result = transform_cpi(synthetic_monthly_cpi)
        assert "cpi_yoy" in result.columns
        assert "cpi" in result.columns

    def test_preserves_original_values(self, synthetic_monthly_cpi):
        result = transform_cpi(synthetic_monthly_cpi)
        pd.testing.assert_series_equal(result["cpi"], synthetic_monthly_cpi["cpi"])


class TestTruflationResampling:
    """Tests for resample_truflation_monthly."""

    def test_resamples_to_monthly(self, synthetic_daily_truflation):
        result = resample_truflation_monthly(synthetic_daily_truflation)
        # Should have ~24 months from 730 days
        assert 23 <= len(result) <= 25
        assert "truflation_index_monthly" in result.columns
        assert "truflation_yoy_monthly" in result.columns


class TestAlignment:
    """Tests for align_monthly."""

    def test_outer_join_preserves_all_dates(self):
        dates_a = pd.date_range("2022-01-31", periods=6, freq="ME")
        dates_b = pd.date_range("2022-03-31", periods=6, freq="ME")

        df_a = pd.DataFrame({"truflation_index_monthly": range(6), "truflation_yoy_monthly": range(6)}, index=dates_a)
        df_b = pd.DataFrame({"cpi": range(6), "cpi_yoy": range(6)}, index=dates_b)

        result = align_monthly(df_a, df_b)
        # Outer join: should cover Jan 2022 through Aug 2022
        assert len(result) >= max(len(df_a), len(df_b))
