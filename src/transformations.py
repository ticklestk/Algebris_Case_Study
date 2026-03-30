"""Data transformations: cleaning, alignment, and feature engineering.

Core responsibility: take raw daily Truflation and monthly CPI data,
compute year-on-year % changes on a consistent basis, and produce
an aligned dataset ready for dashboard display and nowcasting.

Key challenge: Truflation is daily, CPI is monthly. We handle this by:
1. Computing YoY % change on each series at its native frequency.
2. Resampling daily Truflation to month-end for direct comparison.
3. Keeping the daily series available for the dashboard's granular view.
"""

import pandas as pd
from loguru import logger

from src.utils import save_parquet


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Year-on-Year % Change
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_yoy_pct_change(
    df: pd.DataFrame,
    value_col: str,
    periods: int = 12,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Compute year-on-year % change for a time series.

    Args:
        df: DataFrame with DatetimeIndex.
        value_col: Column containing the index/level values.
        periods: Lag for % change (12 for monthly, 365 for daily).
        output_col: Name for the result column (defaults to '{value_col}_yoy').

    Returns:
        DataFrame with the original column plus a new YoY % change column.
    """
    out_col = output_col or f"{value_col}_yoy"
    result = df.copy()
    result[out_col] = result[value_col].pct_change(periods=periods) * 100
    logger.debug(f"Computed YoY% for '{value_col}' (periods={periods}) → '{out_col}'")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Truflation transformations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def transform_truflation(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and compute YoY % change for daily Truflation data.

    Steps:
        1. Remove any duplicate dates (keep last).
        2. Fill small gaps (weekends/holidays) via forward-fill (max 5 days).
        3. Compute daily YoY % change (365-day lag).

    Returns:
        DataFrame with columns: 'truflation_index', 'truflation_yoy'.
    """
    result = df.copy()

    # Forward-fill small gaps (Truflation may skip weekends)
    result = result.asfreq("D").ffill(limit=5)

    # The API already returns YoY % values — no further computation needed
    result["truflation_yoy"] = result["truflation_index"]

    logger.info(f"Truflation transformed: {result['truflation_yoy'].notna().sum()} YoY observations")
    return result


def resample_truflation_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily Truflation to month-end for CPI comparison.

    Uses the mean of daily values within each month, which smooths out
    day-to-day noise and makes it more comparable to the monthly CPI.

    Returns:
        Monthly DataFrame with columns: 'truflation_index_monthly', 'truflation_yoy_monthly'.
    """
    monthly = df[["truflation_index"]].resample("ME").mean()
    monthly = monthly.rename(columns={"truflation_index": "truflation_index_monthly"})

    # API already returns YoY % values — monthly mean IS the monthly YoY
    monthly["truflation_yoy_monthly"] = monthly["truflation_index_monthly"]

    logger.info(f"Truflation monthly: {len(monthly)} months")
    return monthly


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CPI transformations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def transform_cpi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute YoY % change for official monthly CPI.

    Returns:
        DataFrame with columns: 'cpi', 'cpi_yoy'.
    """
    result = compute_yoy_pct_change(
        df,
        value_col="cpi",
        periods=12,
        output_col="cpi_yoy",
    )
    logger.info(f"CPI transformed: {result['cpi_yoy'].notna().sum()} YoY observations")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Alignment: merge Truflation (monthly) + CPI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def align_monthly(
    truflation_monthly: pd.DataFrame,
    cpi_df: pd.DataFrame,
    recession_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge monthly Truflation and CPI on their date index.

    Performs an outer join so we can see where one series has data
    and the other doesn't (useful for the dashboard gap analysis).

    Returns:
        Merged DataFrame with all YoY columns aligned by month.
    """
    # Normalize Truflation month-end index to month-start to align with FRED dates
    truf = truflation_monthly.copy()
    truf.index = truf.index.to_period("M").to_timestamp()
    merged = truf.join(cpi_df, how="outer")

    if recession_df is not None:
        # Resample recession indicator to monthly (take max — any 1 in month = recession)
        rec_monthly = recession_df.resample("ME").max()
        merged = merged.join(rec_monthly, how="left")
        merged["recession"] = merged["recession"].fillna(0).astype(int)

    logger.info(f"Aligned monthly dataset: {len(merged)} rows, columns={list(merged.columns)}")
    save_parquet(merged, "aligned_monthly")
    return merged


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full transformation pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def transform_all(
    raw_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Run all transformations on raw ingested data.

    Args:
        raw_data: Dict from `data_ingestion.ingest_all()`.

    Returns:
        Dict with transformed DataFrames:
          - 'truflation_daily': daily with YoY
          - 'truflation_monthly': month-end resampled with YoY
          - 'cpi': monthly with YoY
          - 'aligned': merged monthly dataset
          - 'recession': recession indicator
    """
    logger.info("=" * 60)
    logger.info("Starting data transformation pipeline")
    logger.info("=" * 60)

    truflation_daily = transform_truflation(raw_data["truflation"])
    truflation_monthly = resample_truflation_monthly(raw_data["truflation"])
    cpi = transform_cpi(raw_data["cpi"])
    recession = raw_data.get("recession")

    aligned = align_monthly(truflation_monthly, cpi, recession)

    # Persist daily truflation for the dashboard's granular view
    save_parquet(truflation_daily, "truflation_daily")
    save_parquet(cpi, "cpi_transformed")

    logger.info("Data transformation complete ✓")
    return {
        "truflation_daily": truflation_daily,
        "truflation_monthly": truflation_monthly,
        "cpi": cpi,
        "aligned": aligned,
        "recession": recession,
    }
