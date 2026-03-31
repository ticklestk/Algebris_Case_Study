"""Data ingestion layer: fetch Truflation and FRED CPI data.

Each data source has:
1. A Pydantic model validating the raw API response schema.
2. A fetch function that returns a clean pd.DataFrame.
3. Retry logic for transient API failures.

Design note: Truflation returns a daily index, FRED CPI is monthly.
Both are returned with DatetimeIndex for downstream alignment.
"""

from datetime import datetime

import pandas as pd
import requests
from fredapi import Fred
from loguru import logger
from pydantic import BaseModel, field_validator

from src.config import settings
from src.utils import retry_on_failure, save_parquet


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pydantic schemas for API response validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TruflationRecord(BaseModel):
    """Single data point from the Truflation API stream."""
    timestamp: int  # Unix timestamp
    value: float    # Index value

    @field_validator("value")
    @classmethod
    def value_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Truflation index value must be positive, got {v}")
        return v

    @property
    def date(self) -> datetime:
        return datetime.utcfromtimestamp(self.timestamp)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Truflation data fetching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@retry_on_failure(exceptions=(requests.RequestException, ConnectionError))
def fetch_truflation() -> pd.DataFrame:
    """Fetch the full Truflation US CPI Inflation Index.

    Returns:
        DataFrame with DatetimeIndex and column 'truflation_index'.
        Frequency: daily.
    """
    logger.info("Fetching Truflation data...")
    url = settings.truflation_full_url
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    raw = response.json()

    # ── Validate & parse ──────────────────────────────────────────
    # The API response structure may vary — adapt parsing here.
    # Common patterns: list of dicts, or nested under a 'data' key.
    records = _parse_truflation_response(raw)

    if not records:
        raise ValueError("Truflation API returned no records")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]  # Deduplicate

    logger.info(
        f"Truflation: {len(df)} daily observations, "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )

    # Persist raw data
    save_parquet(df, "truflation_raw", subdir="raw")
    return df


def _parse_truflation_response(raw: dict | list) -> list[dict]:
    """Parse Truflation API response into a list of {date, truflation_index}.

    Handles both the legacy trufscan.io stream format (Unix timestamps) and
    the current truflation.com REST format (ISO date strings).
    """
    records = []

    # ── New API: {"index": ["2010-01-01", ...], "truflation_us_cpi_frozen_yoy": [...]} ──
    if isinstance(raw, dict) and "index" in raw and "truflation_us_cpi_frozen_yoy" in raw:
        dates = raw["index"]
        values = raw["truflation_us_cpi_frozen_yoy"]
        for date, value in zip(dates, values):
            if value is not None:
                records.append({"date": date, "truflation_index": float(value)})
        return records

    # ── Legacy list format ────────────────────────────────────────────
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                if "date" in item:
                    records.append({
                        "date": item["date"],
                        "truflation_index": float(item.get("value", item.get("v", 0))),
                    })
                else:
                    rec = TruflationRecord(
                        timestamp=item.get("timestamp", item.get("t", 0)),
                        value=item.get("value", item.get("v", 0)),
                    )
                    records.append({"date": rec.date, "truflation_index": rec.value})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                rec = TruflationRecord(timestamp=int(item[0]), value=float(item[1]))
                records.append({"date": rec.date, "truflation_index": rec.value})

    elif isinstance(raw, dict):
        data = raw.get("data", raw.get("records", raw.get("result", [])))
        if isinstance(data, list):
            return _parse_truflation_response(data)
        else:
            logger.warning(f"Unexpected Truflation response structure: {list(raw.keys())}")

    return records


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FRED CPI data fetching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@retry_on_failure(exceptions=(requests.RequestException, ConnectionError, ValueError))
def fetch_cpi_from_fred() -> pd.DataFrame:
    """Fetch the official US CPI-U (All Items, Seasonally Adjusted) from FRED.

    Returns:
        DataFrame with DatetimeIndex and column 'cpi'.
        Frequency: monthly.
    """
    logger.info(f"Fetching FRED series: {settings.fred_cpi_series}")
    fred = Fred(api_key=settings.fred_api_key)

    cpi_series = fred.get_series(settings.fred_cpi_series)
    df = cpi_series.to_frame(name="cpi")
    df.index.name = "date"
    df = df.dropna()

    logger.info(
        f"FRED CPI: {len(df)} monthly observations, "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )

    save_parquet(df, "cpi_fred_raw", subdir="raw")
    return df


@retry_on_failure(exceptions=(requests.RequestException, ConnectionError, ValueError))
def fetch_recession_indicator() -> pd.DataFrame:
    """Fetch NBER recession indicator from FRED (for dashboard shading).

    Returns:
        DataFrame with DatetimeIndex and column 'recession' (0 or 1).
    """
    logger.info(f"Fetching FRED series: {settings.fred_recession_series}")
    fred = Fred(api_key=settings.fred_api_key)

    rec_series = fred.get_series(settings.fred_recession_series)
    df = rec_series.to_frame(name="recession")
    df.index.name = "date"

    logger.info(f"Recession indicator: {len(df)} observations")
    save_parquet(df, "recession_raw", subdir="raw")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Unified ingestion entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ingest_all() -> dict[str, pd.DataFrame]:
    """Run the full data ingestion pipeline.

    Returns:
        Dict with keys 'truflation', 'cpi', 'recession', each a DataFrame.
    """
    settings.ensure_dirs()
    logger.info("=" * 60)
    logger.info("Starting data ingestion pipeline")
    logger.info("=" * 60)

    truflation_df = fetch_truflation()
    cpi_df = fetch_cpi_from_fred()
    recession_df = fetch_recession_indicator()

    logger.info("Data ingestion complete ✓")
    return {
        "truflation": truflation_df,
        "cpi": cpi_df,
        "recession": recession_df,
    }
