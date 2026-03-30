"""Centralised configuration for the Truflation–CPI pipeline.

Uses pydantic-settings for type-safe config with .env file support.
API endpoints and FRED series IDs are defined here so they're easy
to update if upstream schemas change.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Fred API key
    fred_api_key: str = "your_fred_api_key_here"

    # ── Truflation API ────────────────────────────────────────────────
    truflation_api_url: str = (
        "https://truflation.com/api/index-data/us-inflation-rate"
    )

    # ── FRED Series ───────────────────────────────────────────────────
    # CPI-U: All Items, Seasonally Adjusted (monthly)
    fred_cpi_series: str = "CPIAUCSL"
    # NBER recession indicator (for dashboard shading)
    fred_recession_series: str = "USREC"

    # ── File Paths ────────────────────────────────────────────────────
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")

    # ── Pipeline Defaults ─────────────────────────────────────────────
    # How many retries on transient API failures
    api_max_retries: int = 3
    api_retry_delay_secs: float = 2.0

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def truflation_full_url(self) -> str:
        """Return the Truflation API URL."""
        return self.truflation_api_url


# Singleton instance — import this everywhere
settings = Settings()
