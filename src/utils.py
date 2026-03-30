"""Shared utilities: logging setup, retry logic, and I/O helpers.

Provides consistent logging across the pipeline and a reusable retry
decorator for transient API failures.
"""

import functools
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from loguru import logger

from src.config import settings


# ── Logging ───────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    """Configure loguru with a clean format for pipeline runs."""
    logger.remove()  # Remove default handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "{message}"
        ),
        level=level,
    )


# ── Retry Decorator ──────────────────────────────────────────────────

def retry_on_failure(
    max_retries: int | None = None,
    delay: float | None = None,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Retry a function on transient failures with exponential backoff.

    Args:
        max_retries: Number of retries (defaults to config value).
        delay: Base delay in seconds between retries.
        exceptions: Tuple of exception types to catch.

    Returns:
        Decorated function with retry logic.
    """
    _max = max_retries or settings.api_max_retries
    _delay = delay or settings.api_retry_delay_secs

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc = None
            for attempt in range(1, _max + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    wait = _delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{_max} failed: {exc}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
            raise RuntimeError(
                f"{func.__name__} failed after {_max} retries"
            ) from last_exc
        return wrapper
    return decorator


# ── I/O Helpers ───────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, filename: str, subdir: str = "processed") -> Path:
    """Save a DataFrame to parquet in the appropriate data directory.

    Args:
        df: DataFrame to save.
        filename: Output filename (without extension).
        subdir: 'raw' or 'processed'.

    Returns:
        Path to the saved file.
    """
    settings.ensure_dirs()
    base_dir = settings.raw_data_dir if subdir == "raw" else settings.processed_data_dir
    path = base_dir / f"{filename}.parquet"
    df.to_parquet(path, index=True)
    logger.info(f"Saved {len(df)} rows → {path}")
    return path


def load_parquet(filename: str, subdir: str = "processed") -> pd.DataFrame:
    """Load a parquet file from the data directory.

    Args:
        filename: Filename (without extension).
        subdir: 'raw' or 'processed'.

    Returns:
        Loaded DataFrame.
    """
    base_dir = settings.raw_data_dir if subdir == "raw" else settings.processed_data_dir
    path = base_dir / f"{filename}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data at {path}. Run the pipeline first.")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows ← {path}")
    return df


def save_csv(df: pd.DataFrame, filename: str, subdir: str = "processed") -> Path:
    """Save to CSV (for human-readable inspection / notebook display)."""
    settings.ensure_dirs()
    base_dir = settings.raw_data_dir if subdir == "raw" else settings.processed_data_dir
    path = base_dir / f"{filename}.csv"
    df.to_csv(path)
    logger.info(f"Saved {len(df)} rows → {path}")
    return path
