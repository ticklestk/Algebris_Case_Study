"""CLI entry point for the Truflation–CPI pipeline.

Usage:
    python run_pipeline.py              # Run once (ad-hoc)
    python run_pipeline.py --schedule   # Run daily at 08:00 UTC
    python run_pipeline.py --dashboard  # Launch the dashboard
"""

import argparse
import sys

from loguru import logger

from src.config import settings
from src.utils import setup_logging


def run_pipeline() -> None:
    """Execute the full pipeline: ingest → transform → analyse."""
    from src.data_ingestion import ingest_all
    from src.transformations import transform_all
    from src.nowcast import run_full_analysis

    setup_logging("INFO")
    settings.ensure_dirs()

    logger.info("Pipeline triggered")

    # Step 1: Ingest
    raw_data = ingest_all()

    # Step 2: Transform
    transformed = transform_all(raw_data)

    # Step 3: Analyse
    results = run_full_analysis(transformed["aligned"])

    # Output PM commentary
    print("\n" + "=" * 70)
    print(results["commentary"])
    print("=" * 70)

    logger.info("Pipeline complete ✓")


def run_scheduled() -> None:
    """Run pipeline on a daily schedule (08:00 UTC)."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    setup_logging("INFO")
    scheduler = BlockingScheduler()
    scheduler.add_job(run_pipeline, "cron", hour=8, minute=0)

    logger.info("Scheduler started — pipeline will run daily at 08:00 UTC")
    logger.info("Press Ctrl+C to stop")
    scheduler.start()


def run_dashboard() -> None:
    """Launch the Plotly Dash dashboard."""
    from src.dashboard import run_dashboard as _run

    setup_logging("INFO")
    logger.info("Launching dashboard on http://localhost:8050")
    _run(debug=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Truflation–CPI pipeline: ingest, transform, analyse",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run pipeline on a daily schedule (08:00 UTC)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Plotly Dash dashboard",
    )

    args = parser.parse_args()

    if args.dashboard:
        run_dashboard()
    elif args.schedule:
        run_scheduled()
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
