"""
Data update service for initializing and maintaining market and weather data.

This module handles:
- Initial download of all 2025 data
- Daily updates to fetch the most recent data
- Scheduling of automatic updates
"""

import argparse
from datetime import datetime, timedelta

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from app.core.config import MARKET_TYPES, settings
from app.core.logging import get_logger
from app.services.data_store import (
    append_market_data,
    append_weather_data,
    get_data_summary,
    save_market_data,
    save_weather_data,
)
from app.services.entsoe_client import EntsoeClient
from app.services.meteosource_client import MeteosourceClient

logger = get_logger(__name__)


def initialize_historical_data_2025() -> bool:
    """
    Initialize historical data for 2025 from the beginning of the year until today.

    Downloads all available data from ENTSO-E and Meteosource for 2025.
    Note: Meteosource free tier provides 24-hour forecasts only.

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("Starting historical data initialization for 2025")
    logger.info("=" * 80)

    start_date = datetime(2025, 1, 1, 0, 0, 0)
    end_date = datetime.utcnow()

    success = True

    # Initialize ENTSO-E client
    try:
        entsoe_client = EntsoeClient()
    except ValueError as e:
        logger.error(f"Failed to initialize ENTSO-E client: {e}")
        return False

    # Fetch day-ahead prices
    logger.info("\n--- Fetching Day-Ahead Market Data ---")
    try:
        df_day_ahead = entsoe_client.fetch_day_ahead_prices_2025_nl(start_date, end_date)
        if not df_day_ahead.empty:
            save_market_data(df_day_ahead, "day_ahead")
            logger.info(f"✓ Day-ahead: {len(df_day_ahead)} records")
        else:
            logger.warning("✗ No day-ahead data fetched")
            success = False
    except Exception as e:
        logger.error(f"✗ Error fetching day-ahead data: {e}")
        success = False

    # Fetch imbalance data (note: using unified CSV data from imbalance_data_loader)
    logger.info("\n--- Fetching Imbalance Market Data ---")
    try:
        df_imbalance = entsoe_client.fetch_imbalance_data_2025_nl(start_date, end_date)
        if not df_imbalance.empty:
            save_market_data(df_imbalance, "imbalance")
            logger.info(f"✓ Imbalance: {len(df_imbalance)} records")
        else:
            logger.warning("✗ No imbalance data fetched")
            # Not critical, continue
    except Exception as e:
        logger.error(f"✗ Error fetching imbalance data: {e}")
        # Not critical, continue

    # Fetch Meteosource weather forecast
    logger.info("\n--- Fetching Meteosource Weather Forecast ---")
    try:
        meteosource_client = MeteosourceClient()
        df_weather = meteosource_client.fetch_forecast_for_2025()
        if not df_weather.empty:
            save_weather_data(df_weather)
            logger.info(f"✓ Weather: {len(df_weather)} records (24-hour forecast)")
        else:
            logger.warning("✗ No weather data fetched")
            # Not critical for testing, but important for production
    except ValueError as e:
        logger.error(f"✗ Failed to initialize Meteosource client: {e}")
    except Exception as e:
        logger.error(f"✗ Error fetching weather data: {e}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Data Initialization Summary")
    logger.info("=" * 80)
    summary = get_data_summary()

    for market_type, info in summary["markets"].items():
        if info["records"] > 0:
            logger.info(
                f"{market_type:15} : {info['records']:6} records "
                f"({info['start_date'][:10]} to {info['end_date'][:10]})"
            )
        else:
            logger.info(f"{market_type:15} : No data")

    if summary["weather"]["records"] > 0:
        info = summary["weather"]
        logger.info(
            f"{'weather':15} : {info['records']:6} records "
            f"({info['start_date'][:10]} to {info['end_date'][:10]})"
        )
    else:
        logger.info(f"{'weather':15} : No data")

    logger.info("=" * 80)

    return success


def update_latest_data() -> bool:
    """
    Update local data with the most recent available data.

    Fetches data for the last 2 days to ensure we catch any delayed publications.

    Returns:
        True if successful, False otherwise
    """
    logger.info("Starting daily data update")

    # Fetch data for last 2 days to handle any delays
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=2)

    success = True

    # Update ENTSO-E data
    try:
        entsoe_client = EntsoeClient()

        # Update day-ahead
        logger.info("Updating day-ahead data")
        df_day_ahead = entsoe_client.fetch_day_ahead_prices_2025_nl(start_date, end_date)
        if not df_day_ahead.empty:
            append_market_data(df_day_ahead, "day_ahead")
            logger.info(f"✓ Appended {len(df_day_ahead)} day-ahead records")
        else:
            logger.warning("No new day-ahead data")

        # Update intraday
        logger.info("Updating intraday data")
        df_intraday = entsoe_client.fetch_intraday_prices_2025_nl(start_date, end_date)
        if not df_intraday.empty:
            append_market_data(df_intraday, "intraday")
            logger.info(f"✓ Appended {len(df_intraday)} intraday records")

        # Update imbalance
        logger.info("Updating imbalance data")
        df_imbalance = entsoe_client.fetch_imbalance_data_2025_nl(start_date, end_date)
        if not df_imbalance.empty:
            append_market_data(df_imbalance, "imbalance")
            logger.info(f"✓ Appended {len(df_imbalance)} imbalance records")

    except ValueError as e:
        logger.error(f"Failed to initialize ENTSO-E client: {e}")
        success = False
    except Exception as e:
        logger.error(f"Error updating ENTSO-E data: {e}")
        success = False

    # Update Meteosource weather forecast (24 hours)
    try:
        meteosource_client = MeteosourceClient()
        df_weather = meteosource_client.fetch_forecast_for_2025()
        if not df_weather.empty:
            # Replace weather data with latest 24-hour forecast
            save_weather_data(df_weather)
            logger.info(f"✓ Updated weather forecast: {len(df_weather)} records")
    except ValueError as e:
        logger.error(f"Failed to initialize Meteosource client: {e}")
    except Exception as e:
        logger.error(f"Error updating weather data: {e}")

    logger.info("Daily data update completed")
    return success


# Global scheduler instance
_scheduler: BackgroundScheduler = None


def start_scheduler():
    """
    Start the background scheduler for daily data updates.

    Schedules update_latest_data() to run daily at configured time.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        logger.warning("Scheduler already running")
        return

    # Create scheduler with Amsterdam timezone
    tz = pytz.timezone(settings.timezone)
    _scheduler = BackgroundScheduler(timezone=tz)

    # Schedule daily update
    _scheduler.add_job(
        update_latest_data,
        trigger="cron",
        hour=settings.daily_update_hour,
        minute=settings.daily_update_minute,
        id="daily_data_update",
        name="Daily market and weather data update",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        f"Scheduler started - daily updates at "
        f"{settings.daily_update_hour:02d}:{settings.daily_update_minute:02d} {settings.timezone}"
    )


def stop_scheduler():
    """Stop the background scheduler."""
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown()
        logger.info("Scheduler stopped")


# CLI entry point
def main():
    """Command-line interface for data update service."""
    parser = argparse.ArgumentParser(description="Market Forecasting Data Update Service")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize all historical 2025 data",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update with latest data",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show data summary",
    )

    args = parser.parse_args()

    if args.init:
        initialize_historical_data_2025()
    elif args.update:
        update_latest_data()
    elif args.summary:
        summary = get_data_summary()
        print("\n=== Data Summary ===\n")
        for market_type, info in summary["markets"].items():
            if info["records"] > 0:
                print(
                    f"{market_type:15} : {info['records']:6} records "
                    f"({info['start_date'][:10]} to {info['end_date'][:10]})"
                )
            else:
                print(f"{market_type:15} : No data")

        if summary["weather"]["records"] > 0:
            info = summary["weather"]
            print(
                f"{'weather':15} : {info['records']:6} records "
                f"({info['start_date'][:10]} to {info['end_date'][:10]})"
            )
        else:
            print(f"{'weather':15} : No data")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
