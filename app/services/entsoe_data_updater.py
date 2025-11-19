"""
ENTSO-E Data Auto-Updater

Automatically checks and updates day-ahead market data when it becomes stale.
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from app.services.entsoe_client import EntsoeClient
from app.services.data_store import load_market_data, save_market_data
from app.core.logging import get_logger

logger = get_logger(__name__)


def check_and_update_day_ahead_data(max_age_hours: float = 12.0, force: bool = False) -> bool:
    """
    Check if day-ahead data is stale and update if needed.

    Args:
        max_age_hours: Maximum age in hours before data is considered stale
        force: Force update even if data is fresh

    Returns:
        True if data was updated, False if no update was needed or update failed
    """
    # Load existing data
    df_existing = load_market_data("day_ahead")

    if df_existing is None or df_existing.empty:
        logger.warning("No existing day-ahead data found")
        return False

    # Get the latest timestamp in the dataset
    df_existing["timestamp_utc"] = pd.to_datetime(df_existing["timestamp_utc"], utc=True)
    latest_timestamp = df_existing["timestamp_utc"].max()
    current_time = pd.Timestamp.now(tz="UTC")

    # Calculate age of latest data
    data_age_hours = (current_time - latest_timestamp).total_seconds() / 3600

    logger.info(f"Latest day-ahead data: {latest_timestamp} " f"(age: {data_age_hours:.1f} hours)")

    # Check if update is needed
    if not force and data_age_hours < max_age_hours:
        logger.info(f"✓ Data is fresh (age: {data_age_hours:.1f}h < threshold: {max_age_hours}h)")
        logger.info("  Skipping update to save API calls")
        return False

    logger.info(f"⚠ Data is stale (age: {data_age_hours:.1f}h >= threshold: {max_age_hours}h)")
    logger.info("  Fetching latest data from ENTSO-E...")

    # Initialize ENTSO-E client
    try:
        client = EntsoeClient()
    except Exception as e:
        logger.error(f"Failed to initialize ENTSO-E client: {e}")
        return False

    # Fetch data from latest timestamp to now
    # Add a small buffer to ensure we get overlapping data
    fetch_start = latest_timestamp - timedelta(hours=24)
    fetch_end = current_time

    logger.info(f"Fetching data from {fetch_start} to {fetch_end}")

    try:
        df_new = client.fetch_day_ahead_prices_2025_nl(
            fetch_start.to_pydatetime(), fetch_end.to_pydatetime()
        )

        if df_new.empty:
            logger.warning("No new data returned from ENTSO-E API")
            return False

        logger.info(f"✓ Retrieved {len(df_new)} new records")

        # Combine with existing data
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        # Remove duplicates (keep latest values)
        df_combined = (
            df_combined.drop_duplicates(subset=["timestamp_utc"], keep="last")
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )

        # Calculate how many new records were added
        new_records = len(df_combined) - len(df_existing)

        logger.info(f"✓ Added {new_records} new records")
        logger.info(f"  Total records: {len(df_combined)} " f"(was: {len(df_existing)})")
        logger.info(
            f"  New date range: {df_combined['timestamp_utc'].min()} "
            f"to {df_combined['timestamp_utc'].max()}"
        )

        # Save updated data
        result = save_market_data(df_combined, "day_ahead")

        if result:
            logger.info("✓ Day-ahead data updated successfully")
            return True
        else:
            logger.error("✗ Failed to save updated data")
            return False

    except Exception as e:
        logger.error(f"✗ Error updating day-ahead data: {e}")
        return False


def get_data_age_hours(market_type: str = "day_ahead") -> Optional[float]:
    """
    Get the age of the latest data point in hours.

    Args:
        market_type: Type of market data to check

    Returns:
        Age in hours, or None if data doesn't exist
    """
    df = load_market_data(market_type)

    if df is None or df.empty:
        return None

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    latest_timestamp = df["timestamp_utc"].max()
    current_time = pd.Timestamp.now(tz="UTC")

    age_hours = (current_time - latest_timestamp).total_seconds() / 3600
    return age_hours


if __name__ == "__main__":
    # Run as standalone script
    print("\n" + "=" * 80)
    print("🔄 ENTSO-E Day-Ahead Data Auto-Updater")
    print("=" * 80 + "\n")

    # Check current data age
    age = get_data_age_hours("day_ahead")
    if age is None:
        print("✗ No existing data found")
    else:
        print(f"📊 Current data age: {age:.1f} hours\n")

        # Check and update
        updated = check_and_update_day_ahead_data(max_age_hours=12.0)

        if updated:
            print("\n✅ Data updated successfully!")
            new_age = get_data_age_hours("day_ahead")
            print(f"📊 New data age: {new_age:.1f} hours")
        else:
            print("\n✓ No update needed (data is fresh)")

    print("\n" + "=" * 80)
