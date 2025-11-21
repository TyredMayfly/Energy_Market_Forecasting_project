"""
Data refresh service for ensuring fresh data on every forecast request.

This module provides a centralized mechanism to check data freshness and
trigger updates when data is older than the specified threshold (default: 1 hour).
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pytz

from app.core.config import MARKET_TYPES, WEATHER_CONFIG, settings
from app.core.logging import get_logger
from app.services.data_store import (
    append_market_data,
    append_weather_data,
    load_market_data,
    load_weather_data,
)
from app.services.entsoe_client import EntsoeClient
from app.services.knmi_historical_client import KNMIHistoricalClient
from app.services.meteosource_client import MeteosourceClient
from app.services.settlement_price_service import get_settlement_price_service

logger = get_logger(__name__)


def get_latest_timestamp(df: Optional[pd.DataFrame], timestamp_col: str = "timestamp_utc") -> Optional[datetime]:
    """
    Extract the latest timestamp from a DataFrame.

    Args:
        df: DataFrame with timestamp column or DatetimeIndex
        timestamp_col: Name of timestamp column (if not using index)

    Returns:
        Latest timestamp as timezone-aware datetime, or None if DataFrame is empty/None
    """
    if df is None or df.empty:
        return None

    try:
        # Check if using index
        if isinstance(df.index, pd.DatetimeIndex):
            latest = df.index.max()
        elif timestamp_col in df.columns:
            latest = df[timestamp_col].max()
        else:
            logger.warning(f"No timestamp column '{timestamp_col}' found in DataFrame")
            return None

        # Ensure timezone-aware
        if isinstance(latest, pd.Timestamp):
            if latest.tz is None:
                latest = latest.tz_localize("UTC")
            else:
                latest = latest.tz_convert("UTC")
            return latest.to_pydatetime()
        else:
            # Convert to timezone-aware datetime
            if isinstance(latest, datetime):
                if latest.tzinfo is None:
                    latest = pytz.UTC.localize(latest)
                else:
                    latest = latest.astimezone(pytz.UTC)
            return latest

    except Exception as e:
        logger.error(f"Error extracting latest timestamp: {e}")
        return None


def is_data_fresh(latest_ts: Optional[datetime], now: datetime, threshold_minutes: int) -> bool:
    """
    Check if data is fresh based on the latest timestamp.

    Args:
        latest_ts: Latest timestamp in the dataset
        now: Current time (timezone-aware, UTC)
        threshold_minutes: Freshness threshold in minutes

    Returns:
        True if data is fresh (age <= threshold), False otherwise
    """
    if latest_ts is None:
        return False

    age = now - latest_ts
    is_fresh = age <= timedelta(minutes=threshold_minutes)

    if is_fresh:
        logger.debug(f"Data is fresh (age: {age.total_seconds() / 60:.1f} minutes)")
    else:
        logger.info(f"Data is stale (age: {age.total_seconds() / 60:.1f} minutes, threshold: {threshold_minutes} minutes)")

    return is_fresh


def update_market_data(market_type: str, now: datetime) -> bool:
    """
    Update market data by fetching new records from ENTSO-E.

    Args:
        market_type: Type of market to update
        now: Current time (timezone-aware, UTC)

    Returns:
        True if update was successful, False otherwise
    """
    try:
        # Fetch data for the last 3 days to catch any delayed publications
        end_date = now
        start_date = now - timedelta(days=3)

        logger.info(f"Updating {market_type} market data from {start_date} to {end_date}")

        entsoe_client = EntsoeClient()

        if market_type == "day_ahead":
            df_new = entsoe_client.fetch_day_ahead_prices_2025_nl(start_date, end_date)
        elif market_type == "intraday":
            df_new = entsoe_client.fetch_intraday_prices_2025_nl(start_date, end_date)
        elif market_type in ["imbalance_shortage", "imbalance_surplus", "regulation_state"]:
            # Imbalance data uses settlement price service
            settlement_service = get_settlement_price_service()
            # Try to get latest prices (which will auto-refresh from TenneT API if stale)
            try:
                latest_price = settlement_service.get_latest_price(price_type="shortage", current_time=now)
                if latest_price is not None:
                    logger.info(f"✓ Settlement prices refreshed via TenneT API")
                    return True
            except Exception as e:
                logger.warning(f"Settlement price auto-refresh failed: {e}")

            # Fallback: fetch from ENTSO-E
            df_new = entsoe_client.fetch_imbalance_data_2025_nl(start_date, end_date)
        else:
            logger.warning(f"Unknown market type for update: {market_type}")
            return False

        if df_new is None or df_new.empty:
            logger.warning(f"No new data fetched for {market_type}")
            return False

        # Append new data to existing CSV
        success = append_market_data(df_new, market_type)

        if success:
            logger.info(f"✓ Appended {len(df_new)} new records to {market_type}")
        else:
            logger.error(f"✗ Failed to append data for {market_type}")

        return success

    except ValueError as e:
        logger.error(f"Failed to initialize ENTSO-E client: {e}")
        return False
    except Exception as e:
        logger.error(f"Error updating {market_type} market data: {e}")
        return False


def update_weather_data(now: datetime) -> bool:
    """
    Update weather data by fetching new records.

    Args:
        now: Current time (timezone-aware, UTC)

    Returns:
        True if update was successful, False otherwise
    """
    try:
        logger.info("Updating weather data")

        # Update historical weather from KNMI (last 3 days)
        try:
            knmi_client = KNMIHistoricalClient()
            start_date = now - timedelta(days=3)
            end_date = now

            df_historical = knmi_client.fetch_hourly_data(start_date, end_date)

            if df_historical is not None and not df_historical.empty:
                # Append historical weather
                success = append_weather_data(df_historical)
                if success:
                    logger.info(f"✓ Appended {len(df_historical)} historical weather records")
            else:
                logger.warning("No new historical weather data fetched from KNMI")

        except Exception as e:
            logger.warning(f"Error updating KNMI historical data: {e}")

        # Update weather forecast from Meteosource (next 24 hours)
        try:
            meteosource_client = MeteosourceClient()
            df_forecast = meteosource_client.fetch_forecast_for_2025()

            if df_forecast is not None and not df_forecast.empty:
                # Append forecast weather
                success = append_weather_data(df_forecast)
                if success:
                    logger.info(f"✓ Appended {len(df_forecast)} weather forecast records")
                return True
            else:
                logger.warning("No weather forecast data fetched from Meteosource")
                return False

        except ValueError as e:
            logger.error(f"Failed to initialize Meteosource client: {e}")
            return False
        except Exception as e:
            logger.error(f"Error updating Meteosource forecast: {e}")
            return False

    except Exception as e:
        logger.error(f"Error updating weather data: {e}")
        return False


def ensure_fresh_data(
    now: Optional[datetime] = None,
    freshness_threshold_minutes: int = 60,
    markets_to_check: Optional[list] = None,
) -> None:
    """
    Ensure all required data sources are fresh, updating if necessary.

    This function is designed to be called on every forecast request. It checks
    the age of each dataset and triggers updates for any data older than the
    specified threshold.

    Args:
        now: Current time (timezone-aware, UTC). If None, uses datetime.utcnow()
        freshness_threshold_minutes: Maximum age of data in minutes before refresh
        markets_to_check: List of market types to check. If None, checks all configured markets

    Note:
        This function is idempotent and safe to call frequently. If data is already
        fresh, it returns quickly without making any API calls.
    """
    # Get current time in UTC
    if now is None:
        now = datetime.now(pytz.UTC)
    elif now.tzinfo is None:
        now = pytz.UTC.localize(now)
    else:
        now = now.astimezone(pytz.UTC)

    logger.debug(f"Checking data freshness (threshold: {freshness_threshold_minutes} minutes)")

    # Determine which markets to check
    if markets_to_check is None:
        # Check all configured market types
        markets_to_check = list(MARKET_TYPES.keys())

    # Track what was updated
    updates_performed = []
    updates_failed = []

    # Check and update market data
    for market_type in markets_to_check:
        if market_type not in MARKET_TYPES:
            logger.warning(f"Skipping unknown market type: {market_type}")
            continue

        # Load existing data
        df_market = load_market_data(market_type)
        latest_ts = get_latest_timestamp(df_market, timestamp_col="timestamp_utc")

        # Check freshness
        if not is_data_fresh(latest_ts, now, freshness_threshold_minutes):
            logger.info(f"Refreshing {market_type} market data...")
            success = update_market_data(market_type, now)

            if success:
                updates_performed.append(market_type)
            else:
                updates_failed.append(market_type)
        else:
            logger.debug(f"{market_type} data is fresh, no update needed")

    # Check and update weather data
    df_weather = load_weather_data()
    latest_weather_ts = get_latest_timestamp(df_weather)  # Uses index for weather data

    if not is_data_fresh(latest_weather_ts, now, freshness_threshold_minutes):
        logger.info("Refreshing weather data...")
        success = update_weather_data(now)

        if success:
            updates_performed.append("weather")
        else:
            updates_failed.append("weather")
    else:
        logger.debug("Weather data is fresh, no update needed")

    # Log summary
    if updates_performed:
        logger.info(f"✓ Data refresh completed. Updated: {', '.join(updates_performed)}")
    else:
        logger.debug("All data sources are fresh, no updates performed")

    if updates_failed:
        logger.warning(f"⚠ Some updates failed: {', '.join(updates_failed)}")
