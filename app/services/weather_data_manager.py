"""
Weather data manager with smart update logic.

Handles:
- Loading existing weather data
- Checking if forecast is stale
- Updating only when needed to minimize API calls
- Merging historical data with new forecast
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from app.core.config import settings, WEATHER_CONFIG
from app.core.logging import get_logger
from app.services.meteosource_client import MeteosourceClient

logger = get_logger(__name__)


class WeatherDataManager:
    """Manages weather data with smart caching and updates."""

    # Update threshold: only fetch new forecast if data is older than this
    UPDATE_THRESHOLD_MINUTES = 60

    def __init__(self):
        """Initialize the weather data manager."""
        self.data_file = Path(settings.data_dir) / WEATHER_CONFIG["data_file"]
        self.client = None

    def _get_client(self) -> MeteosourceClient:
        """Lazy load the Meteosource client."""
        if self.client is None:
            self.client = MeteosourceClient()
        return self.client

    def load_weather_data(self) -> pd.DataFrame:
        """
        Load weather data from file.

        Returns:
            DataFrame with weather data, or empty DataFrame if file doesn't exist
        """
        if not self.data_file.exists():
            logger.warning(f"Weather data file not found: {self.data_file}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.data_file, index_col=0, parse_dates=True)
            logger.debug(f"Loaded {len(df)} weather records from {self.data_file}")
            return df
        except Exception as e:
            logger.error(f"Error loading weather data: {e}")
            return pd.DataFrame()

    def check_forecast_coverage(self) -> Tuple[bool, Optional[datetime], Optional[datetime]]:
        """
        Check if the current forecast data has sufficient 24-hour coverage.

        Returns:
            Tuple of (has_coverage, last_fetch_time, forecast_end_time)
            - has_coverage: True if forecast extends at least 24 hours into the future
            - last_fetch_time: When the forecast was last fetched, or None if no data
            - forecast_end_time: When the forecast data ends, or None if no data
        """
        df = self.load_weather_data()

        if df.empty:
            logger.info("No existing weather data found")
            return False, None, None

        # Check if fetch_timestamp column exists
        if "fetch_timestamp" not in df.columns:
            logger.warning("Weather data missing fetch_timestamp - needs update")
            return False, None, None

        try:
            # Get the most recent fetch timestamp - use utc=True to handle mixed timezones
            last_fetch = pd.to_datetime(df["fetch_timestamp"], utc=True, format="ISO8601").max()
            if isinstance(last_fetch, pd.Timestamp):
                last_fetch = last_fetch.to_pydatetime()
            # Ensure timezone-aware
            if last_fetch.tzinfo is None:
                last_fetch = last_fetch.replace(tzinfo=timezone.utc)

            # Get forecast end time (latest timestamp in the data)
            forecast_end = df.index.max()
            if isinstance(forecast_end, pd.Timestamp):
                forecast_end = forecast_end.to_pydatetime()
            # Ensure timezone-aware
            if forecast_end.tzinfo is None:
                forecast_end = forecast_end.replace(tzinfo=timezone.utc)

            # Calculate coverage: how many hours into the future does the forecast extend?
            now = datetime.now(timezone.utc)
            hours_ahead = (forecast_end - now).total_seconds() / 3600
            
            # We need at least 24 hours of coverage
            has_coverage = hours_ahead >= 24.0
            
            age_minutes = (now - last_fetch).total_seconds() / 60

            logger.info(
                f"Forecast coverage: {hours_ahead:.1f} hours ahead "
                f"(required: 24.0 hours, age: {age_minutes:.1f} minutes)"
            )
            logger.info(
                f"Forecast extends from {df.index.min()} to {forecast_end}"
            )

            return has_coverage, last_fetch, forecast_end

        except Exception as e:
            logger.error(f"Error checking forecast coverage: {e}")
            return False, None, None

    def update_weather_data(self, force: bool = False) -> bool:
        """
        Update weather data with latest forecast if needed.
        
        This ensures we always have at least 24 hours of forecast data ahead.

        Args:
            force: If True, update regardless of coverage

        Returns:
            True if data was updated, False otherwise
        """
        # Check if update is needed
        if not force:
            has_coverage, last_fetch, forecast_end = self.check_forecast_coverage()

            if has_coverage:
                logger.info(
                    f"✓ Forecast has sufficient 24h coverage (last update: {last_fetch.strftime('%Y-%m-%d %H:%M:%S') if last_fetch else 'unknown'})"
                )
                logger.info(
                    f"  Forecast extends to: {forecast_end.strftime('%Y-%m-%d %H:%M:%S') if forecast_end else 'unknown'}"
                )
                logger.info("  Skipping update to save API calls")
                return False
            else:
                logger.info("Forecast does not have 24-hour coverage - updating...")

        logger.info("Updating weather forecast...")

        # Load existing data
        df_existing = self.load_weather_data()

        # Fetch new 24-hour forecast
        try:
            client = self._get_client()
            df_forecast = client.fetch_hourly_forecast(place_id="amsterdam")

            # Add metadata
            df_forecast["data_type"] = "forecast"
            df_forecast["fetch_timestamp"] = datetime.now(timezone.utc)

            logger.info(f"✓ Fetched {len(df_forecast)} hours of new forecast data")

        except Exception as e:
            logger.error(f"✗ Failed to fetch forecast: {e}")
            return False

        # Merge strategy:
        # 1. Keep all historical data (before forecast start)
        # 2. Replace forecast period with new forecast

        if not df_existing.empty:
            # Ensure both DataFrames have matching timezone info
            if df_existing.index.tz is None:
                df_existing.index = df_existing.index.tz_localize("UTC")
            if df_forecast.index.tz is None:
                df_forecast.index = df_forecast.index.tz_localize("UTC")
            
            # Find where forecast starts
            forecast_start = df_forecast.index.min()

            # Keep only historical data (before forecast starts)
            df_historical = df_existing[df_existing.index < forecast_start].copy()

            # Mark historical data
            if "data_type" not in df_historical.columns:
                df_historical["data_type"] = "historical"

            # Combine historical + new forecast
            df_combined = pd.concat([df_historical, df_forecast])

            logger.info(f"  Historical records: {len(df_historical)}")
            logger.info(f"  New forecast records: {len(df_forecast)}")
            logger.info(f"  Total records: {len(df_combined)}")
        else:
            # No existing data, just use forecast
            df_combined = df_forecast
            logger.info(f"  Creating new weather data file with {len(df_combined)} records")

        # Save to file
        try:
            # Ensure directory exists
            self.data_file.parent.mkdir(parents=True, exist_ok=True)

            # Ensure index is timezone-aware before sorting
            if not hasattr(df_combined.index, 'tz') or df_combined.index.tz is None:
                if isinstance(df_combined.index, pd.DatetimeIndex):
                    df_combined.index = df_combined.index.tz_localize("UTC")
            
            # Sort by timestamp
            df_combined = df_combined.sort_index()

            # Reset index to save timestamp as column (for compatibility with load_weather_data)
            df_to_save = df_combined.reset_index().rename(columns={'index': 'timestamp'})
            df_to_save.to_csv(self.data_file, index=False)

            logger.info(f"✓ Updated weather data saved to {self.data_file}")
            logger.info(f"  Date range: {df_combined.index.min()} to {df_combined.index.max()}")

            return True

        except Exception as e:
            logger.error(f"✗ Failed to save weather data: {e}")
            return False

    def get_latest_weather_data(self, force_update: bool = False) -> pd.DataFrame:
        """
        Get latest weather data, updating if necessary.

        Args:
            force_update: If True, force update regardless of freshness

        Returns:
            DataFrame with weather data
        """
        # Update if needed
        self.update_weather_data(force=force_update)

        # Load and return
        return self.load_weather_data()

    def get_forecast_summary(self) -> dict:
        """
        Get summary of current forecast data.

        Returns:
            Dictionary with forecast summary information
        """
        df = self.load_weather_data()

        if df.empty:
            return {
                "exists": False,
                "total_records": 0,
                "historical_records": 0,
                "forecast_records": 0,
            }

        # Count record types
        historical_count = (
            len(df[df["data_type"] == "historical"]) if "data_type" in df.columns else 0
        )
        forecast_count = (
            len(df[df["data_type"] == "forecast"]) if "data_type" in df.columns else len(df)
        )

        # Get forecast info
        has_coverage, last_fetch, forecast_end = self.check_forecast_coverage()

        return {
            "exists": True,
            "total_records": len(df),
            "historical_records": historical_count,
            "forecast_records": forecast_count,
            "date_range_start": df.index.min(),
            "date_range_end": df.index.max(),
            "last_fetch": last_fetch,
            "has_24h_coverage": has_coverage,
            "forecast_end": forecast_end,
            "file_path": str(self.data_file),
        }


# Global instance
_weather_manager = None


def get_weather_manager() -> WeatherDataManager:
    """Get the global weather data manager instance."""
    global _weather_manager
    if _weather_manager is None:
        _weather_manager = WeatherDataManager()
    return _weather_manager
