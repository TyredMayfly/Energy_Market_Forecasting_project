"""
Weather data manager with smart update logic.

Handles:
- Loading existing weather data
- Checking if forecast is stale
- Updating only when needed to minimize API calls
- Merging historical data with new forecast
"""

from datetime import datetime, timedelta
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

    def check_forecast_freshness(self) -> Tuple[bool, Optional[datetime]]:
        """
        Check if the current forecast data is fresh enough.

        Returns:
            Tuple of (is_fresh, last_fetch_time)
            - is_fresh: True if forecast is recent enough (< UPDATE_THRESHOLD_MINUTES old)
            - last_fetch_time: When the forecast was last fetched, or None if no data
        """
        df = self.load_weather_data()

        if df.empty:
            logger.info("No existing weather data found")
            return False, None

        # Check if fetch_timestamp column exists
        if "fetch_timestamp" not in df.columns:
            logger.warning("Weather data missing fetch_timestamp - needs update")
            return False, None

        # Get the most recent fetch timestamp
        try:
            last_fetch = pd.to_datetime(df["fetch_timestamp"]).max()

            # Convert to datetime if it's a timestamp
            if isinstance(last_fetch, pd.Timestamp):
                last_fetch = last_fetch.to_pydatetime()

            # Check age
            age_minutes = (datetime.utcnow() - last_fetch).total_seconds() / 60
            is_fresh = age_minutes < self.UPDATE_THRESHOLD_MINUTES

            logger.info(
                f"Forecast age: {age_minutes:.1f} minutes "
                f"(threshold: {self.UPDATE_THRESHOLD_MINUTES} minutes)"
            )

            return is_fresh, last_fetch

        except Exception as e:
            logger.error(f"Error checking forecast freshness: {e}")
            return False, None

    def update_weather_data(self, force: bool = False) -> bool:
        """
        Update weather data with latest forecast if needed.

        Args:
            force: If True, update regardless of freshness

        Returns:
            True if data was updated, False otherwise
        """
        # Check if update is needed
        if not force:
            is_fresh, last_fetch = self.check_forecast_freshness()

            if is_fresh:
                logger.info(
                    f"✓ Forecast is fresh (last update: {last_fetch.strftime('%Y-%m-%d %H:%M:%S') if last_fetch else 'unknown'})"
                )
                logger.info("  Skipping update to save API calls")
                return False

        logger.info("Updating weather forecast...")

        # Load existing data
        df_existing = self.load_weather_data()

        # Fetch new 24-hour forecast
        try:
            client = self._get_client()
            df_forecast = client.fetch_hourly_forecast(place_id="amsterdam")

            # Add metadata
            df_forecast["data_type"] = "forecast"
            df_forecast["fetch_timestamp"] = datetime.utcnow()

            logger.info(f"✓ Fetched {len(df_forecast)} hours of new forecast data")

        except Exception as e:
            logger.error(f"✗ Failed to fetch forecast: {e}")
            return False

        # Merge strategy:
        # 1. Keep all historical data (before forecast start)
        # 2. Replace forecast period with new forecast

        if not df_existing.empty:
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

            # Sort by timestamp
            df_combined = df_combined.sort_index()

            # Save
            df_combined.to_csv(self.data_file)

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
        is_fresh, last_fetch = self.check_forecast_freshness()

        return {
            "exists": True,
            "total_records": len(df),
            "historical_records": historical_count,
            "forecast_records": forecast_count,
            "date_range_start": df.index.min(),
            "date_range_end": df.index.max(),
            "last_fetch": last_fetch,
            "is_fresh": is_fresh,
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
