"""
Meteosource Weather API client.

This module provides a client for fetching weather data from the
Meteosource Weather API.

Fetches hourly weather forecasts including temperature, wind speed, and
solar radiation (irradiance) for energy forecasting purposes.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class MeteosourceClient:
    """Client for interacting with the Meteosource Weather API."""

    def __init__(self, api_key: Optional[str] = None, location: Optional[str] = None):
        """
        Initialize the Meteosource client.

        Args:
            api_key: Meteosource API key. If None, uses settings.meteosource_api_key
            location: Location identifier (place_id, city name, or coordinates).
                     If None, uses settings.meteosource_location
        """
        self.api_key = api_key if api_key is not None else settings.meteosource_api_key
        self.base_url = settings.meteosource_base_url
        self.location = location if location is not None else settings.meteosource_location

        if not self.api_key:
            raise ValueError(
                "Meteosource API key is required. Set METEOSOURCE_API_KEY in .env file."
            )

    def _make_request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """
        Make a request to the Meteosource API.

        Args:
            endpoint: API endpoint path (e.g., 'point')
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        url = f"{self.base_url}/{endpoint}"

        if params is None:
            params = {}

        params["key"] = self.api_key

        logger.debug(f"Making request to {url} with params: {params}")

        response = requests.get(url, params=params)
        response.raise_for_status()

        return response.json()

    def fetch_hourly_forecast(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        place_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch 24-hour hourly weather forecast.

        Args:
            lat: Latitude (overrides location if provided)
            lon: Longitude (must be provided with lat)
            place_id: Place identifier (overrides location if provided)

        Returns:
            DataFrame with hourly weather data containing:
                - timestamp: DateTime index
                - temperature_deg_c: Temperature in Celsius
                - wind_speed_m_per_s: Wind speed in m/s
                - global_radiation_w_per_m2: Solar irradiance in W/m²
                - cloud_cover_pct: Cloud cover percentage
                - precipitation_mm: Precipitation in mm

            Note: Free tier does not include humidity_pct or pressure_hpa

        Raises:
            ValueError: If location parameters are invalid
            requests.exceptions.RequestException: If the API request fails
        """
        params = {
            "sections": "hourly",
            "timezone": "UTC",
            "language": "en",
            "units": "metric",
        }

        # Determine location
        if lat is not None and lon is not None:
            params["lat"] = str(lat)
            params["lon"] = str(lon)
            location_str = f"coordinates ({lat}, {lon})"
        elif place_id is not None:
            params["place_id"] = place_id
            location_str = f"place_id '{place_id}'"
        else:
            params["place_id"] = self.location
            location_str = f"default location '{self.location}'"

        logger.info(f"Fetching 24-hour hourly forecast for {location_str}")

        try:
            data = self._make_request("point", params)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch weather data: {e}")
            raise

        # Extract hourly data
        hourly_data = data.get("hourly", {}).get("data", [])

        if not hourly_data:
            logger.warning("No hourly forecast data received")
            return pd.DataFrame()

        logger.info(f"Received {len(hourly_data)} hourly forecast records")

        # Convert to DataFrame
        records = []
        for hour in hourly_data:
            # Parse the timestamp
            timestamp = pd.to_datetime(hour["date"])

            # Extract weather variables
            record = {
                "timestamp": timestamp,
                "temperature_deg_c": hour.get("temperature"),
                "wind_speed_m_per_s": hour.get("wind", {}).get("speed"),
                "global_radiation_w_per_m2": hour.get("irradiance", 0),  # W/m² (solar radiation)
                "cloud_cover_pct": (
                    hour.get("cloud_cover", {}).get("total")
                    if isinstance(hour.get("cloud_cover"), dict)
                    else hour.get("cloud_cover")
                ),
                "precipitation_mm": hour.get("precipitation", {}).get("total", 0),
            }
            records.append(record)

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        logger.info(
            f"Processed hourly forecast: {len(df)} records from " f"{df.index[0]} to {df.index[-1]}"
        )

        return df

    def fetch_current_weather(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        place_id: Optional[str] = None,
    ) -> dict:
        """
        Fetch current weather conditions.

        Args:
            lat: Latitude (overrides location if provided)
            lon: Longitude (must be provided with lat)
            place_id: Place identifier (overrides location if provided)

        Returns:
            Dictionary with current weather data

        Raises:
            ValueError: If location parameters are invalid
            requests.exceptions.RequestException: If the API request fails
        """
        params = {
            "sections": "current",
            "timezone": "UTC",
            "language": "en",
            "units": "metric",
        }

        # Determine location
        if lat is not None and lon is not None:
            params["lat"] = str(lat)
            params["lon"] = str(lon)
        elif place_id is not None:
            params["place_id"] = place_id
        else:
            params["place_id"] = self.location

        logger.info("Fetching current weather")

        try:
            data = self._make_request("point", params)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch current weather: {e}")
            raise

        current = data.get("current", {})

        return {
            "timestamp": datetime.utcnow(),
            "temperature_deg_c": current.get("temperature"),
            "wind_speed_m_per_s": current.get("wind", {}).get("speed"),
            "cloud_cover_pct": current.get("cloud_cover"),
            "precipitation_mm": current.get("precipitation", {}).get("total", 0),
        }

    def fetch_forecast_for_2025(self) -> pd.DataFrame:
        """
        Fetch hourly weather forecast data for 2025.

        Note: This method fetches the next 24 hours only, as that's what
        the free tier provides. For historical data or longer forecasts,
        consider upgrading to a paid tier.

        Returns:
            DataFrame with 24 hours of hourly weather data

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        logger.info("Fetching 24-hour weather forecast")

        df = self.fetch_hourly_forecast()

        if df.empty:
            logger.warning("No forecast data available")
            return df

        logger.info(f"Fetched {len(df)} hours of forecast data")

        return df
