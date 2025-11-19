"""
Tests for Meteosource client.
"""

from unittest.mock import Mock, patch
from datetime import datetime

import pandas as pd
import pytest
import requests
from app.services.meteosource_client import MeteosourceClient


class TestMeteosourceClientInitialization:
    """Test suite for MeteosourceClient initialization."""

    def test_client_initialization_with_env_key(self, mock_env_vars):
        """Test that client initializes with API key from environment."""
        client = MeteosourceClient()
        
        assert client.api_key == "test_meteosource_key"
        assert client.base_url is not None
        assert client.location is not None

    def test_client_initialization_with_explicit_key(self, mock_env_vars):
        """Test initialization with explicitly provided API key."""
        client = MeteosourceClient(api_key="explicit_key", location="london")
        
        assert client.api_key == "explicit_key"
        assert client.location == "london"

    def test_client_initialization_without_key(self):
        """Test that client raises error without API key."""
        with pytest.raises(ValueError, match="API key is required"):
            MeteosourceClient(api_key="")


class TestMeteosourceHourlyForecast:
    """Test suite for Meteosource hourly forecast fetching."""

    @patch("app.services.meteosource_client.requests.get")
    def test_fetch_hourly_forecast_success(self, mock_get, mock_env_vars):
        """Test successful hourly forecast retrieval."""
        client = MeteosourceClient()
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "lat": "52.37403N",
            "lon": "4.88969E",
            "elevation": 13,
            "timezone": "UTC",
            "hourly": {
                "data": [
                    {
                        "date": "2025-11-19T12:00:00",
                        "temperature": 6.5,
                        "wind": {"speed": 3.2, "dir": "NE", "angle": 45},
                        "irradiance": 150.5,
                        "cloud_cover": {"total": 75},
                        "precipitation": {"total": 0.3},
                        "humidity": 85,
                        "pressure": 1013,
                    },
                    {
                        "date": "2025-11-19T13:00:00",
                        "temperature": 7.0,
                        "wind": {"speed": 4.1, "dir": "NE", "angle": 50},
                        "irradiance": 200.0,
                        "cloud_cover": {"total": 60},
                        "precipitation": {"total": 0.0},
                        "humidity": 80,
                        "pressure": 1014,
                    },
                ]
            },
        }
        mock_get.return_value = mock_response
        
        # Fetch data
        df = client.fetch_hourly_forecast(place_id="amsterdam")
        
        # Verify request was made correctly
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "place_id" in call_args[1]["params"]
        assert call_args[1]["params"]["place_id"] == "amsterdam"
        assert call_args[1]["params"]["sections"] == "hourly"
        
        # Verify DataFrame
        assert len(df) == 2
        assert "temperature_deg_c" in df.columns
        assert "wind_speed_m_per_s" in df.columns
        assert "global_radiation_w_per_m2" in df.columns
        assert df["temperature_deg_c"].iloc[0] == 6.5
        assert df["wind_speed_m_per_s"].iloc[0] == 3.2
        assert df["global_radiation_w_per_m2"].iloc[0] == 150.5

    @patch("app.services.meteosource_client.requests.get")
    def test_fetch_hourly_forecast_with_coordinates(self, mock_get, mock_env_vars):
        """Test fetching forecast with lat/lon coordinates."""
        client = MeteosourceClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "lat": "52.1N",
            "lon": "5.18E",
            "hourly": {"data": []},
        }
        mock_get.return_value = mock_response
        
        # Fetch with coordinates
        df = client.fetch_hourly_forecast(lat=52.1, lon=5.18)
        
        # Verify lat/lon were used
        call_args = mock_get.call_args
        assert call_args[1]["params"]["lat"] == "52.1"
        assert call_args[1]["params"]["lon"] == "5.18"
        assert "place_id" not in call_args[1]["params"]

    @patch("app.services.meteosource_client.requests.get")
    def test_fetch_hourly_forecast_api_error(self, mock_get, mock_env_vars):
        """Test handling of API errors."""
        client = MeteosourceClient()
        
        # Mock API error
        mock_get.side_effect = requests.exceptions.HTTPError("API Error")
        
        # Should raise the exception
        with pytest.raises(requests.exceptions.HTTPError):
            client.fetch_hourly_forecast()

    @patch("app.services.meteosource_client.requests.get")
    def test_fetch_hourly_forecast_empty_response(self, mock_get, mock_env_vars):
        """Test handling of empty response."""
        client = MeteosourceClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"hourly": {"data": []}}
        mock_get.return_value = mock_response
        
        df = client.fetch_hourly_forecast()
        
        assert df.empty


class TestMeteosourceCurrentWeather:
    """Test suite for current weather fetching."""

    @patch("app.services.meteosource_client.requests.get")
    def test_fetch_current_weather_success(self, mock_get, mock_env_vars):
        """Test successful current weather retrieval."""
        client = MeteosourceClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "current": {
                "temperature": 8.5,
                "wind": {"speed": 5.2, "dir": "NNE"},
                "cloud_cover": 100,
                "precipitation": {"total": 0.6},
            }
        }
        mock_get.return_value = mock_response
        
        current = client.fetch_current_weather()
        
        assert current["temperature_deg_c"] == 8.5
        assert current["wind_speed_m_per_s"] == 5.2
        assert current["cloud_cover_pct"] == 100
        assert current["precipitation_mm"] == 0.6


class TestMeteosourceForecastFor2025:
    """Test suite for 2025 forecast fetching."""

    @patch("app.services.meteosource_client.requests.get")
    def test_fetch_forecast_for_2025(self, mock_get, mock_env_vars):
        """Test fetching 24-hour forecast for 2025."""
        client = MeteosourceClient()
        
        # Create 24 hours of mock data
        hourly_data = []
        for hour in range(24):
            hourly_data.append({
                "date": f"2025-11-19T{hour:02d}:00:00",
                "temperature": 5.0 + hour * 0.5,
                "wind": {"speed": 3.0 + hour * 0.1},
                "irradiance": 100 if 6 <= hour <= 18 else 0,
                "cloud_cover": {"total": 50},
                "precipitation": {"total": 0},
                "humidity": 80,
                "pressure": 1013,
            })
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "hourly": {"data": hourly_data}
        }
        mock_get.return_value = mock_response
        
        df = client.fetch_forecast_for_2025()
        
        assert len(df) == 24
        assert all(col in df.columns for col in [
            "temperature_deg_c",
            "wind_speed_m_per_s",
            "global_radiation_w_per_m2"
        ])
