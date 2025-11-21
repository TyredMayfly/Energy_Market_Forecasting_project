"""
Tests for data refresh service.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
import pandas as pd
import pytz

from app.services.data_refresh_service import (
    ensure_fresh_data,
    get_latest_timestamp,
    is_data_fresh,
    update_market_data,
    update_weather_data,
)


@pytest.fixture
def mock_now():
    """Fixture for current time in UTC."""
    return datetime(2025, 11, 20, 12, 0, 0, tzinfo=pytz.UTC)


@pytest.fixture
def fresh_timestamp(mock_now):
    """Fixture for a fresh timestamp (30 minutes ago)."""
    return mock_now - timedelta(minutes=30)


@pytest.fixture
def stale_timestamp(mock_now):
    """Fixture for a stale timestamp (2 hours ago)."""
    return mock_now - timedelta(hours=2)


@pytest.fixture
def mock_market_data_fresh(fresh_timestamp):
    """Fixture for fresh market data."""
    timestamps = pd.date_range(fresh_timestamp - timedelta(days=1), fresh_timestamp, freq='15min', tz='UTC')
    return pd.DataFrame({
        'timestamp_utc': timestamps,
        'price_eur_per_mwh': [45.0 + i for i in range(len(timestamps))],
    })


@pytest.fixture
def mock_market_data_stale(stale_timestamp):
    """Fixture for stale market data."""
    timestamps = pd.date_range(stale_timestamp - timedelta(days=1), stale_timestamp, freq='15min', tz='UTC')
    return pd.DataFrame({
        'timestamp_utc': timestamps,
        'price_eur_per_mwh': [45.0 + i for i in range(len(timestamps))],
    })


@pytest.fixture
def mock_weather_data_fresh(fresh_timestamp):
    """Fixture for fresh weather data with timestamp as index."""
    timestamps = pd.date_range(fresh_timestamp - timedelta(hours=24), fresh_timestamp, freq='h', tz='UTC')
    df = pd.DataFrame({
        'temperature_deg_c': [15.0 + i * 0.1 for i in range(len(timestamps))],
        'wind_speed_m_per_s': [5.0] * len(timestamps),
    }, index=timestamps)
    df.index.name = 'timestamp'
    return df


@pytest.fixture
def mock_weather_data_stale(stale_timestamp):
    """Fixture for stale weather data with timestamp as index."""
    timestamps = pd.date_range(stale_timestamp - timedelta(hours=24), stale_timestamp, freq='h', tz='UTC')
    df = pd.DataFrame({
        'temperature_deg_c': [15.0 + i * 0.1 for i in range(len(timestamps))],
        'wind_speed_m_per_s': [5.0] * len(timestamps),
    }, index=timestamps)
    df.index.name = 'timestamp'
    return df


class TestGetLatestTimestamp:
    """Tests for get_latest_timestamp function."""

    def test_with_column_timestamp(self, mock_market_data_fresh, fresh_timestamp):
        """Test extracting latest timestamp from column."""
        latest = get_latest_timestamp(mock_market_data_fresh, timestamp_col='timestamp_utc')
        assert latest is not None
        assert latest == fresh_timestamp
        assert latest.tzinfo is not None  # Timezone-aware

    def test_with_index_timestamp(self, mock_weather_data_fresh, fresh_timestamp):
        """Test extracting latest timestamp from DatetimeIndex."""
        latest = get_latest_timestamp(mock_weather_data_fresh)
        assert latest is not None
        assert latest == fresh_timestamp
        assert latest.tzinfo is not None  # Timezone-aware

    def test_with_empty_dataframe(self):
        """Test with empty DataFrame returns None."""
        df = pd.DataFrame()
        latest = get_latest_timestamp(df)
        assert latest is None

    def test_with_none_dataframe(self):
        """Test with None DataFrame returns None."""
        latest = get_latest_timestamp(None)
        assert latest is None


class TestIsDataFresh:
    """Tests for is_data_fresh function."""

    def test_fresh_data(self, mock_now, fresh_timestamp):
        """Test that recent data is considered fresh."""
        is_fresh = is_data_fresh(fresh_timestamp, mock_now, threshold_minutes=60)
        assert is_fresh is True

    def test_stale_data(self, mock_now, stale_timestamp):
        """Test that old data is considered stale."""
        is_fresh = is_data_fresh(stale_timestamp, mock_now, threshold_minutes=60)
        assert is_fresh is False

    def test_boundary_condition(self, mock_now):
        """Test exactly at threshold boundary."""
        exactly_60_min_ago = mock_now - timedelta(minutes=60)
        is_fresh = is_data_fresh(exactly_60_min_ago, mock_now, threshold_minutes=60)
        assert is_fresh is True

    def test_none_timestamp(self, mock_now):
        """Test that None timestamp is considered not fresh."""
        is_fresh = is_data_fresh(None, mock_now, threshold_minutes=60)
        assert is_fresh is False


class TestUpdateMarketData:
    """Tests for update_market_data function."""

    @patch('app.services.data_refresh_service.EntsoeClient')
    @patch('app.services.data_refresh_service.append_market_data')
    def test_update_day_ahead_success(self, mock_append, mock_entsoe_class, mock_now):
        """Test successful update of day-ahead market data."""
        # Setup mock client
        mock_client = MagicMock()
        mock_entsoe_class.return_value = mock_client

        # Mock fetch response
        new_data = pd.DataFrame({
            'timestamp_utc': pd.date_range(mock_now - timedelta(hours=3), mock_now, freq='15min', tz='UTC'),
            'price_eur_per_mwh': [50.0] * 13,
        })
        mock_client.fetch_day_ahead_prices_2025_nl.return_value = new_data
        mock_append.return_value = True

        # Call function
        result = update_market_data('day_ahead', mock_now)

        # Assertions
        assert result is True
        mock_client.fetch_day_ahead_prices_2025_nl.assert_called_once()
        mock_append.assert_called_once_with(new_data, 'day_ahead')

    @patch('app.services.data_refresh_service.EntsoeClient')
    @patch('app.services.data_refresh_service.append_market_data')
    def test_update_returns_empty_data(self, mock_append, mock_entsoe_class, mock_now):
        """Test handling of empty data from API."""
        mock_client = MagicMock()
        mock_entsoe_class.return_value = mock_client
        mock_client.fetch_day_ahead_prices_2025_nl.return_value = pd.DataFrame()

        result = update_market_data('day_ahead', mock_now)

        assert result is False
        mock_append.assert_not_called()

    @patch('app.services.data_refresh_service.EntsoeClient')
    def test_update_client_initialization_failure(self, mock_entsoe_class, mock_now):
        """Test handling of client initialization failure."""
        mock_entsoe_class.side_effect = ValueError("API key missing")

        result = update_market_data('day_ahead', mock_now)

        assert result is False


class TestUpdateWeatherData:
    """Tests for update_weather_data function."""

    @patch('app.services.data_refresh_service.MeteosourceClient')
    @patch('app.services.data_refresh_service.KNMIHistoricalClient')
    @patch('app.services.data_refresh_service.append_weather_data')
    def test_update_weather_success(self, mock_append, mock_knmi_class, mock_meteosource_class, mock_now):
        """Test successful weather data update."""
        # Setup mock KNMI client
        mock_knmi = MagicMock()
        mock_knmi_class.return_value = mock_knmi
        historical_data = pd.DataFrame({
            'temperature_deg_c': [15.0, 16.0],
            'wind_speed_m_per_s': [5.0, 6.0],
        })
        mock_knmi.fetch_hourly_data.return_value = historical_data

        # Setup mock Meteosource client
        mock_meteosource = MagicMock()
        mock_meteosource_class.return_value = mock_meteosource
        forecast_data = pd.DataFrame({
            'temperature_deg_c': [17.0, 18.0],
            'wind_speed_m_per_s': [7.0, 8.0],
        })
        mock_meteosource.fetch_forecast_for_2025.return_value = forecast_data
        mock_append.return_value = True

        # Call function
        result = update_weather_data(mock_now)

        # Assertions
        assert result is True
        assert mock_append.call_count == 2  # Called for both historical and forecast


class TestEnsureFreshData:
    """Tests for ensure_fresh_data function."""

    @patch('app.services.data_refresh_service.load_market_data')
    @patch('app.services.data_refresh_service.load_weather_data')
    @patch('app.services.data_refresh_service.update_market_data')
    @patch('app.services.data_refresh_service.update_weather_data')
    def test_all_data_fresh_no_updates(
        self,
        mock_update_weather,
        mock_update_market,
        mock_load_weather,
        mock_load_market,
        mock_now,
        mock_market_data_fresh,
        mock_weather_data_fresh,
    ):
        """Test that no updates are performed when all data is fresh."""
        # Setup mocks to return fresh data
        mock_load_market.return_value = mock_market_data_fresh
        mock_load_weather.return_value = mock_weather_data_fresh

        # Call function
        ensure_fresh_data(now=mock_now, freshness_threshold_minutes=60)

        # Assertions - no updates should be called
        mock_update_market.assert_not_called()
        mock_update_weather.assert_not_called()

    @patch('app.services.data_refresh_service.load_market_data')
    @patch('app.services.data_refresh_service.load_weather_data')
    @patch('app.services.data_refresh_service.update_market_data')
    @patch('app.services.data_refresh_service.update_weather_data')
    def test_stale_data_triggers_updates(
        self,
        mock_update_weather,
        mock_update_market,
        mock_load_weather,
        mock_load_market,
        mock_now,
        mock_market_data_stale,
        mock_weather_data_stale,
    ):
        """Test that updates are triggered when data is stale."""
        # Setup mocks to return stale data
        mock_load_market.return_value = mock_market_data_stale
        mock_load_weather.return_value = mock_weather_data_stale
        mock_update_market.return_value = True
        mock_update_weather.return_value = True

        # Call function
        ensure_fresh_data(now=mock_now, freshness_threshold_minutes=60)

        # Assertions - updates should be called
        # Note: will be called for each market type in MARKET_TYPES
        assert mock_update_market.call_count > 0
        mock_update_weather.assert_called_once()

    @patch('app.services.data_refresh_service.load_market_data')
    @patch('app.services.data_refresh_service.load_weather_data')
    @patch('app.services.data_refresh_service.update_market_data')
    @patch('app.services.data_refresh_service.update_weather_data')
    def test_mixed_freshness(
        self,
        mock_update_weather,
        mock_update_market,
        mock_load_weather,
        mock_load_market,
        mock_now,
        mock_market_data_fresh,
        mock_weather_data_stale,
    ):
        """Test that only stale data sources are updated."""
        # Setup mocks - market fresh, weather stale
        mock_load_market.return_value = mock_market_data_fresh
        mock_load_weather.return_value = mock_weather_data_stale
        mock_update_weather.return_value = True

        # Call function
        ensure_fresh_data(now=mock_now, freshness_threshold_minutes=60)

        # Assertions - only weather should be updated
        mock_update_market.assert_not_called()
        mock_update_weather.assert_called_once()

    @patch('app.services.data_refresh_service.load_market_data')
    @patch('app.services.data_refresh_service.load_weather_data')
    @patch('app.services.data_refresh_service.update_market_data')
    def test_specific_markets_only(
        self,
        mock_update_market,
        mock_load_weather,
        mock_load_market,
        mock_now,
        mock_market_data_stale,
    ):
        """Test that only specified markets are checked."""
        # Setup mocks
        mock_load_market.return_value = mock_market_data_stale
        mock_load_weather.return_value = pd.DataFrame()  # Empty weather
        mock_update_market.return_value = True

        # Call function with specific markets
        ensure_fresh_data(
            now=mock_now,
            freshness_threshold_minutes=60,
            markets_to_check=['day_ahead'],
        )

        # Assertions - should only be called for day_ahead
        assert mock_update_market.call_count == 1
        mock_update_market.assert_called_with('day_ahead', mock_now)

    @patch('app.services.data_refresh_service.load_market_data')
    @patch('app.services.data_refresh_service.load_weather_data')
    def test_no_existing_data_files(
        self,
        mock_load_weather,
        mock_load_market,
        mock_now,
    ):
        """Test handling when no data files exist (returns None)."""
        # Setup mocks to return None (no files)
        mock_load_market.return_value = None
        mock_load_weather.return_value = None

        # Should not raise exception
        ensure_fresh_data(now=mock_now, freshness_threshold_minutes=60)

    def test_uses_current_time_when_now_is_none(self):
        """Test that current UTC time is used when now parameter is None."""
        with patch('app.services.data_refresh_service.load_market_data'), \
             patch('app.services.data_refresh_service.load_weather_data'):
            # Should not raise exception
            ensure_fresh_data(now=None, freshness_threshold_minutes=60)
