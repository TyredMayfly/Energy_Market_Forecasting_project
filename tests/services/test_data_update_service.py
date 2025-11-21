"""
Tests for data update service.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call
import pandas as pd
import pytz


@pytest.fixture
def mock_day_ahead_data():
    """Create mock day-ahead price data."""
    timestamps = pd.date_range('2024-10-01', periods=24, freq='h', tz='UTC')
    return pd.DataFrame({
        'timestamp_utc': timestamps,
        'price_eur_per_mwh': [45.5 + i for i in range(24)]
    })


@pytest.fixture
def mock_imbalance_data():
    """Create mock imbalance data."""
    timestamps = pd.date_range('2024-10-01', periods=24, freq='h', tz='UTC')
    return pd.DataFrame({
        'timestamp_utc': timestamps,
        'shortage_price': [50.0 + i for i in range(24)],
        'surplus_price': [40.0 + i for i in range(24)],
        'regulation_state': [1] * 24
    })


@pytest.fixture
def mock_weather_data():
    """Create mock weather data."""
    timestamps = pd.date_range('2024-10-01', periods=24, freq='h', tz='UTC')
    return pd.DataFrame({
        'timestamp_utc': timestamps,
        'temperature_celsius': [15.0 + i * 0.5 for i in range(24)],
        'wind_speed_ms': [5.0] * 24
    })


class TestInitializeHistoricalData:
    """Tests for initialize_historical_data_2025 function."""
    
    @patch('app.services.data_update_service.get_data_summary')
    @patch('app.services.data_update_service.save_weather_data')
    @patch('app.services.data_update_service.save_market_data')
    @patch('app.services.data_update_service.MeteosourceClient')
    @patch('app.services.data_update_service.EntsoeClient')
    def test_initialize_success(
        self, 
        mock_entsoe_cls, 
        mock_meteosource_cls,
        mock_save_market,
        mock_save_weather,
        mock_summary,
        mock_day_ahead_data,
        mock_imbalance_data,
        mock_weather_data
    ):
        """Test successful data initialization."""
        from app.services.data_update_service import initialize_historical_data_2025
        
        # Mock ENTSO-E client
        mock_entsoe = MagicMock()
        mock_entsoe.fetch_day_ahead_prices_2025_nl.return_value = mock_day_ahead_data
        mock_entsoe.fetch_imbalance_data_2025_nl.return_value = mock_imbalance_data
        mock_entsoe_cls.return_value = mock_entsoe
        
        # Mock Meteosource client
        mock_meteosource = MagicMock()
        mock_meteosource.fetch_forecast_for_2025.return_value = mock_weather_data
        mock_meteosource_cls.return_value = mock_meteosource
        
        # Mock summary
        mock_summary.return_value = {
            "markets": {
                "day_ahead": {"records": 24, "start_date": "2024-10-01", "end_date": "2024-10-02"}
            },
            "weather": {"records": 24, "start_date": "2024-10-01", "end_date": "2024-10-02"}
        }
        
        result = initialize_historical_data_2025()
        
        assert result is True
        mock_entsoe.fetch_day_ahead_prices_2025_nl.assert_called_once()
        mock_entsoe.fetch_imbalance_data_2025_nl.assert_called_once()
        mock_meteosource.fetch_forecast_for_2025.assert_called_once()
        assert mock_save_market.call_count >= 1
        mock_save_weather.assert_called_once()
    
    @patch('app.services.data_update_service.EntsoeClient', side_effect=ValueError("No API key"))
    def test_initialize_entsoe_client_failure(self, mock_entsoe_cls):
        """Test initialization fails when ENTSO-E client cannot be created."""
        from app.services.data_update_service import initialize_historical_data_2025
        
        result = initialize_historical_data_2025()
        
        assert result is False
    
    @patch('app.services.data_update_service.get_data_summary')
    @patch('app.services.data_update_service.save_market_data')
    @patch('app.services.data_update_service.MeteosourceClient')
    @patch('app.services.data_update_service.EntsoeClient')
    def test_initialize_empty_day_ahead(
        self,
        mock_entsoe_cls,
        mock_meteosource_cls,
        mock_save_market,
        mock_summary
    ):
        """Test initialization when day-ahead data is empty."""
        from app.services.data_update_service import initialize_historical_data_2025
        
        mock_entsoe = MagicMock()
        mock_entsoe.fetch_day_ahead_prices_2025_nl.return_value = pd.DataFrame()
        mock_entsoe.fetch_imbalance_data_2025_nl.return_value = pd.DataFrame()
        mock_entsoe_cls.return_value = mock_entsoe
        
        mock_meteosource = MagicMock()
        mock_meteosource.fetch_forecast_for_2025.return_value = pd.DataFrame()
        mock_meteosource_cls.return_value = mock_meteosource
        
        mock_summary.return_value = {"markets": {}, "weather": {"records": 0}}
        
        result = initialize_historical_data_2025()
        
        # Should return False because day-ahead is empty
        assert result is False
    
    @patch('app.services.data_update_service.get_data_summary')
    @patch('app.services.data_update_service.save_weather_data')
    @patch('app.services.data_update_service.save_market_data')
    @patch('app.services.data_update_service.MeteosourceClient')
    @patch('app.services.data_update_service.EntsoeClient')
    def test_initialize_entsoe_fetch_exception(
        self,
        mock_entsoe_cls,
        mock_meteosource_cls,
        mock_save_market,
        mock_save_weather,
        mock_summary
    ):
        """Test initialization handles exceptions during data fetch."""
        from app.services.data_update_service import initialize_historical_data_2025
        
        mock_entsoe = MagicMock()
        mock_entsoe.fetch_day_ahead_prices_2025_nl.side_effect = Exception("API error")
        mock_entsoe_cls.return_value = mock_entsoe
        
        mock_meteosource = MagicMock()
        mock_meteosource.fetch_forecast_for_2025.return_value = pd.DataFrame()
        mock_meteosource_cls.return_value = mock_meteosource
        
        mock_summary.return_value = {"markets": {}, "weather": {"records": 0}}
        
        result = initialize_historical_data_2025()
        
        assert result is False


class TestUpdateLatestData:
    """Tests for update_latest_data function."""
    
    @patch('app.services.data_update_service.save_weather_data')
    @patch('app.services.data_update_service.append_market_data')
    @patch('app.services.data_update_service.MeteosourceClient')
    @patch('app.services.data_update_service.EntsoeClient')
    def test_update_success(
        self,
        mock_entsoe_cls,
        mock_meteosource_cls,
        mock_append_market,
        mock_save_weather,
        mock_day_ahead_data,
        mock_imbalance_data,
        mock_weather_data
    ):
        """Test successful data update."""
        from app.services.data_update_service import update_latest_data
        
        # Mock ENTSO-E client
        mock_entsoe = MagicMock()
        mock_entsoe.fetch_day_ahead_prices_2025_nl.return_value = mock_day_ahead_data
        mock_entsoe.fetch_intraday_prices_2025_nl.return_value = pd.DataFrame()
        mock_entsoe.fetch_imbalance_data_2025_nl.return_value = mock_imbalance_data
        mock_entsoe_cls.return_value = mock_entsoe
        
        # Mock Meteosource client
        mock_meteosource = MagicMock()
        mock_meteosource.fetch_forecast_for_2025.return_value = mock_weather_data
        mock_meteosource_cls.return_value = mock_meteosource
        
        result = update_latest_data()
        
        assert result is True
        mock_entsoe.fetch_day_ahead_prices_2025_nl.assert_called_once()
        mock_entsoe.fetch_intraday_prices_2025_nl.assert_called_once()
        mock_entsoe.fetch_imbalance_data_2025_nl.assert_called_once()
        mock_meteosource.fetch_forecast_for_2025.assert_called_once()
        assert mock_append_market.call_count >= 1
        mock_save_weather.assert_called_once()
    
    @patch('app.services.data_update_service.EntsoeClient', side_effect=ValueError("No API key"))
    def test_update_entsoe_client_failure(self, mock_entsoe_cls):
        """Test update fails when ENTSO-E client cannot be created."""
        from app.services.data_update_service import update_latest_data
        
        result = update_latest_data()
        
        assert result is False
    
    @patch('app.services.data_update_service.save_weather_data')
    @patch('app.services.data_update_service.append_market_data')
    @patch('app.services.data_update_service.MeteosourceClient')
    @patch('app.services.data_update_service.EntsoeClient')
    def test_update_empty_data(
        self,
        mock_entsoe_cls,
        mock_meteosource_cls,
        mock_append_market,
        mock_save_weather
    ):
        """Test update when all data is empty."""
        from app.services.data_update_service import update_latest_data
        
        mock_entsoe = MagicMock()
        mock_entsoe.fetch_day_ahead_prices_2025_nl.return_value = pd.DataFrame()
        mock_entsoe.fetch_intraday_prices_2025_nl.return_value = pd.DataFrame()
        mock_entsoe.fetch_imbalance_data_2025_nl.return_value = pd.DataFrame()
        mock_entsoe_cls.return_value = mock_entsoe
        
        mock_meteosource = MagicMock()
        mock_meteosource.fetch_forecast_for_2025.return_value = pd.DataFrame()
        mock_meteosource_cls.return_value = mock_meteosource
        
        result = update_latest_data()
        
        # Should still return True even if data is empty
        assert result is True
        # Should not append empty data
        mock_append_market.assert_not_called()


class TestScheduler:
    """Tests for scheduler functions."""
    
    @patch('app.services.data_update_service.BackgroundScheduler')
    def test_start_scheduler(self, mock_scheduler_cls):
        """Test scheduler starts correctly."""
        from app.services.data_update_service import start_scheduler
        
        mock_scheduler = MagicMock()
        mock_scheduler.running = False
        mock_scheduler_cls.return_value = mock_scheduler
        
        start_scheduler()
        
        mock_scheduler.start.assert_called_once()
        # Verify job was added (at least one call to add_job)
        assert mock_scheduler.add_job.called
    
    @patch('app.services.data_update_service._scheduler', None)
    @patch('app.services.data_update_service.BackgroundScheduler')
    def test_stop_scheduler(self, mock_scheduler_cls):
        """Test scheduler stops correctly."""
        from app.services import data_update_service
        from app.services.data_update_service import start_scheduler, stop_scheduler
        
        mock_scheduler = MagicMock()
        mock_scheduler.running = False  # Initially not running
        mock_scheduler_cls.return_value = mock_scheduler
        
        start_scheduler()
        
        # Set running to True and assign to global
        mock_scheduler.running = True
        data_update_service._scheduler = mock_scheduler
        
        stop_scheduler()
        
        mock_scheduler.shutdown.assert_called_once()
    
    def test_stop_scheduler_when_not_started(self):
        """Test stopping scheduler that was never started."""
        from app.services.data_update_service import stop_scheduler
        
        # Should not raise exception
        stop_scheduler()
