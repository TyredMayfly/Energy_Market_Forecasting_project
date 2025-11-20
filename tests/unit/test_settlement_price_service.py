"""
Unit tests for Settlement Price Service
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import pytz

from app.services.settlement_price_service import (
    SettlementPriceService,
    SettlementPriceUnavailableError,
    get_settlement_price_service,
)
from app.services.tennet_client import TennetApiError


@pytest.fixture
def temp_data_file():
    """Create temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def fresh_local_data():
    """Create fresh local settlement price data (5 minutes old)."""
    now = datetime.now(pytz.UTC)
    timestamps = pd.date_range(end=now - timedelta(minutes=5), periods=10, freq="15min")

    return pd.DataFrame({
        "timestamp_utc": timestamps,
        "shortage_price": [100.0 + i * 5 for i in range(10)],
        "surplus_price": [80.0 + i * 3 for i in range(10)],
        "regulation_state": [1, -1, 2, 1, -1, 2, 1, -1, 2, 1],
    })


@pytest.fixture
def stale_local_data():
    """Create stale local settlement price data (30 minutes old)."""
    now = datetime.now(pytz.UTC)
    timestamps = pd.date_range(end=now - timedelta(minutes=30), periods=10, freq="15min")

    return pd.DataFrame({
        "timestamp_utc": timestamps,
        "shortage_price": [100.0 + i * 5 for i in range(10)],
        "surplus_price": [80.0 + i * 3 for i in range(10)],
        "regulation_state": [1, -1, 2, 1, -1, 2, 1, -1, 2, 1],
    })


@pytest.fixture
def api_response_data():
    """Create sample API response data (current time)."""
    now = datetime.now(pytz.UTC)
    timestamps = pd.date_range(end=now, periods=5, freq="15min")

    return pd.DataFrame({
        "timestamp_utc": timestamps,
        "shortage_price": [150.0 + i * 10 for i in range(5)],
        "surplus_price": [120.0 + i * 5 for i in range(5)],
        "regulation_state": [2, 1, -1, 2, 1],
    })


@pytest.fixture
def service_with_temp_file(temp_data_file):
    """Create service instance with temporary data file."""
    return SettlementPriceService(
        local_data_path=temp_data_file,
        timezone="UTC",
    )


class TestSettlementPriceService:
    """Tests for SettlementPriceService class."""

    def test_init_with_defaults(self):
        """Test initialization with default paths."""
        with patch("app.services.settlement_price_service.settings") as mock_settings:
            mock_settings.data_dir = Path("/test/data")
            mock_settings.timezone = "Europe/Amsterdam"

            service = SettlementPriceService()

            assert service.local_data_path == Path("/test/data/imbalance_unified.csv")
            assert service.timezone.zone == "Europe/Amsterdam"

    def test_init_with_custom_path(self, temp_data_file):
        """Test initialization with custom data path."""
        service = SettlementPriceService(
            local_data_path=temp_data_file,
            timezone="UTC",
        )

        assert service.local_data_path == temp_data_file
        assert service.timezone.zone == "UTC"

    def test_load_local_data_success(self, temp_data_file, fresh_local_data):
        """Test loading local data successfully."""
        # Save test data
        fresh_local_data.to_csv(temp_data_file, index=False)

        service = SettlementPriceService(local_data_path=temp_data_file)
        df = service._load_local_data()

        assert not df.empty
        assert len(df) == 10
        assert "timestamp_utc" in df.columns
        assert df["timestamp_utc"].dt.tz is not None

    def test_load_local_data_file_not_found(self, temp_data_file):
        """Test loading when file doesn't exist."""
        # Don't create the file
        service = SettlementPriceService(local_data_path=temp_data_file)
        df = service._load_local_data()

        assert df.empty

    def test_save_local_data_new_file(self, temp_data_file, fresh_local_data):
        """Test saving data to new file."""
        service = SettlementPriceService(local_data_path=temp_data_file)

        success = service._save_local_data(fresh_local_data)

        assert success
        assert temp_data_file.exists()

        # Verify saved data
        loaded_df = pd.read_csv(temp_data_file, parse_dates=["timestamp_utc"])
        assert len(loaded_df) == 10

    def test_save_local_data_append(self, temp_data_file, fresh_local_data, api_response_data):
        """Test appending data to existing file."""
        # Save initial data
        fresh_local_data.to_csv(temp_data_file, index=False)

        service = SettlementPriceService(local_data_path=temp_data_file)

        # Append new data
        success = service._save_local_data(api_response_data)

        assert success

        # Verify combined data
        df = service._load_local_data()
        assert len(df) == 15  # 10 + 5
        assert df["timestamp_utc"].is_monotonic_increasing

    def test_save_local_data_removes_duplicates(self, temp_data_file, fresh_local_data):
        """Test that saving removes duplicate timestamps."""
        # Save initial data
        fresh_local_data.to_csv(temp_data_file, index=False)

        service = SettlementPriceService(local_data_path=temp_data_file)

        # Try to save overlapping data
        duplicate_data = fresh_local_data.copy()
        duplicate_data["shortage_price"] = 999.0  # Different price

        service._save_local_data(duplicate_data)

        # Verify duplicates removed (keeps last)
        df = service._load_local_data()
        assert len(df) == 10
        assert df.iloc[0]["shortage_price"] == 999.0  # New value kept

    def test_get_current_time_with_now(self, service_with_temp_file):
        """Test getting current time with provided datetime."""
        test_time = datetime(2024, 11, 20, 12, 0, 0, tzinfo=pytz.UTC)

        result = service_with_temp_file._get_current_time(now=test_time)

        assert result == test_time
        assert result.tzinfo is not None

    def test_get_current_time_with_naive_datetime(self, service_with_temp_file):
        """Test getting current time with naive datetime (gets timezone)."""
        naive_time = datetime(2024, 11, 20, 12, 0, 0)

        result = service_with_temp_file._get_current_time(now=naive_time)

        assert result.tzinfo is not None
        assert str(result.tzinfo) == "UTC"

    def test_get_latest_price_fresh_local_data(
        self, temp_data_file, fresh_local_data
    ):
        """Test getting latest price when local data is fresh (<15 min)."""
        # Save fresh data
        fresh_local_data.to_csv(temp_data_file, index=False)

        service = SettlementPriceService(local_data_path=temp_data_file)

        # Mock TenneT client to ensure it's not called
        with patch.object(service.tennet_client, "fetch_latest_settlement_prices") as mock_fetch:
            price = service.get_latest_settlement_price(price_type="shortage")

            # Should return local price without calling API
            mock_fetch.assert_not_called()
            assert isinstance(price, float)
            assert price == fresh_local_data.iloc[-1]["shortage_price"]

    def test_get_latest_price_stale_local_data_api_success(
        self, temp_data_file, stale_local_data, api_response_data
    ):
        """Test getting latest price when local data is stale and API succeeds."""
        # Save stale data
        stale_local_data.to_csv(temp_data_file, index=False)

        service = SettlementPriceService(local_data_path=temp_data_file)

        # Mock successful API call
        with patch.object(
            service.tennet_client,
            "fetch_latest_settlement_prices",
            return_value=api_response_data
        ):
            price = service.get_latest_settlement_price(price_type="shortage")

            # Should return API price
            assert isinstance(price, float)
            assert price == api_response_data.iloc[-1]["shortage_price"]

            # Verify data was saved locally
            local_df = service._load_local_data()
            assert len(local_df) > len(stale_local_data)

    def test_get_latest_price_stale_local_data_api_failure(
        self, temp_data_file, stale_local_data
    ):
        """Test fallback to stale local data when API fails."""
        # Save stale data
        stale_local_data.to_csv(temp_data_file, index=False)

        service = SettlementPriceService(local_data_path=temp_data_file)

        # Mock API failure
        with patch.object(
            service.tennet_client,
            "fetch_latest_settlement_prices",
            side_effect=TennetApiError("API unavailable")
        ):
            price = service.get_latest_settlement_price(price_type="shortage")

            # Should fall back to stale local price
            assert isinstance(price, float)
            assert price == stale_local_data.iloc[-1]["shortage_price"]

    def test_get_latest_price_no_local_data_api_success(
        self, temp_data_file, api_response_data
    ):
        """Test getting price when no local data exists but API succeeds."""
        service = SettlementPriceService(local_data_path=temp_data_file)

        # Mock successful API call
        with patch.object(
            service.tennet_client,
            "fetch_latest_settlement_prices",
            return_value=api_response_data
        ):
            price = service.get_latest_settlement_price(price_type="shortage")

            # Should return API price
            assert isinstance(price, float)
            assert price == api_response_data.iloc[-1]["shortage_price"]

            # Verify data was saved locally
            assert temp_data_file.exists()

    def test_get_latest_price_no_local_data_api_failure(self, temp_data_file):
        """Test error when no local data and API fails."""
        service = SettlementPriceService(local_data_path=temp_data_file)

        # Mock API failure
        with patch.object(
            service.tennet_client,
            "fetch_latest_settlement_prices",
            side_effect=TennetApiError("API unavailable")
        ):
            with pytest.raises(SettlementPriceUnavailableError, match="API failed"):
                service.get_latest_settlement_price(price_type="shortage")

    def test_get_latest_price_surplus_type(self, temp_data_file, fresh_local_data):
        """Test getting surplus price instead of shortage price."""
        fresh_local_data.to_csv(temp_data_file, index=False)

        service = SettlementPriceService(local_data_path=temp_data_file)

        price = service.get_latest_settlement_price(price_type="surplus")

        assert isinstance(price, float)
        assert price == fresh_local_data.iloc[-1]["surplus_price"]

    def test_get_latest_price_invalid_type(self, service_with_temp_file):
        """Test error with invalid price type."""
        with pytest.raises(ValueError, match="Invalid price_type"):
            service_with_temp_file.get_latest_settlement_price(price_type="invalid")

    def test_get_latest_price_api_returns_empty(
        self, temp_data_file, stale_local_data
    ):
        """Test fallback when API returns empty data."""
        stale_local_data.to_csv(temp_data_file, index=False)

        service = SettlementPriceService(local_data_path=temp_data_file)

        # Mock API returning empty DataFrame
        with patch.object(
            service.tennet_client,
            "fetch_latest_settlement_prices",
            return_value=pd.DataFrame()
        ):
            price = service.get_latest_settlement_price(price_type="shortage")

            # Should fall back to stale local price
            assert price == stale_local_data.iloc[-1]["shortage_price"]

    def test_get_prices_between_success(self, temp_data_file, fresh_local_data):
        """Test getting settlement prices for a date range."""
        fresh_local_data.to_csv(temp_data_file, index=False)

        service = SettlementPriceService(local_data_path=temp_data_file)

        start = fresh_local_data["timestamp_utc"].iloc[0]
        end = fresh_local_data["timestamp_utc"].iloc[-1]

        df = service.get_settlement_prices_between(start, end, ensure_fresh=False)

        assert not df.empty
        assert len(df) == 10
        assert all(df["timestamp_utc"] >= start)
        assert all(df["timestamp_utc"] <= end)

    def test_get_prices_between_with_freshness_check(
        self, temp_data_file, stale_local_data, api_response_data
    ):
        """Test getting prices with freshness check enabled."""
        stale_local_data.to_csv(temp_data_file, index=False)

        service = SettlementPriceService(local_data_path=temp_data_file)

        # Mock API call
        with patch.object(
            service.tennet_client,
            "fetch_latest_settlement_prices",
            return_value=api_response_data
        ):
            end = datetime.now(pytz.UTC)
            start = end - timedelta(hours=1)

            df = service.get_settlement_prices_between(
                start, end, ensure_fresh=True
            )

            # Should trigger update and return data
            assert not df.empty

    def test_get_prices_between_naive_datetime(self, service_with_temp_file):
        """Test error when using naive datetimes."""
        start = datetime(2024, 11, 20, 10, 0, 0)  # No timezone
        end = datetime(2024, 11, 20, 12, 0, 0, tzinfo=pytz.UTC)

        with pytest.raises(ValueError, match="must be timezone-aware"):
            service_with_temp_file.get_settlement_prices_between(start, end)

    def test_get_settlement_price_service_singleton(self):
        """Test singleton pattern for service instance."""
        service1 = get_settlement_price_service()
        service2 = get_settlement_price_service()

        assert service1 is service2

    def test_freshness_threshold_constant(self):
        """Test that freshness threshold is set correctly."""
        assert SettlementPriceService.FRESHNESS_THRESHOLD == timedelta(minutes=15)
