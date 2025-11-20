"""
Unit tests for TenneT Settlement Prices API Client
"""

import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import pytz
import requests

from app.services.tennet_client import (
    TennetApiError,
    TennetClient,
    get_tennet_client,
)


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def tennet_client(mock_api_key):
    """Create TenneT client with mock API key."""
    return TennetClient(
        base_url="https://test-api.tennet.eu/v1",
        api_key=mock_api_key,
        timeout=5,
    )


@pytest.fixture
def sample_api_response():
    """Sample API response data."""
    return {
        "data": [
            {
                "timeinterval_start": "2024-11-20T10:00:00Z",
                "shortage_price": 125.50,
                "surplus_price": 85.30,
                "regulation_state": 2,
            },
            {
                "timeinterval_start": "2024-11-20T10:15:00Z",
                "shortage_price": 130.00,
                "surplus_price": 90.00,
                "regulation_state": 1,
            },
            {
                "timeinterval_start": "2024-11-20T10:30:00Z",
                "shortage_price": 120.00,
                "surplus_price": 80.00,
                "regulation_state": -1,
            },
        ]
    }


class TestTennetClient:
    """Tests for TennetClient class."""

    def test_init_with_defaults(self):
        """Test initialization with default settings."""
        with patch("app.services.tennet_client.settings") as mock_settings:
            mock_settings.tennet_settlement_base_url = "https://api.tennet.eu/v1"
            mock_settings.tennet_api_key = "default-key"

            client = TennetClient()

            assert client.base_url == "https://api.tennet.eu/v1"
            assert client.api_key == "default-key"
            assert client.timeout == 10

    def test_init_with_custom_values(self, mock_api_key):
        """Test initialization with custom values."""
        client = TennetClient(
            base_url="https://custom.api/v2",
            api_key=mock_api_key,
            timeout=15,
        )

        assert client.base_url == "https://custom.api/v2"
        assert client.api_key == mock_api_key
        assert client.timeout == 15

    def test_get_headers_with_api_key(self, tennet_client, mock_api_key):
        """Test HTTP headers include authentication."""
        headers = tennet_client._get_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {mock_api_key}"
        assert headers["Accept"] == "application/json"

    def test_get_headers_without_api_key(self):
        """Test HTTP headers without API key."""
        client = TennetClient(api_key="")

        headers = client._get_headers()

        assert "Authorization" not in headers
        assert headers["Accept"] == "application/json"

    @patch("app.services.tennet_client.requests.get")
    def test_make_request_success(self, mock_get, tennet_client, sample_api_response):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_get.return_value = mock_response

        result = tennet_client._make_request("settlement-prices/latest")

        assert result == sample_api_response
        mock_get.assert_called_once()

    @patch("app.services.tennet_client.requests.get")
    def test_make_request_http_error(self, mock_get, tennet_client):
        """Test API request with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"message": "Not found"}
        mock_response.text = "Not found"
        mock_get.return_value = mock_response

        with pytest.raises(TennetApiError, match="HTTP 404"):
            tennet_client._make_request("invalid-endpoint")

    @patch("app.services.tennet_client.requests.get")
    def test_make_request_timeout(self, mock_get, tennet_client):
        """Test API request timeout."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")

        with pytest.raises(TennetApiError, match="Request timeout"):
            tennet_client._make_request("settlement-prices/latest")

    @patch("app.services.tennet_client.requests.get")
    def test_make_request_connection_error(self, mock_get, tennet_client):
        """Test API request connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network unreachable")

        with pytest.raises(TennetApiError, match="Connection error"):
            tennet_client._make_request("settlement-prices/latest")

    def test_parse_settlement_data_success(self, tennet_client, sample_api_response):
        """Test parsing valid API response."""
        df = tennet_client._parse_settlement_data(sample_api_response)

        assert not df.empty
        assert len(df) == 3
        assert "timestamp_utc" in df.columns
        assert "shortage_price" in df.columns
        assert "surplus_price" in df.columns

        # Check timezone awareness
        assert df["timestamp_utc"].dt.tz is not None
        assert str(df["timestamp_utc"].dt.tz) == "UTC"

        # Check data types
        assert pd.api.types.is_float_dtype(df["shortage_price"])
        assert pd.api.types.is_float_dtype(df["surplus_price"])

        # Check values
        assert df.iloc[0]["shortage_price"] == 125.50
        assert df.iloc[1]["surplus_price"] == 90.00

    def test_parse_settlement_data_empty(self, tennet_client):
        """Test parsing empty API response."""
        df = tennet_client._parse_settlement_data({"data": []})

        assert df.empty

    def test_parse_settlement_data_missing_timestamp(self, tennet_client):
        """Test parsing data without timestamp column."""
        invalid_data = {
            "data": [
                {"shortage_price": 100.0, "surplus_price": 80.0}
            ]
        }

        with pytest.raises(TennetApiError, match="No timestamp column"):
            tennet_client._parse_settlement_data(invalid_data)

    def test_parse_settlement_data_alternative_format(self, tennet_client):
        """Test parsing data with alternative column names."""
        alt_data = {
            "data": [
                {
                    "Timeinterval Start Loc": "2024-11-20T10:00:00Z",
                    "Price Shortage": 125.50,
                    "Price Surplus": 85.30,
                }
            ]
        }

        df = tennet_client._parse_settlement_data(alt_data)

        assert not df.empty
        assert "timestamp_utc" in df.columns
        assert "shortage_price" in df.columns
        assert df.iloc[0]["shortage_price"] == 125.50

    @patch.object(TennetClient, "_make_request")
    @patch.object(TennetClient, "_parse_settlement_data")
    def test_fetch_latest_settlement_prices(
        self, mock_parse, mock_request, tennet_client, sample_api_response
    ):
        """Test fetching latest settlement prices."""
        mock_request.return_value = sample_api_response
        mock_df = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(["2024-11-20T10:00:00Z"], utc=True),
            "shortage_price": [125.50],
            "surplus_price": [85.30],
        })
        mock_parse.return_value = mock_df

        df = tennet_client.fetch_latest_settlement_prices(limit=50)

        assert not df.empty
        mock_request.assert_called_once_with(
            "settlement-prices/latest",
            params={"limit": 50}
        )
        mock_parse.assert_called_once()

    @patch.object(TennetClient, "_make_request")
    def test_fetch_latest_api_error(self, mock_request, tennet_client):
        """Test fetch latest with API error."""
        mock_request.side_effect = TennetApiError("API unavailable")

        with pytest.raises(TennetApiError, match="API unavailable"):
            tennet_client.fetch_latest_settlement_prices()

    @patch.object(TennetClient, "_make_request")
    @patch.object(TennetClient, "_parse_settlement_data")
    def test_fetch_settlement_prices_from_to(
        self, mock_parse, mock_request, tennet_client
    ):
        """Test fetching settlement prices for date range."""
        start = datetime(2024, 11, 20, 10, 0, 0, tzinfo=pytz.UTC)
        end = datetime(2024, 11, 20, 12, 0, 0, tzinfo=pytz.UTC)

        mock_df = pd.DataFrame({
            "timestamp_utc": pd.date_range(start, end, freq="15min", tz="UTC"),
            "shortage_price": [100.0] * 9,
            "surplus_price": [80.0] * 9,
        })
        mock_parse.return_value = mock_df
        mock_request.return_value = {}

        df = tennet_client.fetch_settlement_prices_from_to(start, end)

        assert not df.empty
        assert len(df) == 9
        mock_request.assert_called_once()

        # Check that params contain formatted dates
        call_args = mock_request.call_args
        assert "from" in call_args[1]["params"]
        assert "to" in call_args[1]["params"]

    def test_fetch_from_to_no_timezone(self, tennet_client):
        """Test fetch with naive datetime raises error."""
        start = datetime(2024, 11, 20, 10, 0, 0)  # No timezone
        end = datetime(2024, 11, 20, 12, 0, 0, tzinfo=pytz.UTC)

        with pytest.raises(ValueError, match="must be timezone-aware"):
            tennet_client.fetch_settlement_prices_from_to(start, end)

    def test_get_tennet_client_singleton(self):
        """Test singleton pattern for client instance."""
        client1 = get_tennet_client()
        client2 = get_tennet_client()

        assert client1 is client2
