"""
Tests for ENTSO-E client.
"""

from datetime import datetime
from unittest.mock import Mock, patch
from xml.etree import ElementTree as ET

import pandas as pd
import pytest
import requests
from app.services.entsoe_client import EntsoeClient


class TestEntsoeClientInitialization:
    """Test suite for EntsoeClient initialization."""

    def test_client_initialization_with_env_key(self, mock_env_vars):
        """Test that client initializes with API key from environment."""
        client = EntsoeClient()
        
        assert client.api_key == "test_entsoe_key"
        assert client.base_url is not None
        assert client.netherlands_eic == "10YNL----------L"
        assert "ns" in client.namespaces

    def test_client_initialization_with_explicit_key(self, mock_env_vars):
        """Test initialization with explicitly provided API key."""
        client = EntsoeClient(api_key="explicit_key")
        
        assert client.api_key == "explicit_key"

    def test_client_initialization_without_key(self):
        """Test that client allows explicit empty string (for testing purposes)."""
        # Explicitly passing empty string should be allowed for testing
        client = EntsoeClient(api_key="")
        assert client.api_key == ""


class TestEntsoeUrlBuilding:
    """Test suite for URL construction."""

    def test_build_query_url_basic(self, mock_env_vars):
        """Test basic URL construction with required parameters."""
        client = EntsoeClient()
        
        start_date = datetime(2025, 1, 1, 0, 0)
        end_date = datetime(2025, 1, 2, 0, 0)
        
        url = client._build_query_url("A44", start_date, end_date)
        
        assert "securityToken=test_entsoe_key" in url
        assert "documentType=A44" in url
        assert "in_Domain=10YNL----------L" in url
        assert "out_Domain=10YNL----------L" in url
        assert "periodStart=202501010000" in url
        assert "periodEnd=202501020000" in url

    def test_build_query_url_with_process_type(self, mock_env_vars):
        """Test URL construction with optional process type."""
        client = EntsoeClient()
        
        start_date = datetime(2025, 1, 1, 0, 0)
        end_date = datetime(2025, 1, 2, 0, 0)
        
        url = client._build_query_url("A44", start_date, end_date, process_type="A01")
        
        assert "processType=A01" in url

    def test_build_query_url_different_document_types(self, mock_env_vars):
        """Test URL construction for different document types."""
        client = EntsoeClient()
        
        start_date = datetime(2025, 1, 1, 0, 0)
        end_date = datetime(2025, 1, 2, 0, 0)
        
        # Day-ahead
        url_a44 = client._build_query_url("A44", start_date, end_date)
        assert "documentType=A44" in url_a44
        
        # Intraday
        url_a45 = client._build_query_url("A45", start_date, end_date)
        assert "documentType=A45" in url_a45
        
        # Imbalance
        url_a53 = client._build_query_url("A53", start_date, end_date)
        assert "documentType=A53" in url_a53

    def test_build_query_url_timestamp_formatting(self, mock_env_vars):
        """Test that timestamps are formatted correctly."""
        client = EntsoeClient()
        
        # Test with minutes and seconds
        start_date = datetime(2025, 3, 15, 14, 30, 45)
        end_date = datetime(2025, 3, 16, 9, 15, 20)
        
        url = client._build_query_url("A44", start_date, end_date)
        
        assert "periodStart=202503151430" in url
        assert "periodEnd=202503160915" in url


class TestEntsoeXmlParsing:
    """Test suite for XML parsing."""

    def test_parse_price_document_single_point(self, mock_env_vars, sample_entsoe_xml):
        """Test parsing XML with a single price point."""
        client = EntsoeClient()
        
        # Use first point only
        simple_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
    <TimeSeries>
        <Period>
            <timeInterval>
                <start>2025-01-01T00:00:00Z</start>
            </timeInterval>
            <resolution>PT60M</resolution>
            <Point>
                <position>1</position>
                <price.amount>45.50</price.amount>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>
"""
        
        df = client._parse_price_document(simple_xml, "day_ahead")
        
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]["price_eur_per_mwh"] == 45.50
        assert df.iloc[0]["market_type"] == "day_ahead"
        assert isinstance(df.iloc[0]["timestamp_utc"], pd.Timestamp)

    def test_parse_price_document_multiple_points(self, mock_env_vars, sample_entsoe_xml):
        """Test parsing XML with multiple price points."""
        client = EntsoeClient()
        
        df = client._parse_price_document(sample_entsoe_xml, "day_ahead")
        
        assert len(df) == 4
        assert list(df.columns) == ["timestamp_utc", "price_eur_per_mwh", "market_type"]
        assert all(df["market_type"] == "day_ahead")
        
        # Check prices
        expected_prices = [45.50, 46.75, 44.20, 43.80]
        assert df["price_eur_per_mwh"].tolist() == expected_prices

    def test_parse_price_document_sorted_by_time(self, mock_env_vars):
        """Test that parsed data is sorted by timestamp."""
        client = EntsoeClient()
        
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
    <TimeSeries>
        <Period>
            <timeInterval>
                <start>2025-01-01T00:00:00Z</start>
            </timeInterval>
            <resolution>PT60M</resolution>
            <Point>
                <position>1</position>
                <price.amount>10.0</price.amount>
            </Point>
            <Point>
                <position>2</position>
                <price.amount>20.0</price.amount>
            </Point>
            <Point>
                <position>3</position>
                <price.amount>15.0</price.amount>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>
"""
        
        df = client._parse_price_document(xml, "day_ahead")
        
        # Check that timestamps are in order
        assert df["timestamp_utc"].is_monotonic_increasing

    def test_parse_empty_xml(self, mock_env_vars):
        """Test parsing of empty XML document."""
        client = EntsoeClient()
        
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
</Publication_MarketDocument>
"""
        
        df = client._parse_price_document(xml_content, "day_ahead")
        
        assert df.empty
        assert list(df.columns) == ["timestamp_utc", "price_eur_per_mwh", "market_type"]

    def test_parse_malformed_xml(self, mock_env_vars):
        """Test handling of malformed XML."""
        client = EntsoeClient()
        
        malformed_xml = "<broken><xml>no closing tags"
        
        df = client._parse_price_document(malformed_xml, "day_ahead")
        
        # Should return empty DataFrame, not crash
        assert df.empty

    def test_parse_different_resolutions(self, mock_env_vars):
        """Test parsing with different time resolutions."""
        client = EntsoeClient()
        
        # 15-minute resolution
        xml_15min = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
    <TimeSeries>
        <Period>
            <timeInterval>
                <start>2025-01-01T00:00:00Z</start>
            </timeInterval>
            <resolution>PT15M</resolution>
            <Point>
                <position>1</position>
                <price.amount>40.0</price.amount>
            </Point>
            <Point>
                <position>2</position>
                <price.amount>41.0</price.amount>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>
"""
        
        df = client._parse_price_document(xml_15min, "intraday")
        
        assert len(df) == 2
        # Check that timestamps are 15 minutes apart
        time_diff = df.iloc[1]["timestamp_utc"] - df.iloc[0]["timestamp_utc"]
        assert time_diff == pd.Timedelta(minutes=15)


class TestEntsoeDataFetching:
    """Test suite for data fetching."""

    @patch("app.services.entsoe_client.requests.get")
    def test_fetch_data_success(self, mock_get, mock_env_vars, sample_entsoe_xml):
        """Test successful data fetching."""
        client = EntsoeClient()
        
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sample_entsoe_xml
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 2)
        
        df = client._fetch_data("A44", "day_ahead", start_date, end_date, chunk_days=1)
        
        assert not df.empty
        assert mock_get.called
        assert len(df) == 4

    @patch("app.services.entsoe_client.requests.get")
    def test_fetch_data_http_error(self, mock_get, mock_env_vars):
        """Test handling of HTTP errors."""
        client = EntsoeClient()
        
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Not found")
        mock_get.return_value = mock_response
        
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 2)
        
        df = client._fetch_data("A44", "day_ahead", start_date, end_date, chunk_days=1)
        
        # Should return empty DataFrame on error
        assert df.empty

    @patch("app.services.entsoe_client.requests.get")
    def test_fetch_data_request_exception(self, mock_get, mock_env_vars):
        """Test handling of request exceptions."""
        client = EntsoeClient()
        
        # Mock network error
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 2)
        
        df = client._fetch_data("A44", "day_ahead", start_date, end_date, chunk_days=1)
        
        assert df.empty

    @patch("app.services.entsoe_client.requests.get")
    def test_fetch_day_ahead_prices(self, mock_get, mock_env_vars, sample_entsoe_xml):
        """Test fetching day-ahead prices."""
        client = EntsoeClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sample_entsoe_xml
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        df = client.fetch_day_ahead_prices_2025_nl()
        
        assert not df.empty
        assert all(df["market_type"] == "day_ahead")

    @patch("app.services.entsoe_client.requests.get")
    def test_fetch_intraday_prices(self, mock_get, mock_env_vars, sample_entsoe_xml):
        """Test fetching intraday prices."""
        client = EntsoeClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sample_entsoe_xml
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        df = client.fetch_intraday_prices_2025_nl()
        
        assert not df.empty

    @patch("app.services.entsoe_client.requests.get")
    def test_fetch_imbalance_data(self, mock_get, mock_env_vars, sample_entsoe_xml):
        """Test fetching imbalance data."""
        client = EntsoeClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sample_entsoe_xml
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        df = client.fetch_imbalance_data_2025_nl()
        
        assert not df.empty

    @patch("app.services.entsoe_client.requests.get")
    def test_fetch_data_deduplicates_results(self, mock_get, mock_env_vars):
        """Test that duplicate timestamps are removed."""
        client = EntsoeClient()
        
        # XML with duplicate timestamps
        xml_with_dupes = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
    <TimeSeries>
        <Period>
            <timeInterval>
                <start>2025-01-01T00:00:00Z</start>
            </timeInterval>
            <resolution>PT60M</resolution>
            <Point>
                <position>1</position>
                <price.amount>45.0</price.amount>
            </Point>
            <Point>
                <position>1</position>
                <price.amount>46.0</price.amount>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>
"""
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = xml_with_dupes
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 2)
        
        df = client._fetch_data("A44", "day_ahead", start_date, end_date, chunk_days=1)
        
        # Should have only one row after deduplication
        assert len(df) == 1
