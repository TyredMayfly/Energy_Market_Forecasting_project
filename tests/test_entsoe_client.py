"""
Tests for ENTSO-E client.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from app.services.entsoe_client import EntsoeClient


class TestEntsoeClient:
    """Test suite for EntsoeClient."""

    def test_client_initialization(self, mock_env_vars):
        """Test that client initializes with API key."""
        client = EntsoeClient()
        assert client.api_key == "test_entsoe_key"
        assert client.netherlands_eic == "10YNL----------L"

    def test_client_initialization_without_key(self, monkeypatch):
        """Test that client allows initialization with explicit empty key (for testing)."""
        monkeypatch.delenv("ENTSOE_API_KEY", raising=False)

        # Need to reload modules to pick up env changes
        import importlib
        from app.core import config
        from app.services import entsoe_client
        importlib.reload(config)
        importlib.reload(entsoe_client)

        # Should allow initialization with explicit empty string (for testing)
        client = entsoe_client.EntsoeClient(api_key="")
        assert client.api_key == ""

    def test_build_query_url(self, mock_env_vars):
        """Test URL construction."""
        client = EntsoeClient()

        start_date = datetime(2025, 1, 1, 0, 0)
        end_date = datetime(2025, 1, 2, 0, 0)

        url = client._build_query_url("A44", start_date, end_date)

        assert "securityToken=test_entsoe_key" in url
        assert "documentType=A44" in url
        assert "in_Domain=10YNL----------L" in url
        assert "periodStart=202501010000" in url
        assert "periodEnd=202501020000" in url

    def test_parse_price_document(self, mock_env_vars):
        """Test XML parsing."""
        client = EntsoeClient()

        # Sample XML response
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
            <TimeSeries>
                <Period>
                    <timeInterval>
                        <start>2025-01-01T00:00:00Z</start>
                        <end>2025-01-01T01:00:00Z</end>
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

        df = client._parse_price_document(xml_content, "day_ahead")

        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]["price_eur_per_mwh"] == 45.50
        assert df.iloc[0]["market_type"] == "day_ahead"

    def test_parse_empty_xml(self, mock_env_vars):
        """Test parsing of empty XML."""
        client = EntsoeClient()

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
        </Publication_MarketDocument>
        """

        df = client._parse_price_document(xml_content, "day_ahead")

        assert df.empty
        assert list(df.columns) == ["timestamp_utc", "price_eur_per_mwh", "market_type"]

    @patch("app.services.entsoe_client.requests.get")
    def test_fetch_data_success(self, mock_get, mock_env_vars):
        """Test successful data fetching."""
        client = EntsoeClient()

        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
            <TimeSeries>
                <Period>
                    <timeInterval>
                        <start>2025-01-01T00:00:00Z</start>
                    </timeInterval>
                    <resolution>PT60M</resolution>
                    <Point>
                        <position>1</position>
                        <price.amount>50.0</price.amount>
                    </Point>
                </Period>
            </TimeSeries>
        </Publication_MarketDocument>
        """
        mock_get.return_value = mock_response

        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 2)

        df = client._fetch_data("A44", "day_ahead", start_date, end_date, chunk_days=1)

        assert not df.empty
        assert mock_get.called

    @patch("app.services.entsoe_client.requests.get")
    def test_fetch_data_http_error(self, mock_get, mock_env_vars):
        """Test handling of HTTP errors."""
        client = EntsoeClient()

        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("Not found")
        mock_get.return_value = mock_response

        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 2)

        df = client._fetch_data("A44", "day_ahead", start_date, end_date, chunk_days=1)

        # Should return empty DataFrame on error
        assert df.empty
