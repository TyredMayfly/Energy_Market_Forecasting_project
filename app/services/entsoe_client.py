"""
ENTSO-E Transparency Platform API client.

This module provides a client for fetching electricity market data from the
ENTSO-E Transparency Platform for the Netherlands bidding zone.

Supports:
- Day-ahead market prices (document type A44)
- Intraday market prices (document type A45)
- Imbalance-related data (document type A53)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EntsoeClient:
    """Client for interacting with the ENTSO-E Transparency Platform API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ENTSO-E client.

        Args:
            api_key: ENTSO-E API key. If None, uses settings.entsoe_api_key
        """
        self.api_key = api_key if api_key is not None else settings.entsoe_api_key
        self.base_url = settings.entsoe_base_url
        self.netherlands_eic = settings.netherlands_eic_code

        # Only validate if api_key was not explicitly provided (even if empty for testing)
        if api_key is None and not self.api_key:
            raise ValueError("ENTSO-E API key is required. Set ENTSOE_API_KEY in .env file.")

        # XML namespaces used in ENTSO-E responses
        self.namespaces = {
            "ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0",
        }

    def _build_query_url(
        self,
        document_type: str,
        period_start: datetime,
        period_end: datetime,
        process_type: Optional[str] = None,
    ) -> str:
        """
        Build ENTSO-E API query URL.

        Args:
            document_type: ENTSO-E document type (e.g., 'A44', 'A45', 'A53')
            period_start: Start datetime in UTC
            period_end: End datetime in UTC
            process_type: Optional process type code

        Returns:
            Complete query URL
        """
        # Format timestamps as YYYYMMDDHHMM
        start_str = period_start.strftime("%Y%m%d%H%M")
        end_str = period_end.strftime("%Y%m%d%H%M")

        params = {
            "securityToken": self.api_key,
            "documentType": document_type,
            "in_Domain": self.netherlands_eic,
            "out_Domain": self.netherlands_eic,
            "periodStart": start_str,
            "periodEnd": end_str,
        }

        if process_type:
            params["processType"] = process_type

        # Build query string
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.base_url}?{query_string}"

    def _parse_price_document(self, xml_content: str, market_type: str) -> pd.DataFrame:
        """
        Parse ENTSO-E price document XML into a DataFrame.

        Args:
            xml_content: XML response content
            market_type: Market type label (e.g., 'day_ahead', 'intraday')

        Returns:
            DataFrame with columns: timestamp_utc, price_eur_per_mwh, market_type
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            return pd.DataFrame(columns=["timestamp_utc", "price_eur_per_mwh", "market_type"])

        records = []

        # Find all TimeSeries elements
        for timeseries in root.findall(".//ns:TimeSeries", self.namespaces):
            # Get the period
            period = timeseries.find(".//ns:Period", self.namespaces)
            if period is None:
                continue

            # Get time interval start
            time_interval = period.find("ns:timeInterval", self.namespaces)
            if time_interval is None:
                continue

            start_elem = time_interval.find("ns:start", self.namespaces)
            if start_elem is None or start_elem.text is None:
                continue

            # Parse start time
            period_start = pd.to_datetime(start_elem.text)

            # Get resolution (e.g., PT60M for 60 minutes)
            resolution_elem = period.find("ns:resolution", self.namespaces)
            if resolution_elem is None or resolution_elem.text is None:
                resolution_minutes = 60  # Default to hourly
            else:
                # Parse PT60M -> 60
                res_text = resolution_elem.text
                if res_text.startswith("PT") and res_text.endswith("M"):
                    resolution_minutes = int(res_text[2:-1])
                else:
                    resolution_minutes = 60

            # Parse all points
            for point in period.findall("ns:Point", self.namespaces):
                position_elem = point.find("ns:position", self.namespaces)
                price_elem = point.find("ns:price.amount", self.namespaces)

                if position_elem is None or price_elem is None:
                    continue

                if position_elem.text is None or price_elem.text is None:
                    continue

                position = int(position_elem.text)
                price = float(price_elem.text)

                # Calculate timestamp (position is 1-indexed)
                timestamp = period_start + timedelta(minutes=(position - 1) * resolution_minutes)

                records.append(
                    {
                        "timestamp_utc": timestamp,
                        "price_eur_per_mwh": price,
                        "market_type": market_type,
                    }
                )

        if not records:
            logger.warning(f"No price data found in XML for {market_type}")
            return pd.DataFrame(columns=["timestamp_utc", "price_eur_per_mwh", "market_type"])

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp_utc").reset_index(drop=True)

        logger.info(f"Parsed {len(df)} {market_type} price records")
        return df

    def _fetch_data(
        self,
        document_type: str,
        market_type: str,
        start_date: datetime,
        end_date: datetime,
        chunk_days: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch data from ENTSO-E API, chunking requests if needed.

        Args:
            document_type: ENTSO-E document type
            market_type: Market type label
            start_date: Start date in UTC
            end_date: End date in UTC
            chunk_days: Number of days per API request

        Returns:
            Combined DataFrame with all fetched data
        """
        all_data = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)

            logger.info(
                f"Fetching {market_type} data from {current_start.date()} to {current_end.date()}"
            )

            url = self._build_query_url(document_type, current_start, current_end)

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                df_chunk = self._parse_price_document(response.text, market_type)
                if not df_chunk.empty:
                    all_data.append(df_chunk)

            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error fetching {market_type} data: {e}")
                if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                    logger.warning("Rate limit hit, waiting before retry...")
                    import time

                    time.sleep(5)
                    continue
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error fetching {market_type} data: {e}")
            except Exception as e:
                logger.error(f"Unexpected error parsing {market_type} data: {e}")

            current_start = current_end

        if not all_data:
            logger.warning(f"No {market_type} data fetched")
            return pd.DataFrame(columns=["timestamp_utc", "price_eur_per_mwh", "market_type"])

        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
        df = df.reset_index(drop=True)

        logger.info(f"Fetched total of {len(df)} {market_type} records")
        return df

    def fetch_day_ahead_prices_2025_nl(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch day-ahead market prices for the Netherlands in 2025.

        Args:
            start_date: Start date (default: 2025-01-01)
            end_date: End date (default: today)

        Returns:
            DataFrame with day-ahead prices
        """
        if start_date is None:
            start_date = datetime(2025, 1, 1, 0, 0, 0)
        if end_date is None:
            end_date = datetime.utcnow()

        logger.info("Fetching day-ahead prices for Netherlands")
        return self._fetch_data("A44", "day_ahead", start_date, end_date)

    def fetch_intraday_prices_2025_nl(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch intraday market prices for the Netherlands in 2025.

        Args:
            start_date: Start date (default: 2025-01-01)
            end_date: End date (default: today)

        Returns:
            DataFrame with intraday prices
        """
        if start_date is None:
            start_date = datetime(2025, 1, 1, 0, 0, 0)
        if end_date is None:
            end_date = datetime.utcnow()

        logger.info("Fetching intraday prices for Netherlands")
        return self._fetch_data("A45", "intraday", start_date, end_date)

    def fetch_imbalance_data_2025_nl(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch imbalance-related data for the Netherlands in 2025.

        Note: Document type A53 may contain various imbalance-related information.
        Check ENTSO-E documentation for exact interpretation.

        Args:
            start_date: Start date (default: 2025-01-01)
            end_date: End date (default: today)

        Returns:
            DataFrame with imbalance data
        """
        if start_date is None:
            start_date = datetime(2025, 1, 1, 0, 0, 0)
        if end_date is None:
            end_date = datetime.utcnow()

        logger.info("Fetching imbalance data for Netherlands")
        return self._fetch_data("A53", "imbalance", start_date, end_date)
