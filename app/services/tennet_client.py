"""
TenneT Settlement Prices API Client

This module handles interactions with the TenneT settlement prices API.
API Spec: https://developer.tennet.eu/specs/v1/settlement-prices
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


class TennetApiError(Exception):
    """Exception raised for TenneT API errors."""

    pass


class TennetClient:
    """Client for interacting with the TenneT Settlement Prices API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 10,
    ):
        """
        Initialize TenneT API client.

        Args:
            base_url: Base URL for the API (defaults to settings)
            api_key: API key for authentication (defaults to settings)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or settings.tennet_settlement_base_url
        # Handle api_key: use provided value if explicitly set (even if empty),
        # otherwise fall back to settings
        self.api_key = api_key if api_key is not None else settings.tennet_api_key
        self.timeout = timeout

        if not self.api_key:
            logger.warning(
                "TenneT API key not configured. Set TENNET_API_KEY environment variable."
            )

    def _get_headers(self) -> dict:
        """
        Get HTTP headers for API requests.

        Returns:
            Dictionary of HTTP headers including authentication
        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        # Only add API key header if API key is present and non-empty
        if self.api_key and self.api_key.strip():
            # TenneT uses 'apikey' header as per OpenAPI spec
            headers["apikey"] = self.api_key

        return headers

    def _make_request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """
        Make an HTTP GET request to the TenneT API.

        Args:
            endpoint: API endpoint path
            params: Optional query parameters

        Returns:
            JSON response as dictionary

        Raises:
            TennetApiError: If request fails or returns error status
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            logger.debug(f"Making request to {url} with params: {params}")

            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.timeout,
            )

            # Raise exception for HTTP errors
            if response.status_code >= 400:
                error_msg = f"TenneT API error: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if "message" in error_data:
                        error_msg += f" - {error_data['message']}"
                except Exception:
                    error_msg += f" - {response.text[:200]}"

                raise TennetApiError(error_msg)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout as e:
            raise TennetApiError(f"Request timeout after {self.timeout}s: {e}")
        except requests.exceptions.ConnectionError as e:
            raise TennetApiError(f"Connection error: {e}")
        except requests.exceptions.RequestException as e:
            raise TennetApiError(f"Request failed: {e}")

    def _parse_settlement_data(self, data: dict) -> pd.DataFrame:
        """
        Parse settlement price data from API response into DataFrame.

        Args:
            data: JSON response from API

        Returns:
            DataFrame with standardized columns and timezone-aware timestamps (UTC)

        Raises:
            TennetApiError: If data parsing fails
        """
        try:
            # Parse the nested JSON structure from TenneT API
            # Structure: {"Response": {"TimeSeries": [{"Period": {"Points": [...]}}]}}
            if not isinstance(data, dict):
                logger.warning("API response is not a dictionary")
                return pd.DataFrame()
            
            # Navigate to the Points array
            response = data.get("Response", {})
            time_series = response.get("TimeSeries", [])
            
            if not time_series:
                logger.warning("No TimeSeries data in API response")
                return pd.DataFrame()
            
            # Collect all points from all time series
            all_points = []
            for series in time_series:
                period = series.get("Period", {})
                points = period.get("Points", [])
                all_points.extend(points)
            
            if not all_points:
                logger.warning("No Points data in API response")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(all_points)

            # Map API field names to our standard column names
            column_mapping = {
                "timeInterval_start": "timestamp_utc",
                "shortage": "shortage_price",
                "surplus": "surplus_price",
                "regulation_state": "regulation_state",
            }

            # Rename columns
            df = df.rename(columns=column_mapping)

            # Ensure required columns exist
            if "timestamp_utc" not in df.columns:
                raise TennetApiError("No timestamp column found in API response")

            # Parse timestamp and ensure it's timezone-aware UTC
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

            # Convert prices to numeric
            for price_col in ["shortage_price", "surplus_price"]:
                if price_col in df.columns:
                    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

            # Convert regulation_state to int
            if "regulation_state" in df.columns:
                df["regulation_state"] = pd.to_numeric(df["regulation_state"], errors="coerce").astype("Int64")

            # Select and order columns
            standard_cols = ["timestamp_utc", "shortage_price", "surplus_price", "regulation_state"]
            available_cols = [col for col in standard_cols if col in df.columns]
            df = df[available_cols]

            # Sort by timestamp
            df = df.sort_values("timestamp_utc").reset_index(drop=True)

            logger.info(
                f"Parsed {len(df)} settlement price records "
                f"({df['timestamp_utc'].min()} to {df['timestamp_utc'].max()})"
            )

            return df

        except Exception as e:
            raise TennetApiError(f"Failed to parse settlement price data: {e}")

    def fetch_latest_settlement_prices(
        self,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch the most recent settlement prices.

        Args:
            limit: Maximum number of records to fetch (default 100)

        Returns:
            DataFrame with settlement prices, sorted by timestamp
            Columns: timestamp_utc, shortage_price, surplus_price, regulation_state

        Raises:
            TennetApiError: If API call fails
        """
        logger.info(f"Fetching latest {limit} settlement prices from TenneT API")

        # Use the latest endpoint or query with recent date range
        # Based on API spec, typical endpoint is /settlement-prices/latest or /settlement-prices
        params = {"limit": limit}

        try:
            data = self._make_request("publications/v1/settlement-prices/latest", params=params)
            df = self._parse_settlement_data(data)

            logger.info(f"Successfully fetched {len(df)} latest settlement prices")
            return df

        except TennetApiError:
            raise
        except Exception as e:
            raise TennetApiError(f"Unexpected error fetching latest prices: {e}")

    def fetch_settlement_prices_from_to(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch settlement prices for a specific date range.

        Args:
            start: Start datetime (timezone-aware, will be converted to UTC)
            end: End datetime (timezone-aware, will be converted to UTC)

        Returns:
            DataFrame with settlement prices in the specified range
            Columns: timestamp_utc, shortage_price, surplus_price, regulation_state

        Raises:
            TennetApiError: If API call fails
        """
        # Ensure datetimes are timezone-aware and convert to UTC
        if start.tzinfo is None:
            raise ValueError("start datetime must be timezone-aware")
        if end.tzinfo is None:
            raise ValueError("end datetime must be timezone-aware")

        start_utc = start.astimezone(pd.Timestamp("2000-01-01").tz_localize("UTC").tzinfo)
        end_utc = end.astimezone(pd.Timestamp("2000-01-01").tz_localize("UTC").tzinfo)

        logger.info(f"Fetching settlement prices from {start_utc} to {end_utc}")

        # API has 1-hour maximum range per request, so we need to chunk the requests
        all_dfs = []
        current_start = start_utc
        
        while current_start < end_utc:
            # Calculate chunk end (max 1 hour from current start)
            chunk_end = min(current_start + timedelta(hours=1), end_utc)
            
            # Format dates for API (dd-mm-yyyy hh24:mi:ss format as per spec)
            params = {
                "date_from": current_start.strftime("%d-%m-%Y %H:%M:%S"),
                "date_to": chunk_end.strftime("%d-%m-%Y %H:%M:%S"),
            }
            
            logger.debug(f"Fetching chunk: {params['date_from']} to {params['date_to']}")
            
            try:
                data = self._make_request("publications/v1/settlement-prices", params=params)
                
                # Handle "no data" response
                if isinstance(data, dict) and data.get("error") == "No data found":
                    logger.debug(f"No data for chunk {params['date_from']} to {params['date_to']}")
                else:
                    df_chunk = self._parse_settlement_data(data)
                    if not df_chunk.empty:
                        all_dfs.append(df_chunk)
                        
            except TennetApiError as e:
                logger.warning(f"Error fetching chunk {params['date_from']} to {params['date_to']}: {e}")
            
            # Move to next chunk
            current_start = chunk_end
        
        # Combine all chunks
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
            logger.info(f"Successfully fetched {len(df)} settlement prices for date range")
            return df
        else:
            logger.warning("No data fetched for the requested range")
            return pd.DataFrame()


# Module-level instance for convenience
_client_instance = None


def get_tennet_client() -> TennetClient:
    """
    Get singleton TenneT client instance.

    Returns:
        Configured TennetClient instance
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = TennetClient()
    return _client_instance
