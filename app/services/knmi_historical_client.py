"""
KNMI Historical Weather Data Client

Fetches historical hourly weather data from KNMI (Royal Netherlands Meteorological Institute).
Free API for historical data with hourly measurements.

API Documentation: https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script
Interactive Selection: https://daggegevens.knmi.nl/klimatologie/uurgegevens
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KNMIHistoricalClient:
    """Client for fetching historical hourly weather data from KNMI API."""

    BASE_URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"

    # Station 240 = Schiphol Airport (closest to Amsterdam)
    DEFAULT_STATION = "240"

    # Variable groups for KNMI API
    # TEMP = T:TD (Temperature and dew point)
    # WIND = DD:FH:FF:FX (Wind direction, avg speed, avg speed 10min, max gust)
    # SUNR = SQ:Q (Sunshine duration and global radiation)
    # PRCP = DR:RH (Precipitation duration and amount)
    # VICL = VV:N:U (Visibility, cloud cover, humidity)
    # ALL = all variables

    def __init__(self, station: str = DEFAULT_STATION):
        """
        Initialize KNMI Historical Client.

        Args:
            station: KNMI station number (default: 240 = Schiphol)
        """
        self.station = station

    def fetch_hourly_data(
        self, start_date: datetime, end_date: datetime, variables: str = "TEMP:WIND:SUNR:PRCP:VICL"
    ) -> pd.DataFrame:
        """
        Fetch hourly historical weather data from KNMI.

        Args:
            start_date: Start datetime (inclusive)
            end_date: End datetime (inclusive)
            variables: Colon-separated variable groups or individual variables
                      (e.g., "TEMP:WIND:SUNR" or "T:DD:Q")

        Returns:
            DataFrame with hourly weather data
        """
        # Format dates for KNMI API (YYYYMMDDHH)
        start_str = start_date.strftime("%Y%m%d%H")
        end_str = end_date.strftime("%Y%m%d%H")

        logger.info(f"Fetching KNMI historical data for station {self.station}")
        logger.info(f"  Period: {start_date} to {end_date}")
        logger.info(f"  Variables: {variables}")

        # Build POST data
        post_data = {"start": start_str, "end": end_str, "vars": variables, "stns": self.station}

        try:
            response = requests.post(self.BASE_URL, data=post_data, timeout=30)
            response.raise_for_status()

            # Parse CSV response
            df = self._parse_knmi_response(response.text)

            logger.info(f"✓ Received {len(df)} hourly records from KNMI")
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to fetch KNMI data: {e}")
            raise

    def _parse_knmi_response(self, response_text: str) -> pd.DataFrame:
        """
        Parse KNMI CSV response into DataFrame.

        The response has:
        - Header lines starting with '#'
        - Column header line: # STN,YYYYMMDD,HH,var1,var2,...
        - Data lines with format: STN,YYYYMMDD,HH,var1,var2,...

        Args:
            response_text: Raw CSV response from KNMI

        Returns:
            Parsed DataFrame
        """
        lines = response_text.strip().split("\n")

        # Find column header line (starts with # STN,YYYYMMDD,HH)
        header_line = None
        data_start_idx = 0

        for i, line in enumerate(lines):
            if line.startswith("# STN,YYYYMMDD,HH"):
                header_line = line[2:].strip()  # Remove '# '
                data_start_idx = i + 1
                break

        if not header_line:
            raise ValueError("Could not find column header in KNMI response")

        # Parse header - comma separated, remove extra spaces
        columns = [col.strip() for col in header_line.split(",")]

        # Parse data lines (skip comment lines)
        data_lines = [line for line in lines[data_start_idx:] if not line.startswith("#")]

        # Create DataFrame
        data = []
        for line in data_lines:
            if line.strip():
                values = [val.strip() for val in line.split(",")]
                data.append(values)

        df = pd.DataFrame(data, columns=columns)

        # Convert numeric columns
        for col in df.columns:
            if col not in ["STN"]:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except:
                    pass

        # Create timestamp from YYYYMMDD and HH
        # KNMI uses hour 1-24 (24 = midnight of next day)
        # Convert to standard 0-23 format
        timestamps = []
        for _, row in df.iterrows():
            date_str = str(int(row["YYYYMMDD"]))
            hour = int(row["HH"])

            # Parse base date
            base_date = pd.to_datetime(date_str, format="%Y%m%d")

            # Handle hour 24 (midnight of next day)
            if hour == 24:
                timestamp = base_date + timedelta(days=1)
            else:
                timestamp = base_date + timedelta(hours=hour)

            timestamps.append(timestamp)

        df["timestamp"] = timestamps

        return df

    def convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert KNMI data to standard format matching Meteosource.

        KNMI variables (subset matching Meteosource free tier):
        - T: Temperature (0.1 °C)
        - FH: Hourly mean wind speed (0.1 m/s)
        - Q: Global radiation (J/cm²/h)
        - N: Cloud cover (oktas, 0-8, 9=invisible)
        - RH: Precipitation (0.1 mm)

        Args:
            df: Raw KNMI DataFrame

        Returns:
            DataFrame with standardized columns
        """
        result = pd.DataFrame()
        result["timestamp"] = df["timestamp"]

        # Temperature: 0.1 °C → °C
        if "T" in df.columns:
            result["temperature_deg_c"] = df["T"] / 10.0

        # Wind speed: 0.1 m/s → m/s
        if "FH" in df.columns:
            result["wind_speed_m_per_s"] = df["FH"] / 10.0

        # Global radiation: J/cm²/h → W/m²
        # 1 J/cm²/h = 10000 J/m²/h = 10000/3600 W/m² ≈ 2.78 W/m²
        if "Q" in df.columns:
            result["global_radiation_w_per_m2"] = df["Q"] * 10000 / 3600

        # Cloud cover: oktas (0-8) → percentage
        # 0 oktas = 0%, 1 okta = 12.5%, ..., 8 oktas = 100%
        if "N" in df.columns:
            cloud_oktas = df["N"].replace(9, pd.NA)  # 9 = sky invisible
            result["cloud_cover_pct"] = (cloud_oktas / 8.0 * 100).round(0)

        # Precipitation: 0.1 mm → mm
        if "RH" in df.columns:
            precip = df["RH"].replace(-1, 0)  # -1 means <0.05 mm
            result["precipitation_mm"] = precip / 10.0

        # Add metadata
        result["data_type"] = "historical"
        result["fetch_timestamp"] = datetime.now()

        result.set_index("timestamp", inplace=True)

        logger.info(f"✓ Converted {len(result)} records to standard format")
        return result

    def fetch_historical_for_2025(self, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch all historical hourly data from Jan 1, 2025 to specified end date.

        Args:
            end_date: End date (default: now)

        Returns:
            DataFrame with standardized hourly weather data
        """
        start = datetime(2025, 1, 1, 1)  # Jan 1, 2025, 01:00
        end = end_date or datetime.now()

        # Fetch raw data (exclude U:P to match Meteosource free tier)
        raw_df = self.fetch_hourly_data(start, end, variables="T:FH:Q:N:RH")

        # Convert to standard format
        standard_df = self.convert_to_standard_format(raw_df)

        return standard_df


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    client = KNMIHistoricalClient()

    # Fetch data from Jan 1, 2025 to now
    df = client.fetch_historical_for_2025()

    print(f"\n✓ Fetched {len(df)} hourly records")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nFirst 5 records:")
    print(df.head())
    print(f"\nLast 5 records:")
    print(df.tail())
    print(f"\nColumns: {', '.join(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)
