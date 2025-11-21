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

    # Cumulative vs Instantaneous field mapping
    # KNMI hourly data has cumulative measurements over the PREVIOUS hour
    # Timestamp HH:00 represents cumulative total from (HH-1):00 to HH:00
    # These must be time-shifted backward by 30 minutes to place value at interval center
    CUMULATIVE_KNMI_FIELDS = {
        "Q",  # Global radiation (J/cm² over previous hour)
        "RH",  # Precipitation duration (0.1 hour over previous hour)
        "DR",  # Precipitation amount (0.1 mm over previous hour)
        "SQ",  # Sunshine duration (0.1 hour over previous hour)
        "EV24",  # Potential evapotranspiration (0.1 mm over previous 24h)
    }

    # Instantaneous/averaged fields (measured AT the timestamp, not cumulative)
    # These do NOT need time-shift
    INSTANTANEOUS_KNMI_FIELDS = {
        "T",  # Temperature (0.1 °C) - instantaneous
        "TD",  # Dew point (0.1 °C) - instantaneous
        "DD",  # Wind direction (degrees) - averaged over previous hour
        "FH",  # Wind speed (0.1 m/s) - averaged over previous hour
        "FF",  # Wind speed 10-min (0.1 m/s) - averaged
        "FX",  # Wind gust (0.1 m/s) - max over previous hour
        "N",  # Cloud cover (oktas) - instantaneous
        "U",  # Relative humidity (%) - instantaneous
        "P",  # Air pressure (0.1 hPa) - instantaneous
        "VV",  # Visibility (0-50=km, 51-80=5-30km coded, >89=<100m) - instantaneous
    }

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

            # Parse base date (KNMI timestamps are in UTC)
            base_date = pd.to_datetime(date_str, format="%Y%m%d", utc=True)

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
        - T: Temperature (0.1 °C) - instantaneous
        - FH: Hourly mean wind speed (0.1 m/s) - averaged over previous hour
        - Q: Global radiation (J/cm²/h) - cumulative over previous hour
        - N: Cloud cover (oktas, 0-8, 9=invisible) - instantaneous
        - RH: Precipitation (0.1 mm) - cumulative over previous hour

        IMPORTANT: Cumulative measurements (Q, RH) are time-shifted backward by 30 minutes
        to place the value at the center of the measurement interval, then interpolated
        to create smooth hourly series.

        Args:
            df: Raw KNMI DataFrame

        Returns:
            DataFrame with standardized columns and corrected timestamps
        """
        # Create separate DataFrames for cumulative and instantaneous fields
        df_instantaneous = pd.DataFrame()
        df_cumulative = pd.DataFrame()

        # Set timestamp as index for both
        df_instantaneous["timestamp"] = df["timestamp"]
        df_cumulative["timestamp"] = df["timestamp"]

        # === INSTANTANEOUS FIELDS (no time-shift needed) ===

        # Temperature: 0.1 °C → °C (instantaneous)
        if "T" in df.columns:
            df_instantaneous["temperature_deg_c"] = pd.to_numeric(df["T"], errors="coerce") / 10.0

        # Wind speed: 0.1 m/s → m/s (averaged over previous hour, but treated as instantaneous)
        if "FH" in df.columns:
            df_instantaneous["wind_speed_m_per_s"] = pd.to_numeric(df["FH"], errors="coerce") / 10.0

        # Cloud cover: oktas (0-8) → percentage (instantaneous)
        # 0 oktas = 0%, 1 okta = 12.5%, ..., 8 oktas = 100%
        if "N" in df.columns:
            # Convert to numeric, coercing errors to NaN
            cloud_oktas = pd.to_numeric(df["N"], errors="coerce")
            cloud_oktas = cloud_oktas.replace(9, float("nan"))  # 9 = sky invisible
            df_instantaneous["cloud_cover_pct"] = (cloud_oktas / 8.0 * 100).round(0)

        # === CUMULATIVE FIELDS (require time-shift + interpolation) ===

        # Global radiation: J/cm²/h → W/m² (cumulative over previous hour)
        # 1 J/cm²/h = 10000 J/m²/h = 10000/3600 W/m² ≈ 2.78 W/m²
        # Convert cumulative J to average W by dividing by 3600 seconds
        if "Q" in df.columns:
            df_cumulative["global_radiation_w_per_m2"] = (
                pd.to_numeric(df["Q"], errors="coerce") * 10000 / 3600
            )

        # Precipitation: 0.1 mm → mm (cumulative over previous hour)
        if "RH" in df.columns:
            # Convert to numeric, coercing errors to NaN
            precip = pd.to_numeric(df["RH"], errors="coerce")
            precip = precip.replace(-1, 0)  # -1 means <0.05 mm
            df_cumulative["precipitation_mm"] = precip / 10.0

        # === TIME-SHIFT CORRECTION FOR CUMULATIVE FIELDS ===
        # KNMI timestamps represent END of measurement interval
        # Shift backward by 30 minutes to place value at interval center
        if not df_cumulative.empty and len(df_cumulative.columns) > 1:  # More than just timestamp
            df_cumulative = df_cumulative.set_index("timestamp")
            # Shift index backward by 30 minutes
            df_cumulative.index = df_cumulative.index - pd.Timedelta(minutes=30)
            logger.info(f"Applied 30-minute time-shift to {len(df_cumulative.columns)} cumulative fields")

            # Interpolate to create smooth hourly series
            # Strategy: Create hourly range, reindex to include both shifted and hourly points, then interpolate
            # This preserves the shifted values while creating hourly grid points
            start_hour = df_cumulative.index.min().floor("1h")
            end_hour = df_cumulative.index.max().ceil("1h")
            hourly_index = pd.date_range(start=start_hour, end=end_hour, freq="1h")
            
            # Reindex to include both original shifted times and hourly grid
            combined_index = df_cumulative.index.union(hourly_index).sort_values()
            df_cumulative = df_cumulative.reindex(combined_index)
            
            # Interpolate linearly between points
            df_cumulative = df_cumulative.interpolate(method="linear", limit_direction="both")
            
            # Keep only the hourly grid points
            df_cumulative = df_cumulative.reindex(hourly_index)
            
            logger.info(f"Interpolated cumulative fields to hourly resolution: {len(df_cumulative)} records")
        else:
            # No cumulative fields present
            df_cumulative = df_cumulative.set_index("timestamp") if "timestamp" in df_cumulative.columns else pd.DataFrame()

        # === MERGE INSTANTANEOUS AND CUMULATIVE FIELDS ===
        # Check if we have actual data (more than just timestamp column)
        has_instantaneous = not df_instantaneous.empty and len(df_instantaneous.columns) > 1
        has_cumulative = not df_cumulative.empty and isinstance(df_cumulative.index, pd.DatetimeIndex)

        if has_instantaneous:
            df_instantaneous = df_instantaneous.set_index("timestamp")

        # Combine both DataFrames
        if has_cumulative and has_instantaneous:
            # Merge on index (timestamp)
            result = df_instantaneous.join(df_cumulative, how="outer")
            # Forward-fill and back-fill to ensure complete coverage
            result = result.ffill().bfill()
        elif has_instantaneous:
            result = df_instantaneous
        elif has_cumulative:
            result = df_cumulative
        else:
            result = pd.DataFrame()

        # Add metadata
        if not result.empty:
            result["data_type"] = "historical"
            result["fetch_timestamp"] = datetime.now()

        logger.info(f"✓ Converted {len(result)} records to standard format")
        logger.info(f"  Columns: {', '.join(result.columns)}")
        return result

    def fetch_historical_for_2025(self, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch all historical hourly data from Oct 1, 2024 to specified end date.

        Args:
            end_date: End date (default: now)

        Returns:
            DataFrame with standardized hourly weather data
        """
        start = datetime(2024, 10, 1, 1, tzinfo=pd.Timestamp.now(tz="UTC").tzinfo)  # Oct 1, 2024, 01:00 UTC
        end = end_date or pd.Timestamp.now(tz="UTC").to_pydatetime()

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
