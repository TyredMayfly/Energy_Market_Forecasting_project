"""
KNMI Open Data API client.

This module provides a client for fetching meteorological data from the
KNMI (Royal Netherlands Meteorological Institute) Open Data Platform.

Fetches weather observations including temperature, wind speed, and
solar radiation for forecasting purposes.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import tempfile

import pandas as pd
import requests
import xarray as xr
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class KnmiClient:
    """Client for interacting with the KNMI Open Data Platform API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the KNMI client.

        Args:
            api_key: KNMI API key. If None, uses settings.knmi_api_key
        """
        self.api_key = api_key if api_key is not None else settings.knmi_api_key
        self.base_url = settings.knmi_base_url
        self.dataset_name = settings.knmi_dataset_name
        self.dataset_version = settings.knmi_dataset_version

        if not self.api_key:
            raise ValueError("KNMI API key is required. Set KNMI_API_KEY in .env file.")

        self.headers = {"Authorization": self.api_key}

    def _list_files(
        self,
        max_keys: int = 500,
        start_after_filename: Optional[str] = None,
    ) -> List[Dict]:
        """
        List files available in the KNMI dataset.

        Args:
            max_keys: Maximum number of files to return
            start_after_filename: Start listing after this filename

        Returns:
            List of file metadata dictionaries
        """
        url = f"{self.base_url}/datasets/{self.dataset_name}/versions/{self.dataset_version}/files"

        params = {"maxKeys": max_keys}
        if start_after_filename:
            params["startAfterFilename"] = start_after_filename

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            files = data.get("files", [])
            logger.info(f"Listed {len(files)} files from KNMI dataset")
            return files
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing KNMI files: {e}")
            return []

    def _get_download_url(self, filename: str) -> Optional[str]:
        """
        Get temporary download URL for a specific file.

        Args:
            filename: Name of the file to download

        Returns:
            Temporary download URL or None if error
        """
        url = f"{self.base_url}/datasets/{self.dataset_name}/versions/{self.dataset_version}/files/{filename}/url"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            download_url = data.get("temporaryDownloadUrl")
            logger.debug(f"Got download URL for {filename}")
            return download_url
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting download URL for {filename}: {e}")
            return None

    def _download_file(self, download_url: str, local_path: Path) -> bool:
        """
        Download a file from KNMI to local storage.

        Args:
            download_url: Temporary download URL
            local_path: Local path to save the file

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(download_url, timeout=60, stream=True)
            response.raise_for_status()

            local_path.parent.mkdir(parents=True, exist_ok=True)

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded file to {local_path}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading file: {e}")
            return False

    def _parse_netcdf_to_dataframe(self, netcdf_path: Path) -> pd.DataFrame:
        """
        Parse a NetCDF file into a pandas DataFrame.

        Extracts key meteorological variables relevant for forecasting.

        Args:
            netcdf_path: Path to the NetCDF file

        Returns:
            DataFrame with weather observations
        """
        try:
            ds = xr.open_dataset(netcdf_path)

            # Extract relevant variables
            records = []

            # Check available variables
            available_vars = list(ds.data_vars)
            logger.debug(f"Available variables in NetCDF: {available_vars}")

            # Common KNMI variable names (may vary by dataset)
            var_mapping = {
                "time": "time",
                "station": "station",
                "T": "temperature_deg_c",  # Temperature
                "T10": "temperature_deg_c",  # Temperature at 10m
                "FF": "wind_speed_m_per_s",  # Wind speed
                "FH": "wind_speed_m_per_s",  # Wind speed (alternative)
                "Q": "global_radiation_w_per_m2",  # Global radiation
                "SQ": "sunshine_duration_min",  # Sunshine duration
            }

            # Try to extract data
            if "time" in ds.coords or "time" in ds.dims:
                time_data = pd.to_datetime(ds["time"].values)

                # Get station IDs if available
                if "station" in ds.coords or "station" in ds.dims:
                    stations = ds["station"].values
                else:
                    stations = [None]

                for station in stations:
                    for var_knmi, var_standard in var_mapping.items():
                        if var_knmi in ds.data_vars:
                            if station is not None:
                                values = ds[var_knmi].sel(station=station).values
                            else:
                                values = ds[var_knmi].values

                            for i, timestamp in enumerate(time_data):
                                value = values[i] if i < len(values) else None

                                # Skip invalid values
                                if value is None or pd.isna(value):
                                    continue

                                records.append(
                                    {
                                        "timestamp_utc": pd.to_datetime(timestamp),
                                        "station_id": str(station) if station is not None else "unknown",
                                        "variable": var_standard,
                                        "value": float(value),
                                    }
                                )

            ds.close()

            if not records:
                logger.warning(f"No data extracted from {netcdf_path}")
                return pd.DataFrame(
                    columns=["timestamp_utc", "station_id", "variable", "value"]
                )

            df = pd.DataFrame(records)
            logger.info(f"Parsed {len(df)} records from {netcdf_path.name}")
            return df

        except Exception as e:
            logger.error(f"Error parsing NetCDF file {netcdf_path}: {e}")
            return pd.DataFrame(columns=["timestamp_utc", "station_id", "variable", "value"])

    def _filter_2025_files(self, files: List[Dict]) -> List[Dict]:
        """
        Filter files to only those from 2025.

        Args:
            files: List of file metadata

        Returns:
            Filtered list of 2025 files
        """
        files_2025 = []
        for file in files:
            filename = file.get("filename", "")
            # Check if filename contains '2025' or parse date if structured
            if "2025" in filename:
                files_2025.append(file)

        logger.info(f"Found {len(files_2025)} files from 2025")
        return files_2025

    def list_files_for_2025(self) -> List[Dict]:
        """
        List all available files from 2025 in the KNMI dataset.

        Returns:
            List of file metadata for 2025
        """
        logger.info("Listing KNMI files for 2025")
        all_files = self._list_files(max_keys=1000)
        return self._filter_2025_files(all_files)

    def download_and_process_files_for_2025(
        self, max_files: int = 50
    ) -> pd.DataFrame:
        """
        Download and process KNMI weather files for 2025.

        Args:
            max_files: Maximum number of files to download and process

        Returns:
            Combined DataFrame with weather observations
        """
        logger.info("Downloading and processing KNMI files for 2025")

        files_2025 = self.list_files_for_2025()

        if not files_2025:
            logger.warning("No 2025 files found")
            return pd.DataFrame(
                columns=["timestamp_utc", "station_id", "variable", "value"]
            )

        # Limit number of files
        files_to_process = files_2025[:max_files]
        logger.info(f"Processing {len(files_to_process)} files")

        all_data = []

        for file_info in files_to_process:
            filename = file_info.get("filename")
            if not filename:
                continue

            # Get download URL
            download_url = self._get_download_url(filename)
            if not download_url:
                continue

            # Download to temporary file
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / filename

                if self._download_file(download_url, tmp_path):
                    # Parse NetCDF
                    df = self._parse_netcdf_to_dataframe(tmp_path)
                    if not df.empty:
                        all_data.append(df)

        if not all_data:
            logger.warning("No weather data extracted from files")
            return pd.DataFrame(
                columns=["timestamp_utc", "station_id", "variable", "value"]
            )

        # Combine all data
        df_combined = pd.concat(all_data, ignore_index=True)

        # Pivot to wide format for easier use
        df_wide = df_combined.pivot_table(
            index=["timestamp_utc", "station_id"],
            columns="variable",
            values="value",
            aggfunc="mean",
        ).reset_index()

        # Average across all stations to get Netherlands-wide values
        df_aggregated = (
            df_wide.groupby("timestamp_utc")
            .agg(
                {
                    "temperature_deg_c": "mean",
                    "wind_speed_m_per_s": "mean",
                    "global_radiation_w_per_m2": "mean",
                }
            )
            .reset_index()
        )

        df_aggregated = df_aggregated.sort_values("timestamp_utc").reset_index(drop=True)

        logger.info(f"Processed total of {len(df_aggregated)} aggregated weather records")
        return df_aggregated
