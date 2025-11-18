"""
Data persistence layer for market and weather data.

Handles saving and loading of DataFrames to/from CSV files in the data directory.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from app.core.config import MARKET_TYPES, WEATHER_CONFIG, get_data_path
from app.core.logging import get_logger

logger = get_logger(__name__)


def save_market_data(df: pd.DataFrame, market_type: str) -> bool:
    """
    Save market data DataFrame to CSV.

    Args:
        df: DataFrame with market data
        market_type: Type of market ('day_ahead', 'intraday', 'imbalance')

    Returns:
        True if successful, False otherwise
    """
    if market_type not in MARKET_TYPES:
        logger.error(f"Invalid market type: {market_type}")
        return False

    filename = MARKET_TYPES[market_type]["data_file"]
    filepath = get_data_path(filename)

    try:
        # Ensure timestamp column is in proper format
        if "timestamp_utc" in df.columns:
            df = df.copy()
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} {market_type} records to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving {market_type} data: {e}")
        return False


def load_market_data(market_type: str) -> Optional[pd.DataFrame]:
    """
    Load market data DataFrame from CSV.

    Args:
        market_type: Type of market ('day_ahead', 'intraday', 'imbalance')

    Returns:
        DataFrame with market data or None if file doesn't exist
    """
    if market_type not in MARKET_TYPES:
        logger.error(f"Invalid market type: {market_type}")
        return None

    filename = MARKET_TYPES[market_type]["data_file"]
    filepath = get_data_path(filename)

    if not filepath.exists():
        logger.warning(f"Market data file not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)

        # Parse timestamp column
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

        logger.info(f"Loaded {len(df)} {market_type} records from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading {market_type} data: {e}")
        return None


def append_market_data(df_new: pd.DataFrame, market_type: str) -> bool:
    """
    Append new market data to existing CSV, removing duplicates.

    Args:
        df_new: New DataFrame with market data to append
        market_type: Type of market ('day_ahead', 'intraday', 'imbalance')

    Returns:
        True if successful, False otherwise
    """
    # Load existing data
    df_existing = load_market_data(market_type)

    if df_existing is None or df_existing.empty:
        # No existing data, just save new data
        return save_market_data(df_new, market_type)

    try:
        # Combine and remove duplicates
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        # Remove duplicates based on timestamp
        df_combined = df_combined.drop_duplicates(subset=["timestamp_utc"])

        # Sort by timestamp
        df_combined = df_combined.sort_values("timestamp_utc").reset_index(drop=True)

        # Save combined data
        return save_market_data(df_combined, market_type)
    except Exception as e:
        logger.error(f"Error appending {market_type} data: {e}")
        return False


def save_weather_data(df: pd.DataFrame) -> bool:
    """
    Save weather data DataFrame to CSV.

    Args:
        df: DataFrame with weather data

    Returns:
        True if successful, False otherwise
    """
    filename = WEATHER_CONFIG["data_file"]
    filepath = get_data_path(filename)

    try:
        # Ensure timestamp column is in proper format
        if "timestamp_utc" in df.columns:
            df = df.copy()
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} weather records to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving weather data: {e}")
        return False


def load_weather_data() -> Optional[pd.DataFrame]:
    """
    Load weather data DataFrame from CSV.

    Returns:
        DataFrame with weather data or None if file doesn't exist
    """
    filename = WEATHER_CONFIG["data_file"]
    filepath = get_data_path(filename)

    if not filepath.exists():
        logger.warning(f"Weather data file not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)

        # Parse timestamp column
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        
        # Rename columns to match expected format (support both old and new column names)
        column_mapping = {
            "temperature_celsius": "temperature_deg_c",
            "wind_speed_ms": "wind_speed_m_per_s",
            "cloud_cover_oktas": "cloud_cover_oktas",
            "precipitation_mm": "precipitation_mm"
        }
        df = df.rename(columns=column_mapping)
        
        # For backward compatibility, if global_radiation doesn't exist, use cloud_cover as proxy
        if "global_radiation_w_per_m2" not in df.columns and "cloud_cover_oktas" in df.columns:
            # Invert cloud cover to estimate radiation (8 oktas = fully cloudy = low radiation)
            df["global_radiation_w_per_m2"] = (8 - df["cloud_cover_oktas"]) * 50  # Simple proxy

        logger.info(f"Loaded {len(df)} weather records from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading weather data: {e}")
        return None


def append_weather_data(df_new: pd.DataFrame) -> bool:
    """
    Append new weather data to existing CSV, removing duplicates.

    Args:
        df_new: New DataFrame with weather data to append

    Returns:
        True if successful, False otherwise
    """
    # Load existing data
    df_existing = load_weather_data()

    if df_existing is None or df_existing.empty:
        # No existing data, just save new data
        return save_weather_data(df_new)

    try:
        # Combine and remove duplicates
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        # Remove duplicates based on timestamp
        df_combined = df_combined.drop_duplicates(subset=["timestamp_utc"])

        # Sort by timestamp
        df_combined = df_combined.sort_values("timestamp_utc").reset_index(drop=True)

        # Save combined data
        return save_weather_data(df_combined)
    except Exception as e:
        logger.error(f"Error appending weather data: {e}")
        return False


def get_data_summary() -> dict:
    """
    Get a summary of available data.

    Returns:
        Dictionary with data availability information
    """
    summary = {"markets": {}, "weather": {}}

    # Check market data
    for market_type in MARKET_TYPES.keys():
        df = load_market_data(market_type)
        if df is not None and not df.empty:
            summary["markets"][market_type] = {
                "records": len(df),
                "start_date": df["timestamp_utc"].min().isoformat(),
                "end_date": df["timestamp_utc"].max().isoformat(),
            }
        else:
            summary["markets"][market_type] = {"records": 0}

    # Check weather data
    df_weather = load_weather_data()
    if df_weather is not None and not df_weather.empty:
        summary["weather"] = {
            "records": len(df_weather),
            "start_date": df_weather["timestamp_utc"].min().isoformat(),
            "end_date": df_weather["timestamp_utc"].max().isoformat(),
        }
    else:
        summary["weather"] = {"records": 0}

    return summary
