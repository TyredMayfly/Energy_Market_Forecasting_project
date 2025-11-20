"""
Time-alignment and timezone validation utilities for data sources.

This module provides centralized validation for ensuring all data sources
(ENTSO-E day-ahead, imbalance, KNMI weather, Meteosource forecast) have:
- Consistent timezone handling (UTC for market data, configurable for weather)
- Aligned start and end dates across all sources
- Timezone-aware timestamps (no naive datetimes)

Raises clear ValueError exceptions when validation fails.
"""

from typing import List, Optional, Tuple

import pandas as pd
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class TimezoneValidationError(ValueError):
    """Raised when timezone validation fails."""

    pass


class DateRangeValidationError(ValueError):
    """Raised when date range validation fails."""

    pass


def validate_timezones(
    *dfs: pd.DataFrame,
    expected_tz: str = "UTC",
    timestamp_cols: Optional[List[str]] = None,
) -> None:
    """
    Validate that all DataFrames have timezone-aware timestamps with the expected timezone.

    Args:
        *dfs: Variable number of DataFrames to validate
        expected_tz: Expected timezone string (e.g., 'UTC', 'Europe/Amsterdam')
        timestamp_cols: List of column names to check (if None, checks index)

    Raises:
        TimezoneValidationError: If any DataFrame has incorrect or missing timezone
    """
    if not dfs:
        return

    for i, df in enumerate(dfs):
        if df is None or df.empty:
            continue

        # Determine what to check: index or specific columns
        if timestamp_cols:
            for col in timestamp_cols:
                if col not in df.columns:
                    raise TimezoneValidationError(
                        f"DataFrame {i}: Column '{col}' not found. Available columns: {df.columns.tolist()}"
                    )
                _validate_datetime_series_tz(df[col], expected_tz, f"DataFrame {i}, column '{col}'")
        else:
            # Check the index
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TimezoneValidationError(
                    f"DataFrame {i}: Index is not a DatetimeIndex (got {type(df.index).__name__})"
                )
            _validate_datetimeindex_tz(df.index, expected_tz, f"DataFrame {i} index")


def _validate_datetime_series_tz(series: pd.Series, expected_tz: str, label: str) -> None:
    """Validate timezone of a datetime Series."""
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise TimezoneValidationError(f"{label}: Not a datetime type (got {series.dtype})")

    # Check if timezone-aware
    if series.dt.tz is None:
        raise TimezoneValidationError(
            f"{label}: Timestamps are timezone-naive. "
            f"All timestamps must be timezone-aware. "
            f"Use pd.to_datetime(..., utc=True) or .tz_localize('{expected_tz}')"
        )

    # Normalize timezone name for comparison
    actual_tz = str(series.dt.tz)
    if actual_tz != expected_tz and not _timezones_equivalent(actual_tz, expected_tz):
        raise TimezoneValidationError(
            f"{label}: Expected timezone '{expected_tz}' but got '{actual_tz}'. "
            f"Use .tz_convert('{expected_tz}') to convert."
        )


def _validate_datetimeindex_tz(index: pd.DatetimeIndex, expected_tz: str, label: str) -> None:
    """Validate timezone of a DatetimeIndex."""
    if index.tz is None:
        raise TimezoneValidationError(
            f"{label}: Timestamps are timezone-naive. "
            f"All timestamps must be timezone-aware. "
            f"Use .tz_localize('{expected_tz}') or pd.to_datetime(..., utc=True)"
        )

    actual_tz = str(index.tz)
    if actual_tz != expected_tz and not _timezones_equivalent(actual_tz, expected_tz):
        raise TimezoneValidationError(
            f"{label}: Expected timezone '{expected_tz}' but got '{actual_tz}'. "
            f"Use .tz_convert('{expected_tz}') to convert."
        )


def _timezones_equivalent(tz1: str, tz2: str) -> bool:
    """Check if two timezone strings represent the same timezone."""
    # Handle common equivalences
    utc_variants = {"UTC", "UTC+00:00", "utc"}
    if tz1 in utc_variants and tz2 in utc_variants:
        return True
    return tz1 == tz2


def validate_start_end_dates(
    *dfs: pd.DataFrame,
    timestamp_cols: Optional[List[str]] = None,
    tolerance_seconds: int = 0,
) -> None:
    """
    Validate that all DataFrames share the same start and end dates.

    Args:
        *dfs: Variable number of DataFrames to validate
        timestamp_cols: List of column names to check (if None, checks index)
        tolerance_seconds: Allow this many seconds of difference (default: 0 for exact match)

    Raises:
        DateRangeValidationError: If start or end dates don't match across DataFrames
    """
    if not dfs or len(dfs) < 2:
        return

    # Filter out None and empty DataFrames
    valid_dfs = [(i, df) for i, df in enumerate(dfs) if df is not None and not df.empty]
    if len(valid_dfs) < 2:
        return

    # Extract min and max timestamps from each DataFrame
    date_ranges: List[Tuple[int, pd.Timestamp, pd.Timestamp]] = []

    for i, df in valid_dfs:
        if timestamp_cols:
            # Use first available timestamp column
            col = None
            for c in timestamp_cols:
                if c in df.columns:
                    col = c
                    break
            if col is None:
                raise DateRangeValidationError(
                    f"DataFrame {i}: None of the timestamp columns {timestamp_cols} found"
                )
            min_ts = df[col].min()
            max_ts = df[col].max()
        else:
            # Use index
            if not isinstance(df.index, pd.DatetimeIndex):
                raise DateRangeValidationError(
                    f"DataFrame {i}: Index is not a DatetimeIndex for date range validation"
                )
            min_ts = df.index.min()
            max_ts = df.index.max()

        # Ensure timestamps are timezone-aware
        if min_ts.tz is None or max_ts.tz is None:
            raise DateRangeValidationError(
                f"DataFrame {i}: Cannot validate date ranges with timezone-naive timestamps"
            )

        date_ranges.append((i, min_ts, max_ts))

    # Compare all date ranges with the first one
    reference_idx, reference_min, reference_max = date_ranges[0]

    for i, min_ts, max_ts in date_ranges[1:]:
        # Convert to same timezone for comparison
        min_ts_cmp = min_ts.tz_convert(reference_min.tz)
        max_ts_cmp = max_ts.tz_convert(reference_max.tz)

        # Check start dates
        start_diff_seconds = abs((min_ts_cmp - reference_min).total_seconds())
        if start_diff_seconds > tolerance_seconds:
            raise DateRangeValidationError(
                f"Start date mismatch: DataFrame {reference_idx} starts at {reference_min}, "
                f"but DataFrame {i} starts at {min_ts} (difference: {start_diff_seconds:.0f} seconds). "
                f"All data sources must have the same start date."
            )

        # Check end dates
        end_diff_seconds = abs((max_ts_cmp - reference_max).total_seconds())
        if end_diff_seconds > tolerance_seconds:
            raise DateRangeValidationError(
                f"End date mismatch: DataFrame {reference_idx} ends at {reference_max}, "
                f"but DataFrame {i} ends at {max_ts} (difference: {end_diff_seconds:.0f} seconds). "
                f"All data sources must have the same end date."
            )


def ensure_utc_timestamps(df: pd.DataFrame, timestamp_col: str = "timestamp_utc") -> pd.DataFrame:
    """
    Ensure a DataFrame has UTC timezone-aware timestamps.

    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column (can be index if it's named)

    Returns:
        DataFrame with UTC timestamps

    Raises:
        ValueError: If timestamp column not found or invalid
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Handle both column and index cases
    is_index = isinstance(df.index, pd.DatetimeIndex) and df.index.name == timestamp_col

    if is_index:
        # Work with index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    else:
        # Work with column
        if timestamp_col not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_col}' not found. Available: {df.columns.tolist()}"
            )

        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        if df[timestamp_col].dt.tz is None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize("UTC")
        else:
            df[timestamp_col] = df[timestamp_col].dt.tz_convert("UTC")

    return df


def ensure_local_timestamps(
    df: pd.DataFrame, timestamp_col: str = "timestamp", tz: Optional[str] = None
) -> pd.DataFrame:
    """
    Ensure a DataFrame has local timezone-aware timestamps.

    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column (can be index if it's named)
        tz: Timezone string (defaults to settings.timezone)

    Returns:
        DataFrame with local timezone timestamps

    Raises:
        ValueError: If timestamp column not found or invalid
    """
    if df is None or df.empty:
        return df

    if tz is None:
        tz = settings.timezone

    df = df.copy()

    # Handle both column and index cases
    is_index = isinstance(df.index, pd.DatetimeIndex) and df.index.name == timestamp_col

    if is_index:
        # Work with index
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)
    else:
        # Work with column
        if timestamp_col not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_col}' not found. Available: {df.columns.tolist()}"
            )

        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        if df[timestamp_col].dt.tz is None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(tz)
        else:
            df[timestamp_col] = df[timestamp_col].dt.tz_convert(tz)

    return df


def validate_and_align_market_weather_data(
    df_market: pd.DataFrame,
    df_weather: Optional[pd.DataFrame],
    market_timestamp_col: str = "timestamp_utc",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Validate and align market and weather data.

    Ensures:
    - Market data has UTC timestamps
    - Weather data (if provided) has timezone-aware timestamps
    - Both datasets cover the same date range (weather can be hourly, will be aligned)

    Args:
        df_market: Market data DataFrame
        df_weather: Weather data DataFrame (optional)
        market_timestamp_col: Name of timestamp column in market data

    Returns:
        Tuple of (validated_market_df, validated_weather_df)

    Raises:
        TimezoneValidationError: If timezone validation fails
        DateRangeValidationError: If date ranges don't align
    """
    if df_market is None or df_market.empty:
        return df_market, df_weather

    # Ensure market data is UTC
    df_market = ensure_utc_timestamps(df_market, market_timestamp_col)

    # Validate market data timezone
    validate_timezones(df_market, expected_tz="UTC", timestamp_cols=[market_timestamp_col])

    if df_weather is None or df_weather.empty:
        return df_market, df_weather

    # Ensure weather data has timezone (assume UTC if index, otherwise use config)
    if isinstance(df_weather.index, pd.DatetimeIndex):
        if df_weather.index.tz is None:
            df_weather = df_weather.copy()
            df_weather.index = df_weather.index.tz_localize("UTC")
        validate_timezones(df_weather, expected_tz="UTC")
    elif "timestamp" in df_weather.columns:
        df_weather = ensure_utc_timestamps(df_weather, "timestamp")
        validate_timezones(df_weather, expected_tz="UTC", timestamp_cols=["timestamp"])

    # Note: We don't enforce exact date range matching for weather data
    # since it's typically hourly and will be forward-filled during merging.
    # The feature engineering layer handles the alignment.
    # However, we do check that weather data covers at least the market data range.

    market_min = df_market[market_timestamp_col].min()
    market_max = df_market[market_timestamp_col].max()

    if isinstance(df_weather.index, pd.DatetimeIndex):
        weather_min = df_weather.index.min()
        weather_max = df_weather.index.max()
    else:
        weather_min = df_weather["timestamp"].min()
        weather_max = df_weather["timestamp"].max()

    # Weather should cover the market data range (with some tolerance)
    if weather_min > market_min:
        logger.warning(
            f"Weather data starts later than market data "
            f"(weather: {weather_min}, market: {market_min})"
        )

    if weather_max < market_max:
        logger.warning(
            f"Weather data ends earlier than market data "
            f"(weather: {weather_max}, market: {market_max})"
        )

    return df_market, df_weather
