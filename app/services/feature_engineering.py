"""
Feature engineering for power market forecasting.

Builds feature matrices (X) and target arrays (y) from market and weather data.
Includes lag features, time features, and weather variables.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from app.core.config import MARKET_TYPES, settings
from app.core.logging import get_logger
from app.services.data_store import load_market_data, load_weather_data

logger = get_logger(__name__)


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from timestamp.

    Args:
        df: DataFrame with 'timestamp_utc' column

    Returns:
        DataFrame with added time features
    """
    df = df.copy()

    # Extract time components
    df["hour_of_day"] = df["timestamp_utc"].dt.hour
    df["day_of_week"] = df["timestamp_utc"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["day_of_year"] = df["timestamp_utc"].dt.dayofyear
    df["month"] = df["timestamp_utc"].dt.month

    # Cyclic encoding for hour and day of week (to capture periodicity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Weekend indicator
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


def create_lag_features(
    df: pd.DataFrame,
    value_column: str,
    lag_hours: list = None,
) -> pd.DataFrame:
    """
    Create lagged features from a value column.

    Args:
        df: DataFrame sorted by timestamp
        value_column: Column name to create lags from
        lag_hours: List of lag hours (default: [1, 2, 3, 24, 48, 168])

    Returns:
        DataFrame with added lag features
    """
    if lag_hours is None:
        lag_hours = [1, 2, 3, 24, 48, 168]  # 1h, 2h, 3h, 1d, 2d, 1w

    df = df.copy()

    for lag in lag_hours:
        df[f"{value_column}_lag_{lag}h"] = df[value_column].shift(lag)

    # Rolling statistics
    df[f"{value_column}_rolling_mean_24h"] = (
        df[value_column].shift(1).rolling(window=24, min_periods=1).mean()
    )
    df[f"{value_column}_rolling_std_24h"] = (
        df[value_column].shift(1).rolling(window=24, min_periods=1).std()
    )

    return df


def merge_weather_data(
    df_market: pd.DataFrame,
    df_weather: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge weather data with market data based on timestamps.

    Args:
        df_market: Market data DataFrame with 'timestamp_utc'
        df_weather: Weather data DataFrame (if None, loads from storage)

    Returns:
        Merged DataFrame
    """
    if df_weather is None:
        df_weather = load_weather_data()

    if df_weather is None or df_weather.empty:
        logger.warning("No weather data available, proceeding without weather features")
        # Add empty weather columns
        df_market["temperature_deg_c"] = np.nan
        df_market["wind_speed_m_per_s"] = np.nan
        df_market["global_radiation_w_per_m2"] = np.nan
        df_market["cloud_cover_pct"] = np.nan
        df_market["precipitation_mm"] = np.nan
        return df_market

    # Ensure both DataFrames have datetime index
    df_market = df_market.copy()
    df_weather = df_weather.copy()

    # Reset weather index to access timestamp as column
    if df_weather.index.name == "timestamp":
        df_weather = df_weather.reset_index()
        weather_timestamp_col = "timestamp"
    else:
        weather_timestamp_col = "timestamp_utc"

    # Ensure both market and weather timestamps have consistent timezone info
    # Convert to timezone-naive for merging (pandas merge requirement)
    if df_market["timestamp_utc"].dt.tz is not None:
        df_market = df_market.copy()
        df_market["timestamp_utc"] = df_market["timestamp_utc"].dt.tz_localize(None)

    if df_weather[weather_timestamp_col].dt.tz is not None:
        df_weather = df_weather.copy()
        df_weather[weather_timestamp_col] = df_weather[weather_timestamp_col].dt.tz_localize(None)

    # Round timestamps to nearest hour for matching
    df_market["timestamp_hour"] = df_market["timestamp_utc"].dt.floor("h")
    df_weather["timestamp_hour"] = df_weather[weather_timestamp_col].dt.floor("h")

    # Drop duplicate hours in weather data (keep first value per hour)
    # This prevents cartesian product when both datasets have sub-hourly resolution
    df_weather_hourly = df_weather.drop_duplicates(subset=["timestamp_hour"], keep="first")

    # Merge on hourly timestamps
    weather_columns_to_merge = ["timestamp_hour"]
    if "temperature_deg_c" in df_weather_hourly.columns:
        weather_columns_to_merge.append("temperature_deg_c")
    if "wind_speed_m_per_s" in df_weather_hourly.columns:
        weather_columns_to_merge.append("wind_speed_m_per_s")
    if "global_radiation_w_per_m2" in df_weather_hourly.columns:
        weather_columns_to_merge.append("global_radiation_w_per_m2")
    if "cloud_cover_pct" in df_weather_hourly.columns:
        weather_columns_to_merge.append("cloud_cover_pct")
    if "precipitation_mm" in df_weather_hourly.columns:
        weather_columns_to_merge.append("precipitation_mm")

    df_merged = df_market.merge(
        df_weather_hourly[weather_columns_to_merge],
        on="timestamp_hour",
        how="left",
    )

    # Drop temporary column
    df_merged = df_merged.drop(columns=["timestamp_hour"])

    # Fill missing weather values with forward fill then backward fill
    weather_cols = []
    for col in [
        "temperature_deg_c",
        "wind_speed_m_per_s",
        "global_radiation_w_per_m2",
        "cloud_cover_pct",
        "precipitation_mm",
    ]:
        if col in df_merged.columns:
            weather_cols.append(col)

    if weather_cols:
        df_merged[weather_cols] = df_merged[weather_cols].ffill().bfill()
        logger.info(
            f"Merged weather data: {df_merged[weather_cols].notna().sum().min()} valid records"
        )

    return df_merged


def build_features_and_target(
    market_type: str,
    lag_hours: Optional[list] = None,
    include_weather: bool = True,
    weather_features: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build complete feature matrix (X) and target array (y) for a market type.

    Args:
        market_type: Type of market ('day_ahead', 'imbalance_shortage', 'imbalance_surplus', 'regulation_state')
        lag_hours: List of lag hours for price features (default: [1, 2, 3, 24, 48, 168])
        include_weather: Whether to include weather features
        weather_features: Dict specifying which weather features to include
            (e.g., {'temperature': True, 'wind_speed': True, 'cloud_cover': False, 'precipitation': False})

    Returns:
        Tuple of (X, y) where:
            - X: DataFrame with features
            - y: Series with target values (price)
    """
    # Load market data
    df = load_market_data(market_type)

    if df is None or df.empty:
        logger.error(f"No market data available for {market_type}")
        return pd.DataFrame(), pd.Series(dtype=float)

    # Get target column name (price for regression, regulation_state for classification)
    target_column = MARKET_TYPES[market_type]["target_column"]

    # Ensure sorted by timestamp
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Create time features
    df = create_time_features(df)

    # Auto-select lag hours for production
    if lag_hours is None:
        # Production lags: 1h, 2h, 3h, 24h (1 day), 48h (2 days), 168h (1 week)
        lag_hours = [1, 2, 3, 24, 48, 168]
        logger.info(f"Using production lag features: {lag_hours}")

    # Create lag features
    df = create_lag_features(df, target_column, lag_hours)

    # Merge weather data if requested
    if include_weather:
        df = merge_weather_data(df)

    # Define feature columns
    feature_columns = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month",
    ]

    # Add lag features
    for lag in lag_hours:
        feature_columns.append(f"{target_column}_lag_{lag}h")

    feature_columns.extend(
        [
            f"{target_column}_rolling_mean_24h",
            f"{target_column}_rolling_std_24h",
        ]
    )

    # Add weather features if available and requested
    if include_weather:
        if weather_features is None:
            # Default: use all available weather features
            weather_features = {
                "temperature": True,
                "wind_speed": True,
                "cloud_cover": True,
                "precipitation": True,
            }

        if weather_features.get("temperature", True) and "temperature_deg_c" in df.columns:
            feature_columns.append("temperature_deg_c")
        if weather_features.get("wind_speed", True) and "wind_speed_m_per_s" in df.columns:
            feature_columns.append("wind_speed_m_per_s")
        if weather_features.get("cloud_cover", True) and "cloud_cover_pct" in df.columns:
            feature_columns.append("cloud_cover_pct")
        if weather_features.get("precipitation", True) and "precipitation_mm" in df.columns:
            feature_columns.append("precipitation_mm")
        # Always include global_radiation if available (derived from cloud_cover)
        if "global_radiation_w_per_m2" in df.columns:
            feature_columns.append("global_radiation_w_per_m2")

    # Extract features and target
    # Remove rows with NaN values (due to lagging and rolling)
    df_clean = df.dropna(subset=feature_columns + [target_column])

    if df_clean.empty:
        logger.error(f"No valid samples after feature engineering for {market_type}")
        return pd.DataFrame(), pd.Series(dtype=float)

    X = df_clean[feature_columns].copy()
    y = df_clean[target_column].copy()

    # Store timestamp for reference (not used in modeling)
    X["timestamp_utc"] = df_clean["timestamp_utc"].values

    logger.info(
        f"Built features for {market_type}: {len(X)} samples, {len(feature_columns)} features"
    )

    return X, y


def build_forecast_features(
    market_type: str,
    forecast_start: pd.Timestamp,
    horizon_hours: int,
    lag_hours: Optional[list] = None,
    include_weather: bool = True,
    weather_features: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build feature matrix for generating forecasts.

    Args:
        market_type: Type of market
        forecast_start: Starting timestamp for forecast
        horizon_hours: Number of hours to forecast
        lag_hours: List of lag hours
        include_weather: Whether to include weather features
        weather_features: Dict specifying which weather features to include

    Returns:
        DataFrame with features for each forecast hour
    """
    # Load historical data to get recent values for lags
    df_history = load_market_data(market_type)

    if df_history is None or df_history.empty:
        logger.error(f"No historical data for {market_type}")
        return pd.DataFrame()

    target_column = MARKET_TYPES[market_type]["target_column"]

    # Ensure sorted
    df_history = df_history.sort_values("timestamp_utc").reset_index(drop=True)

    # Create future timestamps at 15-minute intervals
    # horizon_hours * 4 to get 15-minute intervals
    future_timestamps = pd.date_range(
        start=forecast_start,
        periods=horizon_hours * 4,
        freq="15min",
    )

    # Initialize future dataframe
    df_future = pd.DataFrame({"timestamp_utc": future_timestamps})

    # Create time features for future
    df_future = create_time_features(df_future)

    # For lag features, we need to combine history with iterative forecasts
    # For simplicity, use persistence assumption: future = last known value
    # In production, you'd iteratively update with model predictions

    last_known_value = df_history[target_column].iloc[-1]

    # Use production lag hours (must match training)
    if lag_hours is None:
        lag_hours = [1, 2, 3, 24, 48, 168]

    # Create lag features using last known values
    for lag in lag_hours:
        # Use last known price as approximation
        df_future[f"{target_column}_lag_{lag}h"] = last_known_value

    # Rolling statistics from recent history
    recent_mean = df_history[target_column].tail(24).mean()
    recent_std = df_history[target_column].tail(24).std()

    df_future[f"{target_column}_rolling_mean_24h"] = recent_mean
    df_future[f"{target_column}_rolling_std_24h"] = recent_std

    # Weather features
    if include_weather:
        if weather_features is None:
            # Default: use all available weather features
            weather_features = {
                "temperature": True,
                "wind_speed": True,
                "cloud_cover": True,
                "precipitation": True,
            }

        df_weather = load_weather_data()
        if df_weather is not None and not df_weather.empty:
            # Use most recent weather or forecast if available
            # For simplicity, use last known values
            if (
                weather_features.get("temperature", True)
                and "temperature_deg_c" in df_weather.columns
            ):
                df_future["temperature_deg_c"] = df_weather["temperature_deg_c"].iloc[-1]
            if (
                weather_features.get("wind_speed", True)
                and "wind_speed_m_per_s" in df_weather.columns
            ):
                df_future["wind_speed_m_per_s"] = df_weather["wind_speed_m_per_s"].iloc[-1]
            if (
                weather_features.get("cloud_cover", True)
                and "cloud_cover_pct" in df_weather.columns
            ):
                df_future["cloud_cover_pct"] = df_weather["cloud_cover_pct"].iloc[-1]
            if (
                weather_features.get("precipitation", True)
                and "precipitation_mm" in df_weather.columns
            ):
                df_future["precipitation_mm"] = df_weather["precipitation_mm"].iloc[-1]
            if "global_radiation_w_per_m2" in df_weather.columns:
                df_future["global_radiation_w_per_m2"] = df_weather[
                    "global_radiation_w_per_m2"
                ].iloc[-1]
        else:
            # Default values for missing weather data
            if weather_features.get("temperature", True):
                df_future["temperature_deg_c"] = 10.0
            if weather_features.get("wind_speed", True):
                df_future["wind_speed_m_per_s"] = 5.0
            if weather_features.get("cloud_cover", True):
                df_future["cloud_cover_pct"] = 50.0
            if weather_features.get("precipitation", True):
                df_future["precipitation_mm"] = 0.0
            df_future["global_radiation_w_per_m2"] = 100.0

    logger.info(f"Built forecast features: {len(df_future)} intervals (15-min resolution)")

    return df_future
