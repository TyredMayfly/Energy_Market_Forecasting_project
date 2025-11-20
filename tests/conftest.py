"""
Pytest configuration and fixtures.

Shared fixtures used across all test modules.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def test_data_dir(tmp_path):
    """
    Create a temporary data directory for tests.

    Returns:
        Path to temporary data directory
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_env_vars(monkeypatch, test_data_dir):
    """
    Set up mock environment variables for testing.

    Args:
        monkeypatch: pytest monkeypatch fixture
        test_data_dir: temporary data directory
    """
    monkeypatch.setenv("ENTSOE_API_KEY", "test_entsoe_key")
    monkeypatch.setenv("METEOSOURCE_API_KEY", "test_meteosource_key")
    monkeypatch.setenv("DATA_DIR", str(test_data_dir))

    # Reload all relevant modules to pick up new environment variables
    from app.core import config
    from app.services import entsoe_client, meteosource_client, data_store
    import importlib

    importlib.reload(config)
    importlib.reload(entsoe_client)
    importlib.reload(meteosource_client)
    importlib.reload(data_store)

    # Also set directly on settings object
    config.settings.data_dir = test_data_dir
    config.settings.entsoe_api_key = "test_entsoe_key"
    config.settings.meteosource_api_key = "test_meteosource_key"


@pytest.fixture
def sample_timestamps():
    """
    Generate sample 15-minute resolution timestamps starting from 2024-10-01.

    Provides data from 2024-10-01 until 2025-11-18 (current date).

    Returns:
        List of timezone-aware datetime objects at 15-minute intervals (UTC)
    """
    import pandas as pd
    start = datetime(2024, 10, 1, 0, 0, 0)
    end = datetime(2025, 11, 18, 23, 45, 0)  # Until end of November 18, 2025
    # Create timezone-aware timestamps using pandas
    return pd.date_range(start, end, freq="15min", tz="UTC").tolist()


@pytest.fixture
def sample_market_data(sample_timestamps):
    """
    Create sample market price data.

    Returns:
        DataFrame with market prices
    """
    return pd.DataFrame(
        {
            "timestamp_utc": sample_timestamps,
            "price_eur_per_mwh": np.random.uniform(30, 80, len(sample_timestamps)),
            "market_type": "day_ahead",
        }
    )


@pytest.fixture
def sample_weather_data(sample_timestamps):
    """
    Create sample weather data.

    Returns:
        DataFrame with weather observations (timestamp as index, matching real format)
    """
    df = pd.DataFrame(
        {
            "timestamp": sample_timestamps,
            "temperature_deg_c": np.random.uniform(5, 20, len(sample_timestamps)),
            "wind_speed_m_per_s": np.random.uniform(2, 15, len(sample_timestamps)),
            "global_radiation_w_per_m2": np.random.uniform(0, 500, len(sample_timestamps)),
            "cloud_cover_pct": np.random.uniform(0, 100, len(sample_timestamps)),
            "precipitation_mm": np.random.uniform(0, 10, len(sample_timestamps)),
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def sample_entsoe_xml():
    """
    Provide valid ENTSO-E XML response for testing.

    Returns:
        XML string representing ENTSO-E price document
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
    <mRID>test-document-id</mRID>
    <revisionNumber>1</revisionNumber>
    <type>A44</type>
    <TimeSeries>
        <mRID>1</mRID>
        <businessType>A62</businessType>
        <in_Domain.mRID codingScheme="A01">10YNL----------L</in_Domain.mRID>
        <out_Domain.mRID codingScheme="A01">10YNL----------L</out_Domain.mRID>
        <Period>
            <timeInterval>
                <start>2025-01-01T00:00:00Z</start>
                <end>2025-01-01T04:00:00Z</end>
            </timeInterval>
            <resolution>PT60M</resolution>
            <Point>
                <position>1</position>
                <price.amount>45.50</price.amount>
            </Point>
            <Point>
                <position>2</position>
                <price.amount>46.75</price.amount>
            </Point>
            <Point>
                <position>3</position>
                <price.amount>44.20</price.amount>
            </Point>
            <Point>
                <position>4</position>
                <price.amount>43.80</price.amount>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>
"""


@pytest.fixture
def sample_knmi_files_response():
    """
    Provide sample KNMI file listing response.

    Returns:
        Dictionary representing KNMI API response
    """
    return {
        "files": [
            {
                "filename": "KMDS__OPER_P___10M_OBS_L2_202501010000.nc",
                "size": 12345,
                "lastModified": "2025-01-01T01:00:00Z",
            },
            {
                "filename": "KMDS__OPER_P___10M_OBS_L2_202501020000.nc",
                "size": 12346,
                "lastModified": "2025-01-02T01:00:00Z",
            },
            {
                "filename": "KMDS__OPER_P___10M_OBS_L2_202412310000.nc",
                "size": 12344,
                "lastModified": "2024-12-31T01:00:00Z",
            },
        ],
        "isTruncated": False,
    }


@pytest.fixture
def sample_knmi_download_url_response():
    """
    Provide sample KNMI download URL response.

    Returns:
        Dictionary with temporary download URL
    """
    return {"temporaryDownloadUrl": "https://example.com/download/file.nc?token=abc123"}


# ============================================================================
# Realistic Training Data Fixtures
# ============================================================================


@pytest.fixture
def realistic_day_ahead_features():
    """
    Create realistic day-ahead price features for model training.

    Generates 1000+ samples with realistic patterns:
    - Price patterns with daily and weekly seasonality
    - Weather correlations (temperature, wind, solar)
    - Time features (hour, day, weekend)
    - Lag features (1h, 2h, 3h, 24h, 48h, 168h)

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    np.random.seed(42)  # For reproducibility

    # Generate 1200 samples (50 days of hourly data)
    n_samples = 1200

    # Create base time series with realistic patterns
    hours = np.arange(n_samples)

    # Base price with daily cycle (peak during day, low at night)
    base_price = 50 + 15 * np.sin(2 * np.pi * hours / 24)

    # Weekly cycle (higher on weekdays, lower on weekends)
    weekly_pattern = 5 * np.sin(2 * np.pi * hours / (24 * 7))

    # Random noise
    noise = np.random.randn(n_samples) * 8

    # Final price series
    y = base_price + weekly_pattern + noise

    # Create features
    X = pd.DataFrame()

    # Lag features (production config: [24, 48, 168])
    for lag in [24, 48, 168]:
        X[f"price_lag_{lag}h"] = np.concatenate(
            [np.full(lag, y[0]), y[:-lag]]  # Fill initial values
        )

    # Time features
    X["hour_of_day"] = hours % 24
    X["day_of_week"] = (hours // 24) % 7
    X["hour_sin"] = np.sin(2 * np.pi * X["hour_of_day"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour_of_day"] / 24)
    X["day_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
    X["day_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
    X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)

    # Weather features (correlated with price)
    temperature = 10 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.randn(n_samples) * 2
    wind_speed = 8 + 4 * np.random.randn(n_samples)
    wind_speed = np.clip(wind_speed, 0, None)  # No negative wind
    cloud_cover = 50 + 30 * np.random.randn(n_samples)
    cloud_cover = np.clip(cloud_cover, 0, 100)

    X["temperature_deg_c"] = temperature
    X["wind_speed_m_per_s"] = wind_speed
    X["cloud_cover_pct"] = cloud_cover

    # Convert to pandas Series
    y = pd.Series(y, name="price_eur_per_mwh")

    # Train/test split (80/20)
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, y_train, X_test, y_test


@pytest.fixture
def realistic_imbalance_features():
    """
    Create realistic imbalance price features for model training.

    Imbalance prices are more volatile than day-ahead prices.
    Generates 800+ samples with realistic patterns.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    np.random.seed(43)  # Different seed for variety

    # Generate 1000 samples
    n_samples = 1000
    hours = np.arange(n_samples)

    # Imbalance prices are more volatile and can spike
    base_price = 60 + 20 * np.sin(2 * np.pi * hours / 24)

    # Add occasional spikes (5% of time)
    spikes = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    spike_values = spikes * np.random.uniform(50, 150, n_samples)

    # Higher noise for imbalance
    noise = np.random.randn(n_samples) * 15

    y = base_price + spike_values + noise

    # Create features
    X = pd.DataFrame()

    # Lag features
    for lag in [24, 48, 168]:
        X[f"price_lag_{lag}h"] = np.concatenate([np.full(lag, y[0]), y[:-lag]])

    # Time features
    X["hour_of_day"] = hours % 24
    X["day_of_week"] = (hours // 24) % 7
    X["hour_sin"] = np.sin(2 * np.pi * X["hour_of_day"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour_of_day"] / 24)
    X["day_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
    X["day_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
    X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)

    # Weather features
    temperature = 12 + 6 * np.sin(2 * np.pi * hours / 24) + np.random.randn(n_samples) * 3
    wind_speed = 10 + 5 * np.random.randn(n_samples)
    wind_speed = np.clip(wind_speed, 0, None)

    X["temperature_deg_c"] = temperature
    X["wind_speed_m_per_s"] = wind_speed

    y = pd.Series(y, name="price_eur_per_mwh")

    # Train/test split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, y_train, X_test, y_test


@pytest.fixture
def realistic_regulation_features():
    """
    Create realistic regulation state classification features.

    Generates features for predicting regulation states:
    - DOWN (-1): System needs downward regulation
    - BALANCED (0): System is balanced
    - UP (1): System needs upward regulation
    - UP_AND_DOWN (2): Both directions needed

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    np.random.seed(44)

    # Generate 900 samples
    n_samples = 900
    hours = np.arange(n_samples)

    # Create features
    X = pd.DataFrame()

    # Time features influence regulation state
    X["hour_of_day"] = hours % 24
    X["day_of_week"] = (hours // 24) % 7
    X["hour_sin"] = np.sin(2 * np.pi * X["hour_of_day"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour_of_day"] / 24)
    X["day_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
    X["day_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
    X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)

    # Price features influence regulation
    base_price = 50 + 15 * np.sin(2 * np.pi * hours / 24)
    X["day_ahead_price"] = base_price + np.random.randn(n_samples) * 8

    # Imbalance price (more volatile)
    imbalance_price = base_price + np.random.randn(n_samples) * 20
    X["imbalance_price"] = imbalance_price

    # Price difference indicates regulation need
    X["price_difference"] = X["imbalance_price"] - X["day_ahead_price"]

    # Lag features
    for lag in [1, 2, 3, 24]:
        X[f"price_diff_lag_{lag}h"] = np.concatenate(
            [np.full(lag, 0.0), X["price_difference"].values[:-lag]]
        )

    # Weather features
    X["temperature_deg_c"] = (
        10 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.randn(n_samples) * 2
    )
    X["wind_speed_m_per_s"] = np.clip(8 + 4 * np.random.randn(n_samples), 0, None)

    # Create target based on price differences and patterns
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        price_diff = X["price_difference"].iloc[i]

        # Classification logic
        if abs(price_diff) < 5:
            y[i] = 0  # BALANCED
        elif price_diff > 15:
            y[i] = 1  # UP
        elif price_diff < -15:
            y[i] = -1  # DOWN
        else:
            # Random assignment for borderline cases
            y[i] = np.random.choice([-1, 0, 1, 2], p=[0.25, 0.4, 0.25, 0.1])

    # Add some UP_AND_DOWN cases (10% of samples)
    up_and_down_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[up_and_down_indices] = 2

    y = pd.Series(y, name="regulation_state")

    # Train/test split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, y_train, X_test, y_test
