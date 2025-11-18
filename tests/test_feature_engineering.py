"""
Tests for feature engineering.
"""

import numpy as np
import pandas as pd
import pytest
from app.services.feature_engineering import (
    create_lag_features,
    create_time_features,
    merge_weather_data,
)


class TestFeatureEngineering:
    """Test suite for feature engineering functions."""

    def test_create_time_features(self):
        """Test time feature creation."""
        # Create sample data
        timestamps = pd.date_range("2025-01-01", periods=24, freq="h")
        df = pd.DataFrame({"timestamp_utc": timestamps})

        df_features = create_time_features(df)

        # Check that time features are added
        assert "hour_of_day" in df_features.columns
        assert "day_of_week" in df_features.columns
        assert "is_weekend" in df_features.columns
        assert "hour_sin" in df_features.columns
        assert "hour_cos" in df_features.columns

        # Check ranges
        assert df_features["hour_of_day"].min() >= 0
        assert df_features["hour_of_day"].max() <= 23
        assert df_features["is_weekend"].isin([0, 1]).all()

    def test_create_lag_features(self):
        """Test lag feature creation."""
        # Create sample price data
        n = 200
        timestamps = pd.date_range("2025-01-01", periods=n, freq="h")
        prices = np.random.uniform(30, 60, n)

        df = pd.DataFrame({"timestamp_utc": timestamps, "price_eur_per_mwh": prices})

        lag_hours = [1, 24]
        df_lagged = create_lag_features(df, "price_eur_per_mwh", lag_hours)

        # Check that lag features are added
        assert "price_eur_per_mwh_lag_1h" in df_lagged.columns
        assert "price_eur_per_mwh_lag_24h" in df_lagged.columns
        assert "price_eur_per_mwh_rolling_mean_24h" in df_lagged.columns

        # Check that lag values are correct
        assert df_lagged["price_eur_per_mwh_lag_1h"].iloc[1] == df["price_eur_per_mwh"].iloc[0]

        # Check that first values are NaN (no data to lag from)
        assert pd.isna(df_lagged["price_eur_per_mwh_lag_1h"].iloc[0])

    def test_merge_weather_data(self):
        """Test merging weather data with market data."""
        # Create sample market data
        timestamps_market = pd.date_range("2025-01-01", periods=48, freq="h")
        df_market = pd.DataFrame(
            {
                "timestamp_utc": timestamps_market,
                "price_eur_per_mwh": np.random.uniform(30, 60, 48),
            }
        )

        # Create sample weather data
        timestamps_weather = pd.date_range("2025-01-01", periods=48, freq="h")
        df_weather = pd.DataFrame(
            {
                "timestamp_utc": timestamps_weather,
                "temperature_deg_c": np.random.uniform(5, 15, 48),
                "wind_speed_m_per_s": np.random.uniform(2, 10, 48),
                "global_radiation_w_per_m2": np.random.uniform(0, 500, 48),
            }
        )

        df_merged = merge_weather_data(df_market, df_weather)

        # Check that weather columns are present
        assert "temperature_deg_c" in df_merged.columns
        assert "wind_speed_m_per_s" in df_merged.columns
        assert "global_radiation_w_per_m2" in df_merged.columns

        # Check that we have the right number of rows
        assert len(df_merged) == len(df_market)

    def test_merge_weather_data_with_missing(self):
        """Test merging when weather data has gaps."""
        # Market data every hour
        timestamps_market = pd.date_range("2025-01-01", periods=48, freq="h")
        df_market = pd.DataFrame(
            {
                "timestamp_utc": timestamps_market,
                "price_eur_per_mwh": np.random.uniform(30, 60, 48),
            }
        )

        # Weather data every 2 hours (gaps)
        timestamps_weather = pd.date_range("2025-01-01", periods=24, freq="2h")
        df_weather = pd.DataFrame(
            {
                "timestamp_utc": timestamps_weather,
                "temperature_deg_c": np.random.uniform(5, 15, 24),
                "wind_speed_m_per_s": np.random.uniform(2, 10, 24),
                "global_radiation_w_per_m2": np.random.uniform(0, 500, 24),
            }
        )

        df_merged = merge_weather_data(df_market, df_weather)

        # Should still have all market rows
        assert len(df_merged) == len(df_market)

        # Weather columns should be filled (via forward/backward fill)
        assert df_merged["temperature_deg_c"].notna().sum() > 0
