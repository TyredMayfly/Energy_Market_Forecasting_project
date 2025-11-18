"""
Tests for feature engineering.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.services.feature_engineering import (
    create_time_features,
    create_lag_features,
    merge_weather_data,
    build_features_and_target,
    build_forecast_features,
)
from app.services.data_store import save_market_data, save_weather_data


class TestTimeFeatures:
    """Test suite for time feature creation."""

    def test_create_time_features_basic(self, sample_timestamps):
        """Test basic time feature creation."""
        df = pd.DataFrame({'timestamp_utc': sample_timestamps})
        
        result = create_time_features(df)
        
        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns

    def test_time_features_hour_range(self, sample_timestamps):
        """Test that hour_of_day is in valid range."""
        df = pd.DataFrame({'timestamp_utc': sample_timestamps})
        
        result = create_time_features(df)
        
        assert result["hour_of_day"].min() >= 0
        assert result["hour_of_day"].max() <= 23

    def test_time_features_day_of_week_range(self, sample_timestamps):
        """Test that day_of_week is in valid range."""
        df = pd.DataFrame({'timestamp_utc': sample_timestamps})
        
        result = create_time_features(df)
        
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 6

    def test_time_features_cyclic_encoding(self, sample_timestamps):
        """Test cyclic encoding for hour and day."""
        df = pd.DataFrame({'timestamp_utc': sample_timestamps})
        
        result = create_time_features(df)
        
        # Check cyclic features exist
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "day_sin" in result.columns
        assert "day_cos" in result.columns
        
        # Check range [-1, 1]
        assert result["hour_sin"].min() >= -1
        assert result["hour_sin"].max() <= 1
        assert result["hour_cos"].min() >= -1
        assert result["hour_cos"].max() <= 1

    def test_time_features_weekend_detection(self):
        """Test weekend indicator."""
        # Create specific dates: Monday through Sunday
        timestamps = pd.date_range('2025-01-06', periods=7, freq='D')  # Starts on Monday
        df = pd.DataFrame({'timestamp_utc': timestamps})
        
        result = create_time_features(df)
        
        # First 5 days should be weekdays (0), last 2 should be weekend (1)
        assert result["is_weekend"].iloc[:5].sum() == 0
        assert result["is_weekend"].iloc[5:].sum() == 2

    def test_time_features_month_range(self, sample_timestamps):
        """Test that month is in valid range."""
        df = pd.DataFrame({'timestamp_utc': sample_timestamps})
        
        result = create_time_features(df)
        
        assert result["month"].min() >= 1
        assert result["month"].max() <= 12

    def test_time_features_preserves_original(self, sample_timestamps):
        """Test that original timestamp column is preserved."""
        df = pd.DataFrame({'timestamp_utc': sample_timestamps})
        
        result = create_time_features(df)
        
        assert "timestamp_utc" in result.columns
        pd.testing.assert_series_equal(result["timestamp_utc"], df["timestamp_utc"])


class TestLagFeatures:
    """Test suite for lag feature creation."""

    def test_create_lag_features_basic(self, sample_timestamps):
        """Test basic lag feature creation."""
        df = pd.DataFrame({
            'timestamp_utc': sample_timestamps,
            'price': np.arange(len(sample_timestamps))
        })
        
        result = create_lag_features(df, 'price', lag_hours=[1, 24])
        
        assert "price_lag_1h" in result.columns
        assert "price_lag_24h" in result.columns

    def test_lag_features_default_lags(self, sample_timestamps):
        """Test default lag hours."""
        df = pd.DataFrame({
            'timestamp_utc': sample_timestamps,
            'price': np.arange(len(sample_timestamps))
        })
        
        result = create_lag_features(df, 'price')
        
        # Default lags: [1, 2, 3, 24, 48, 168]
        assert "price_lag_1h" in result.columns
        assert "price_lag_24h" in result.columns
        assert "price_lag_168h" in result.columns

    def test_lag_features_correctness(self):
        """Test that lag values are correct."""
        df = pd.DataFrame({
            'timestamp_utc': pd.date_range('2025-01-01', periods=50, freq='h'),
            'price': np.arange(50)
        })
        
        result = create_lag_features(df, 'price', lag_hours=[1, 2])
        
        # Lag 1 hour should shift by 1
        assert result["price_lag_1h"].iloc[1] == 0
        assert result["price_lag_1h"].iloc[2] == 1
        
        # Lag 2 hours should shift by 2
        assert result["price_lag_2h"].iloc[2] == 0
        assert result["price_lag_2h"].iloc[3] == 1

    def test_lag_features_nan_handling(self, sample_timestamps):
        """Test that initial lag values are NaN."""
        df = pd.DataFrame({
            'timestamp_utc': sample_timestamps,
            'price': np.arange(len(sample_timestamps))
        })
        
        result = create_lag_features(df, 'price', lag_hours=[1, 24])
        
        # First value should be NaN for lag_1h
        assert pd.isna(result["price_lag_1h"].iloc[0])
        
        # First 24 values should be NaN for lag_24h
        assert pd.isna(result["price_lag_24h"].iloc[0])
        assert pd.isna(result["price_lag_24h"].iloc[23])

    def test_rolling_mean_feature(self, sample_timestamps):
        """Test rolling mean feature."""
        df = pd.DataFrame({
            'timestamp_utc': sample_timestamps,
            'price': np.ones(len(sample_timestamps)) * 50
        })
        
        result = create_lag_features(df, 'price', lag_hours=[1])
        
        assert "price_rolling_mean_24h" in result.columns
        
        # With constant values, rolling mean should equal value (after warmup)
        assert np.isclose(result["price_rolling_mean_24h"].iloc[-1], 50, atol=0.1)

    def test_rolling_std_feature(self, sample_timestamps):
        """Test rolling standard deviation feature."""
        df = pd.DataFrame({
            'timestamp_utc': sample_timestamps,
            'price': np.ones(len(sample_timestamps)) * 50
        })
        
        result = create_lag_features(df, 'price', lag_hours=[1])
        
        assert "price_rolling_std_24h" in result.columns
        
        # With constant values, rolling std should be 0
        assert np.isclose(result["price_rolling_std_24h"].iloc[-1], 0, atol=0.1)


class TestWeatherMerge:
    """Test suite for weather data merging."""

    def test_merge_weather_data_basic(self, sample_market_data, sample_weather_data):
        """Test basic weather data merging."""
        result = merge_weather_data(sample_market_data, sample_weather_data)
        
        assert "temperature_deg_c" in result.columns
        assert "wind_speed_m_per_s" in result.columns
        assert "global_radiation_w_per_m2" in result.columns

    def test_merge_weather_preserves_market_rows(self, sample_market_data, sample_weather_data):
        """Test that merge preserves all market data rows (exactly, no duplicates)."""
        result = merge_weather_data(sample_market_data, sample_weather_data)
        
        # Should preserve exact number of market rows (no duplicates from cartesian product)
        assert len(result) == len(sample_market_data)
        # All original market timestamps should be preserved
        assert set(sample_market_data["timestamp_utc"]) == set(result["timestamp_utc"])

    def test_merge_weather_with_none(self, mock_env_vars, sample_market_data):
        """Test merging when weather data is None."""
        result = merge_weather_data(sample_market_data, df_weather=None)
        
        # Should add weather columns with NaN
        assert "temperature_deg_c" in result.columns
        assert result["temperature_deg_c"].isna().all()

    def test_merge_weather_with_empty_df(self, sample_market_data):
        """Test merging with empty weather DataFrame."""
        empty_weather = pd.DataFrame()
        
        result = merge_weather_data(sample_market_data, empty_weather)
        
        # Should add weather columns with NaN
        assert "temperature_deg_c" in result.columns

    def test_merge_weather_timestamp_alignment(self):
        """Test that weather data aligns correctly with market timestamps."""
        market_timestamps = pd.date_range('2025-01-01 00:00', periods=24, freq='h')
        market_df = pd.DataFrame({
            'timestamp_utc': market_timestamps,
            'price_eur_per_mwh': np.arange(24)
        })
        
        weather_df = pd.DataFrame({
            'timestamp_utc': market_timestamps,
            'temperature_deg_c': np.arange(24) * 2,
            'wind_speed_m_per_s': np.arange(24),
            'global_radiation_w_per_m2': np.arange(24) * 10
        })
        
        result = merge_weather_data(market_df, weather_df)
        
        # Weather values should align
        assert result["temperature_deg_c"].iloc[0] == 0
        assert result["temperature_deg_c"].iloc[23] == 46


class TestBuildFeaturesAndTarget:
    """Test suite for complete feature building."""

    def test_build_features_with_data(self, mock_env_vars, sample_market_data, sample_weather_data):
        """Test building features with available data."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)
        
        X, y = build_features_and_target("day_ahead", include_weather=True)
        
        assert not X.empty
        assert not y.empty
        assert len(X) == len(y)

    def test_build_features_without_weather(self, mock_env_vars, sample_market_data):
        """Test building features without weather data."""
        save_market_data(sample_market_data, "day_ahead")
        
        X, y = build_features_and_target("day_ahead", include_weather=False)
        
        assert not X.empty
        assert "temperature_deg_c" not in X.columns

    def test_build_features_columns(self, mock_env_vars, sample_market_data, sample_weather_data):
        """Test that expected feature columns are present."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)
        
        X, y = build_features_and_target("day_ahead", include_weather=True)
        
        # Time features
        assert "hour_of_day" in X.columns
        assert "is_weekend" in X.columns
        
        # Weather features
        assert "temperature_deg_c" in X.columns
        assert "wind_speed_m_per_s" in X.columns

    def test_build_features_no_data(self, mock_env_vars):
        """Test building features when no data exists."""
        X, y = build_features_and_target("day_ahead")
        
        assert X.empty
        assert y.empty

    def test_build_features_removes_nans(self, mock_env_vars):
        """Test that rows with NaN lag features are removed."""
        # Create minimal data (lag features will create NaNs)
        timestamps = pd.date_range('2025-01-01', periods=10, freq='h')
        df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'price_eur_per_mwh': np.arange(10),
            'market_type': 'day_ahead'
        })
        save_market_data(df, "day_ahead")
        
        X, y = build_features_and_target("day_ahead", include_weather=False, lag_hours=[1, 2])
        
        # Should have fewer rows due to lag NaNs
        assert len(X) < len(df)
        # No NaNs should remain in features
        assert not X.select_dtypes(include=[np.number]).isna().any().any()

    def test_build_features_custom_lags(self, mock_env_vars, sample_market_data):
        """Test building features with custom lag hours."""
        save_market_data(sample_market_data, "day_ahead")
        
        X, y = build_features_and_target("day_ahead", lag_hours=[1, 2, 3], include_weather=False)
        
        assert "price_eur_per_mwh_lag_1h" in X.columns
        assert "price_eur_per_mwh_lag_2h" in X.columns
        assert "price_eur_per_mwh_lag_3h" in X.columns


class TestBuildForecastFeatures:
    """Test suite for forecast feature building."""

    def test_build_forecast_features_basic(self, mock_env_vars, sample_market_data):
        """Test basic forecast feature building."""
        save_market_data(sample_market_data, "day_ahead")
        
        forecast_start = pd.Timestamp('2025-01-08 00:00:00')
        
        X_forecast = build_forecast_features(
            "day_ahead",
            forecast_start=forecast_start,
            horizon_hours=24,
            include_weather=False
        )
        
        assert not X_forecast.empty
        # 24 hours * 4 intervals per hour (15-minute resolution)
        assert len(X_forecast) == 96

    def test_build_forecast_features_timestamps(self, mock_env_vars, sample_market_data):
        """Test that forecast timestamps are correct."""
        save_market_data(sample_market_data, "day_ahead")
        
        forecast_start = pd.Timestamp('2025-01-08 00:00:00')
        
        X_forecast = build_forecast_features(
            "day_ahead",
            forecast_start=forecast_start,
            horizon_hours=24,
            include_weather=False
        )
        
        assert X_forecast["timestamp_utc"].iloc[0] == forecast_start
        # Last timestamp is 23:45 (23 hours 45 minutes after start)
        assert X_forecast["timestamp_utc"].iloc[-1] == forecast_start + pd.Timedelta(hours=23, minutes=45)

    def test_build_forecast_features_no_data(self, mock_env_vars):
        """Test forecast feature building with no historical data."""
        forecast_start = pd.Timestamp('2025-01-08 00:00:00')
        
        X_forecast = build_forecast_features(
            "day_ahead",
            forecast_start=forecast_start,
            horizon_hours=24
        )
        
        assert X_forecast.empty

    def test_build_forecast_features_includes_time_features(self, mock_env_vars, sample_market_data):
        """Test that forecast includes time features."""
        save_market_data(sample_market_data, "day_ahead")
        
        forecast_start = pd.Timestamp('2025-01-08 00:00:00')
        
        X_forecast = build_forecast_features(
            "day_ahead",
            forecast_start=forecast_start,
            horizon_hours=24,
            include_weather=False
        )
        
        assert "hour_of_day" in X_forecast.columns
        assert "is_weekend" in X_forecast.columns
        assert "hour_sin" in X_forecast.columns

    def test_build_forecast_features_includes_lag_features(self, mock_env_vars, sample_market_data):
        """Test that forecast includes lag features."""
        save_market_data(sample_market_data, "day_ahead")
        
        forecast_start = pd.Timestamp('2025-01-08 00:00:00')
        
        X_forecast = build_forecast_features(
            "day_ahead",
            forecast_start=forecast_start,
            horizon_hours=24,
            lag_hours=[1, 24],
            include_weather=False
        )
        
        assert "price_eur_per_mwh_lag_1h" in X_forecast.columns
        assert "price_eur_per_mwh_lag_24h" in X_forecast.columns

    def test_build_forecast_features_with_weather(self, mock_env_vars, sample_market_data, sample_weather_data):
        """Test forecast feature building with weather."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)
        
        forecast_start = pd.Timestamp('2025-01-08 00:00:00')
        
        X_forecast = build_forecast_features(
            "day_ahead",
            forecast_start=forecast_start,
            horizon_hours=24,
            include_weather=True
        )
        
        assert "temperature_deg_c" in X_forecast.columns
        assert "wind_speed_m_per_s" in X_forecast.columns

    def test_build_forecast_features_horizon_length(self, mock_env_vars, sample_market_data):
        """Test different forecast horizons."""
        save_market_data(sample_market_data, "day_ahead")
        
        forecast_start = pd.Timestamp('2025-01-08 00:00:00')
        
        for horizon in [6, 12, 24, 36]:
            X_forecast = build_forecast_features(
                "day_ahead",
                forecast_start=forecast_start,
                horizon_hours=horizon,
                include_weather=False
            )
            
            # Each hour has 4 intervals (15-minute resolution)
            assert len(X_forecast) == horizon * 4
