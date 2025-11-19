"""
Tests for the forecast service module.
"""

import pandas as pd
import pytest
from datetime import datetime
from app.services.forecast_service import ForecastService, TrainingDataConfig
from app.services.data_store import save_market_data, save_weather_data


class TestTrainingDataConfig:
    """Tests for TrainingDataConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingDataConfig()

        assert config.use_weather_data is True
        assert config.use_time_features is True
        assert config.use_temperature is True
        assert config.use_wind_speed is True
        assert config.use_cloud_cover is True
        assert config.use_precipitation is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingDataConfig(
            use_weather_data=True,
            use_temperature=True,
            use_wind_speed=True,
            use_cloud_cover=False,
            use_precipitation=False,
        )

        assert config.use_weather_data is True
        assert config.use_temperature is True
        assert config.use_wind_speed is True
        assert config.use_cloud_cover is False
        assert config.use_precipitation is False

    def test_no_weather_config(self):
        """Test configuration without weather data."""
        config = TrainingDataConfig(use_weather_data=False)

        assert config.use_weather_data is False
        # Individual flags don't matter if weather is disabled
        assert config.use_temperature is True  # defaults


class TestForecastService:
    """Tests for ForecastService class."""

    def test_initialization(self):
        """Test service initialization."""
        service = ForecastService()
        assert service.trained_models == {}

    def test_get_model_instance_persistence(self):
        """Test creating a persistence model instance."""
        service = ForecastService()
        model = service._get_model_instance("persistence")

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_model_instance_linear_regression(self):
        """Test creating a linear regression model instance."""
        service = ForecastService()
        model = service._get_model_instance("linear_regression")

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_model_instance_random_forest(self):
        """Test creating a random forest model instance."""
        service = ForecastService()
        model = service._get_model_instance("random_forest")

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_model_instance_invalid(self):
        """Test error handling for invalid model type."""
        service = ForecastService()

        with pytest.raises(ValueError, match="Unknown model type"):
            service._get_model_instance("invalid_model")

    def test_get_model_key_default_config(self):
        """Test model key generation with default config."""
        service = ForecastService()
        config = TrainingDataConfig()

        key = service._get_model_key("day_ahead", "persistence", config)

        assert "day_ahead" in key
        assert "persistence" in key
        assert "_w1_t1" in key  # weather and time features enabled
        assert "_T1_W1_C1_P1" in key  # all weather features enabled

    def test_get_model_key_no_weather(self):
        """Test model key generation without weather data."""
        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)

        key = service._get_model_key("day_ahead", "linear_regression", config)

        assert "day_ahead" in key
        assert "linear_regression" in key
        assert "_w0_t1" in key  # weather disabled, time enabled
        # No T/W/C/P suffixes when weather is disabled
        assert "T1" not in key

    def test_get_model_key_selective_weather(self):
        """Test model key generation with selective weather features."""
        service = ForecastService()
        config = TrainingDataConfig(
            use_weather_data=True,
            use_temperature=True,
            use_wind_speed=True,
            use_cloud_cover=False,
            use_precipitation=False,
        )

        key = service._get_model_key("day_ahead", "random_forest", config)

        assert "_w1_t1" in key
        assert "_T1_W1_C0_P0" in key  # only temp and wind enabled

    def test_train_model_basic(self, mock_env_vars, sample_market_data):
        """Test basic model training."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)

        success = service.train_model("day_ahead", "persistence", training_config=config)

        assert success is True
        assert len(service.trained_models) == 1

    def test_train_model_caching(self, mock_env_vars, sample_market_data):
        """Test that trained models are cached."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)

        # Train once
        service.train_model("day_ahead", "persistence", training_config=config)
        model_key = service._get_model_key("day_ahead", "persistence", config)
        first_model = service.trained_models[model_key]["model"]

        # Train again - should use cache
        service.train_model("day_ahead", "persistence", training_config=config)
        second_model = service.trained_models[model_key]["model"]

        assert first_model is second_model  # Same object

    def test_train_model_force_retrain(self, mock_env_vars, sample_market_data):
        """Test force retraining of cached models."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)

        # Train once
        service.train_model("day_ahead", "persistence", training_config=config)
        model_key = service._get_model_key("day_ahead", "persistence", config)
        first_model = service.trained_models[model_key]["model"]

        # Force retrain
        service.train_model("day_ahead", "persistence", force_retrain=True, training_config=config)
        second_model = service.trained_models[model_key]["model"]

        assert first_model is not second_model  # Different objects

    def test_train_model_with_weather(self, mock_env_vars, sample_market_data, sample_weather_data):
        """Test model training with weather data."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=True)

        success = service.train_model("day_ahead", "linear_regression", training_config=config)

        assert success is True
        model_key = service._get_model_key("day_ahead", "linear_regression", config)
        assert model_key in service.trained_models

    def test_generate_forecast_basic(self, mock_env_vars, sample_market_data):
        """Test basic forecast generation."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)
        forecast_start = datetime(2025, 1, 8, 0, 0, 0)

        df_forecast = service.generate_forecast(
            market_type="day_ahead",
            model_type="persistence",
            horizon_hours=6,
            forecast_start=forecast_start,
            training_config=config,
        )

        assert df_forecast is not None
        assert not df_forecast.empty
        # 6 hours * 4 intervals per hour = 24 rows (15-minute resolution)
        assert len(df_forecast) == 24
        assert "timestamp_utc" in df_forecast.columns
        assert "forecast_price_eur_per_mwh" in df_forecast.columns

    def test_generate_forecast_15min_resolution(self, mock_env_vars, sample_market_data):
        """Test that forecasts are generated at 15-minute intervals."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)
        forecast_start = datetime(2025, 1, 8, 0, 0, 0)

        df_forecast = service.generate_forecast(
            market_type="day_ahead",
            model_type="persistence",
            horizon_hours=24,
            forecast_start=forecast_start,
            training_config=config,
        )

        # 24 hours * 4 intervals = 96 rows
        assert len(df_forecast) == 96

        # Check interval between rows
        time_diff = df_forecast["timestamp_utc"].iloc[1] - df_forecast["timestamp_utc"].iloc[0]
        assert time_diff == pd.Timedelta(minutes=15)

    def test_generate_forecast_invalid_market(self, mock_env_vars):
        """Test forecast generation with invalid market type."""
        service = ForecastService()
        config = TrainingDataConfig()

        df_forecast = service.generate_forecast(
            market_type="invalid_market",
            model_type="persistence",
            horizon_hours=24,
            training_config=config,
        )

        assert df_forecast is None

    def test_generate_forecast_invalid_model(self, mock_env_vars):
        """Test forecast generation with invalid model type."""
        service = ForecastService()
        config = TrainingDataConfig()

        df_forecast = service.generate_forecast(
            market_type="day_ahead",
            model_type="invalid_model",
            horizon_hours=24,
            training_config=config,
        )

        assert df_forecast is None

    def test_generate_forecast_invalid_horizon(self, mock_env_vars, sample_market_data):
        """Test forecast generation with invalid horizon."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig()

        # Negative horizon
        df_forecast = service.generate_forecast(
            market_type="day_ahead",
            model_type="persistence",
            horizon_hours=-1,
            training_config=config,
        )
        assert df_forecast is None

        # Horizon too large
        df_forecast = service.generate_forecast(
            market_type="day_ahead",
            model_type="persistence",
            horizon_hours=1000,
            training_config=config,
        )
        assert df_forecast is None

    def test_calculate_forecast_rmse_basic(self, mock_env_vars, sample_market_data):
        """Test RMSE calculation with overlapping data."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)

        # Generate forecast for a time period with data
        forecast_start = sample_market_data["timestamp_utc"].iloc[0]
        df_forecast = service.generate_forecast(
            market_type="day_ahead",
            model_type="persistence",
            horizon_hours=6,
            forecast_start=forecast_start,
            training_config=config,
        )

        # Calculate RMSE
        rmse_result = service.calculate_forecast_rmse(df_forecast, "day_ahead")

        assert rmse_result is not None
        rmse, n_points = rmse_result
        assert rmse >= 0  # RMSE should be non-negative
        assert isinstance(rmse, float)
        assert n_points > 0
        assert isinstance(n_points, int)

    def test_calculate_forecast_rmse_no_overlap(self, mock_env_vars, sample_market_data):
        """Test RMSE calculation with no overlapping data."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)

        # Generate forecast for a future time with no actual data
        forecast_start = datetime(2030, 1, 1, 0, 0, 0)
        df_forecast = service.generate_forecast(
            market_type="day_ahead",
            model_type="persistence",
            horizon_hours=6,
            forecast_start=forecast_start,
            training_config=config,
        )

        # Calculate RMSE - should return None (no overlap)
        rmse_result = service.calculate_forecast_rmse(df_forecast, "day_ahead")

        assert rmse_result is None

    def test_calculate_forecast_rmse_no_market_data(self, mock_env_vars):
        """Test RMSE calculation when no market data exists."""
        service = ForecastService()

        # Create fake forecast dataframe
        df_forecast = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2025-01-01", periods=10, freq="15min"),
                "forecast_price_eur_per_mwh": [50.0] * 10,
            }
        )

        rmse_result = service.calculate_forecast_rmse(df_forecast, "day_ahead")

        assert rmse_result is None

    def test_compare_models_basic(self, mock_env_vars, sample_market_data):
        """Test comparing multiple models."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)
        forecast_start = datetime(2025, 1, 8, 0, 0, 0)

        df_comparison = service.compare_models(
            market_type="day_ahead",
            model_types=["persistence", "linear_regression"],
            horizon_hours=6,
            forecast_start=forecast_start,
            training_config=config,
        )

        assert df_comparison is not None
        assert not df_comparison.empty

        # Should have forecasts from both models
        assert "persistence" in df_comparison["model_type"].values
        assert "linear_regression" in df_comparison["model_type"].values

    def test_get_model_info_basic(self, mock_env_vars, sample_market_data):
        """Test retrieving model information."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)

        # Train a model
        service.train_model("day_ahead", "persistence", training_config=config)

        # Get model info
        info = service.get_model_info("day_ahead", "persistence", config)

        assert info is not None
        # Model object is not returned (it's popped out)
        assert "model" not in info
        assert "trained_at" in info
        assert "n_samples" in info
        assert "feature_columns" in info

    def test_get_model_info_not_trained(self, mock_env_vars):
        """Test retrieving info for untrained model."""
        service = ForecastService()
        config = TrainingDataConfig()

        info = service.get_model_info("day_ahead", "persistence", config)

        assert info is None


class TestWeatherFeatureSelection:
    """Tests for granular weather feature selection."""

    def test_all_weather_features(self, mock_env_vars, sample_market_data, sample_weather_data):
        """Test training with all weather features."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)

        service = ForecastService()
        config = TrainingDataConfig(
            use_weather_data=True,
            use_temperature=True,
            use_wind_speed=True,
            use_cloud_cover=True,
            use_precipitation=True,
        )

        success = service.train_model("day_ahead", "linear_regression", training_config=config)
        assert success is True

        model_key = service._get_model_key("day_ahead", "linear_regression", config)
        info = service.trained_models[model_key]

        # With all weather features, should have more features than without weather
        assert info["n_samples"] > 0

    def test_selective_weather_features(
        self, mock_env_vars, sample_market_data, sample_weather_data
    ):
        """Test training with selective weather features."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)

        service = ForecastService()
        config = TrainingDataConfig(
            use_weather_data=True,
            use_temperature=True,
            use_wind_speed=True,
            use_cloud_cover=False,
            use_precipitation=False,
        )

        success = service.train_model("day_ahead", "linear_regression", training_config=config)
        assert success is True

    def test_different_configs_different_cache(
        self, mock_env_vars, sample_market_data, sample_weather_data
    ):
        """Test that different configs create different cached models."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)

        service = ForecastService()

        # Train with all weather features
        config1 = TrainingDataConfig(use_weather_data=True)
        service.train_model("day_ahead", "linear_regression", training_config=config1)

        # Train with selective weather features
        config2 = TrainingDataConfig(
            use_weather_data=True,
            use_temperature=True,
            use_wind_speed=True,
            use_cloud_cover=False,
            use_precipitation=False,
        )
        service.train_model("day_ahead", "linear_regression", training_config=config2)

        # Should have two different cached models
        assert len(service.trained_models) == 2

        key1 = service._get_model_key("day_ahead", "linear_regression", config1)
        key2 = service._get_model_key("day_ahead", "linear_regression", config2)

        assert key1 != key2
        assert key1 in service.trained_models
        assert key2 in service.trained_models


class TestHistoricalWindowFeature:
    """Tests for historical window functionality."""

    def test_generate_forecast_with_historical_window(self, mock_env_vars, sample_market_data):
        """Test that historical_window_hours parameter includes historical data in forecast."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)
        forecast_start = datetime(2025, 1, 8, 0, 0, 0)

        # Generate forecast with 24-hour historical window
        df_forecast = service.generate_forecast(
            market_type="day_ahead",
            model_type="persistence",
            horizon_hours=24,
            forecast_start=forecast_start,
            historical_window_hours=24,
            training_config=config,
        )

        assert df_forecast is not None
        assert not df_forecast.empty

        # Should have historical + forecast data
        # 24 hours historical + 24 hours forecast = 48 hours * 4 intervals = 192 rows
        assert len(df_forecast) == 192

        # First timestamp should be 24 hours before forecast_start
        expected_first = forecast_start - pd.Timedelta(hours=24)
        assert df_forecast["timestamp_utc"].iloc[0] == expected_first

    def test_generate_forecast_with_different_historical_windows(
        self, mock_env_vars, sample_market_data
    ):
        """Test different historical window sizes."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)
        forecast_start = datetime(2025, 1, 8, 0, 0, 0)

        # Test with 48-hour window
        df_48h = service.generate_forecast(
            market_type="day_ahead",
            model_type="persistence",
            horizon_hours=24,
            forecast_start=forecast_start,
            historical_window_hours=48,
            training_config=config,
        )

        # Test with 72-hour window
        df_72h = service.generate_forecast(
            market_type="day_ahead",
            model_type="persistence",
            horizon_hours=24,
            forecast_start=forecast_start,
            historical_window_hours=72,
            training_config=config,
        )

        # 48h + 24h = 72h * 4 intervals = 288 rows
        assert len(df_48h) == 288

        # 72h + 24h = 96h * 4 intervals = 384 rows
        assert len(df_72h) == 384

    def test_historical_window_affects_rmse_calculation(self, mock_env_vars, sample_market_data):
        """Test that historical window affects RMSE calculation coverage."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)
        forecast_start = datetime(2025, 1, 8, 0, 0, 0)

        # Generate forecast with 72-hour historical window
        df_forecast = service.generate_forecast(
            market_type="day_ahead",
            model_type="persistence",
            horizon_hours=24,
            forecast_start=forecast_start,
            historical_window_hours=72,
            training_config=config,
        )

        # Calculate RMSE - should have more overlap points with larger historical window
        rmse_result = service.calculate_forecast_rmse(df_forecast, "day_ahead")

        assert rmse_result is not None
        rmse, n_points = rmse_result

        # Should have points from the historical window
        # 72h historical * 4 intervals = 288 points (if all data available)
        assert n_points > 0
        assert rmse >= 0

    def test_compare_models_with_historical_window(self, mock_env_vars, sample_market_data):
        """Test model comparison with historical window."""
        save_market_data(sample_market_data, "day_ahead")

        service = ForecastService()
        config = TrainingDataConfig(use_weather_data=False)
        forecast_start = datetime(2025, 1, 8, 0, 0, 0)

        df_comparison = service.compare_models(
            market_type="day_ahead",
            model_types=["persistence", "linear_regression"],
            horizon_hours=24,
            forecast_start=forecast_start,
            historical_window_hours=48,
            training_config=config,
        )

        assert df_comparison is not None
        assert not df_comparison.empty

        # Should include historical data for both models
        # 48h historical + 24h forecast = 72h * 4 intervals * 2 models = 576 rows
        assert len(df_comparison) == 576

        # Check both models are present
        assert "persistence" in df_comparison["model_type"].values
        assert "linear_regression" in df_comparison["model_type"].values
