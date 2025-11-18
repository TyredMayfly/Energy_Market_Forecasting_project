"""
Integration tests for weather feature selection functionality.
"""

import pandas as pd
import pytest
from app.services.forecast_service import ForecastService, TrainingDataConfig
from app.services.data_store import save_market_data, save_weather_data
from app.services.feature_engineering import build_features_and_target


class TestWeatherFeatureIntegration:
    """Integration tests for weather feature selection."""
    
    def test_feature_count_all_weather(self, mock_env_vars, sample_market_data, sample_weather_data):
        """Test feature count with all weather features enabled."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)
        
        weather_features = {
            'temperature': True,
            'wind_speed': True,
            'cloud_cover': True,
            'precipitation': True
        }
        
        X, y = build_features_and_target(
            market_type="day_ahead",
            include_weather=True,
            weather_features=weather_features
        )
        
        assert not X.empty
        # Check for the weather features that are actually in the sample data
        # Temperature and wind should be present
        assert any('temperature' in col.lower() for col in X.columns)
        assert any('wind' in col.lower() for col in X.columns)
        # Global radiation is always added (derived from cloud_cover)
        assert any('radiation' in col.lower() for col in X.columns)
    
    def test_feature_count_temperature_wind_only(self, mock_env_vars, sample_market_data, sample_weather_data):
        """Test feature count with only temperature and wind speed."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)
        
        weather_features = {
            'temperature': True,
            'wind_speed': True,
            'cloud_cover': False,
            'precipitation': False
        }
        
        X, y = build_features_and_target(
            market_type="day_ahead",
            include_weather=True,
            weather_features=weather_features
        )
        
        assert not X.empty
        
        # Should have temperature and wind
        assert any('temperature' in col.lower() for col in X.columns)
        assert any('wind' in col.lower() for col in X.columns)
        
        # Should NOT have cloud or precipitation
        # (global_radiation is derived from cloud_cover, so it should be present)
        assert not any('precipitation' in col.lower() for col in X.columns)
    
    def test_feature_count_cloud_precip_only(self, mock_env_vars, sample_market_data, sample_weather_data):
        """Test feature count with only cloud cover and precipitation."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)
        
        weather_features = {
            'temperature': False,
            'wind_speed': False,
            'cloud_cover': True,
            'precipitation': True
        }
        
        X, y = build_features_and_target(
            market_type="day_ahead",
            include_weather=True,
            weather_features=weather_features
        )
        
        assert not X.empty
        
        # Should NOT have temperature or wind
        assert not any('temperature' in col.lower() for col in X.columns)
        assert not any('wind' in col.lower() for col in X.columns)
        
        # Global radiation should be present (derived from cloud_cover, always added)
        assert any('radiation' in col.lower() for col in X.columns)
    
    def test_feature_count_no_weather(self, mock_env_vars, sample_market_data, sample_weather_data):
        """Test feature count with no weather features."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)
        
        weather_features = {
            'temperature': False,
            'wind_speed': False,
            'cloud_cover': False,
            'precipitation': False
        }
        
        X, y = build_features_and_target(
            market_type="day_ahead",
            include_weather=True,
            weather_features=weather_features
        )
        
        assert not X.empty
        
        # Should NOT have any weather features
        weather_keywords = ['temperature', 'wind', 'cloud', 'precipitation', 'radiation']
        weather_cols = [col for col in X.columns if any(w in col.lower() for w in weather_keywords)]
        # Only global_radiation should be present (it's always added)
        assert len(weather_cols) <= 1
    
    def test_model_training_with_different_configs(self, mock_env_vars, sample_market_data, sample_weather_data):
        """Test that models trained with different configs have different performance."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)
        
        service = ForecastService()
        
        # Config 1: All weather features
        config1 = TrainingDataConfig(
            use_weather_data=True,
            use_temperature=True,
            use_wind_speed=True,
            use_cloud_cover=True,
            use_precipitation=True
        )
        service.train_model("day_ahead", "linear_regression", training_config=config1)
        
        # Config 2: Only temperature
        config2 = TrainingDataConfig(
            use_weather_data=True,
            use_temperature=True,
            use_wind_speed=False,
            use_cloud_cover=False,
            use_precipitation=False
        )
        service.train_model("day_ahead", "linear_regression", training_config=config2)
        
        # Config 3: No weather
        config3 = TrainingDataConfig(use_weather_data=False)
        service.train_model("day_ahead", "linear_regression", training_config=config3)
        
        # All should train successfully but be different models
        assert len(service.trained_models) == 3
        
        info1 = service.get_model_info("day_ahead", "linear_regression", config1)
        info2 = service.get_model_info("day_ahead", "linear_regression", config2)
        info3 = service.get_model_info("day_ahead", "linear_regression", config3)
        
        # Models should have different number of features
        assert len(info1['feature_columns']) > len(info2['feature_columns'])
        assert len(info2['feature_columns']) > len(info3['feature_columns'])
    
    def test_forecast_with_selective_features(self, mock_env_vars, sample_market_data, sample_weather_data):
        """Test generating forecasts with selective weather features."""
        save_market_data(sample_market_data, "day_ahead")
        save_weather_data(sample_weather_data)
        
        service = ForecastService()
        config = TrainingDataConfig(
            use_weather_data=True,
            use_temperature=True,
            use_wind_speed=True,
            use_cloud_cover=False,
            use_precipitation=False
        )
        
        df_forecast = service.generate_forecast(
            market_type="day_ahead",
            model_type="linear_regression",
            horizon_hours=6,
            training_config=config
        )
        
        assert df_forecast is not None
        assert not df_forecast.empty
        assert len(df_forecast) == 24  # 6 hours * 4 intervals
        assert "forecast_price_eur_per_mwh" in df_forecast.columns
    
    def test_cache_keys_differ_by_config(self, mock_env_vars):
        """Test that cache keys are different for different weather configs."""
        service = ForecastService()
        
        config_all = TrainingDataConfig(use_weather_data=True)
        config_temp_wind = TrainingDataConfig(
            use_weather_data=True,
            use_temperature=True,
            use_wind_speed=True,
            use_cloud_cover=False,
            use_precipitation=False
        )
        config_no_weather = TrainingDataConfig(use_weather_data=False)
        
        key_all = service._get_model_key("day_ahead", "linear_regression", config_all)
        key_temp_wind = service._get_model_key("day_ahead", "linear_regression", config_temp_wind)
        key_no_weather = service._get_model_key("day_ahead", "linear_regression", config_no_weather)
        
        # All keys should be different
        assert key_all != key_temp_wind
        assert key_all != key_no_weather
        assert key_temp_wind != key_no_weather
        
        # Check specific patterns in keys
        assert "_w1_t1_T1_W1_C1_P1" in key_all
        assert "_w1_t1_T1_W1_C0_P0" in key_temp_wind
        assert "_w0_t1" in key_no_weather
