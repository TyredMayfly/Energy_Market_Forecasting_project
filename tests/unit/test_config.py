"""
Tests for configuration module.
"""

import pytest
from app.core.config import settings, get_data_path, MARKET_TYPES, MODEL_TYPES


class TestConfiguration:
    """Test suite for configuration management."""

    def test_settings_with_env_vars(self, mock_env_vars):
        """Test that settings load from environment variables."""
        # Reload settings to pick up mocked env vars
        from app.core.config import Settings
        test_settings = Settings()
        
        assert test_settings.entsoe_api_key == "test_entsoe_key"
        assert test_settings.knmi_api_key == "test_knmi_key"

    def test_entsoe_base_url_default(self, mock_env_vars):
        """Test ENTSO-E base URL has correct default."""
        from app.core.config import Settings
        test_settings = Settings()
        
        assert "entsoe.eu" in test_settings.entsoe_base_url

    def test_netherlands_eic_code(self, mock_env_vars):
        """Test Netherlands EIC code is correctly set."""
        from app.core.config import Settings
        test_settings = Settings()
        
        assert test_settings.netherlands_eic_code == "10YNL----------L"

    def test_get_data_path_creates_directory(self, test_data_dir, monkeypatch):
        """Test that get_data_path creates directory if it doesn't exist."""
        monkeypatch.setattr('app.core.config.settings.data_dir', test_data_dir)
        
        file_path = get_data_path("test_file.csv")
        
        assert file_path.parent.exists()
        assert file_path.parent == test_data_dir
        assert file_path.name == "test_file.csv"

    def test_market_types_configuration(self):
        """Test that market types are properly configured."""
        assert "day_ahead" in MARKET_TYPES
        assert "intraday" in MARKET_TYPES
        assert "imbalance" in MARKET_TYPES
        
        # Check structure
        for market_type, config in MARKET_TYPES.items():
            assert "display_name" in config
            assert "entsoe_document_type" in config
            assert "data_file" in config
            assert "price_column" in config

    def test_model_types_configuration(self):
        """Test that model types are properly configured."""
        assert "persistence" in MODEL_TYPES
        assert "linear_regression" in MODEL_TYPES
        assert "random_forest" in MODEL_TYPES
        
        # Check structure
        for model_type, config in MODEL_TYPES.items():
            assert "display_name" in config
            assert "description" in config

    def test_max_forecast_horizon(self, mock_env_vars):
        """Test maximum forecast horizon setting."""
        from app.core.config import Settings
        test_settings = Settings()
        
        assert test_settings.max_forecast_horizon_hours == 36

    def test_timezone_setting(self, mock_env_vars):
        """Test timezone is set to Europe/Amsterdam."""
        from app.core.config import Settings
        test_settings = Settings()
        
        assert test_settings.timezone == "Europe/Amsterdam"
