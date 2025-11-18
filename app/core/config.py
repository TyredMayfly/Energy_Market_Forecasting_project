"""
Configuration management for the Market Forecasting application.

This module handles all configuration settings, including:
- Environment variable loading
- API credentials
- Data source configurations
- Application settings
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    entsoe_api_key: str = ""
    knmi_api_key: str = ""

    # ENTSO-E Configuration
    entsoe_base_url: str = "https://web-api.tp.entsoe.eu/api"
    netherlands_eic_code: str = "10YNL----------L"

    # KNMI Configuration
    knmi_base_url: str = "https://api.dataplatform.knmi.nl/open-data/v1"
    knmi_dataset_name: str = "Actuele10mindataKNMIstations"
    knmi_dataset_version: str = "2"

    # Application Settings
    timezone: str = "Europe/Amsterdam"
    data_dir: Path = Path(__file__).parent.parent.parent / "data"
    demo_mode: bool = False  # Use existing sample data, reduced lag features

    # Update Schedule
    daily_update_hour: int = 0
    daily_update_minute: int = 0

    # Forecasting Settings
    max_forecast_horizon_hours: int = 36
    default_lag_hours: int = 168  # 7 days
    min_training_samples: int = 720  # 30 days at hourly resolution (or 7.5 days at 15-min)
    demo_lag_hours: int = 24  # Reduced lags for demo mode (24 hours)
    demo_min_training_samples: int = 48  # 2 days at hourly (or 12 hours at 15-min)

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_data_path(filename: str) -> Path:
    """
    Get the full path to a data file.

    Args:
        filename: Name of the data file

    Returns:
        Full path to the data file
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_dir / filename


# Market type configurations
MARKET_TYPES = {
    "day_ahead": {
        "display_name": "Day-Ahead Market",
        "description": "Prices determined one day before delivery. Primary market for wholesale electricity trading.",
        "entsoe_document_type": "A44",
        "data_file": "entsoe_day_ahead_prices_2025.csv",
        "price_column": "price_eur_per_mwh",
    },
    "intraday": {
        "display_name": "Intraday Market",
        "description": "Continuous trading closer to delivery time. Allows market participants to adjust positions based on updated forecasts.",
        "entsoe_document_type": "A45",
        "data_file": "entsoe_intraday_prices_2025.csv",
        "price_column": "price_eur_per_mwh",
    },
    "imbalance": {
        "display_name": "Imbalance Market",
        "description": "Real-time settlement prices for supply-demand imbalances. Reflects actual system conditions.",
        "entsoe_document_type": "A53",
        "data_file": "entsoe_imbalance_data_2025.csv",
        "price_column": "imbalance_price_eur_per_mwh",
    },
}

# Weather data configuration
WEATHER_CONFIG = {
    "data_file": "knmi_weather_2025.csv",
    "required_variables": [
        "temperature_deg_c",
        "wind_speed_m_per_s",
        "global_radiation_w_per_m2",
    ],
}

# Model configurations
MODEL_TYPES = {
    "persistence": {
        "display_name": "Persistence Model",
        "description": "Simple baseline that assumes future prices will equal the most recent observed value. Fast and interpretable, but doesn't capture trends or patterns.",
    },
    "linear_regression": {
        "display_name": "Linear Regression",
        "description": "Statistical model that learns linear relationships between price lags, time patterns, and weather conditions. Balances speed with reasonable accuracy.",
    },
    "random_forest": {
        "display_name": "Random Forest",
        "description": "Advanced ensemble model using decision trees to capture non-linear patterns and complex interactions between features. More accurate but computationally intensive.",
        "n_estimators": 50,
        "max_depth": 10,
        "random_state": 42,
    },
}
