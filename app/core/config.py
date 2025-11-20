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
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    entsoe_api_key: str = ""
    meteosource_api_key: str = ""

    # ENTSO-E Configuration
    entsoe_base_url: str = "https://web-api.tp.entsoe.eu/api"
    netherlands_eic_code: str = "10YNL----------L"

    # Meteosource Configuration
    meteosource_base_url: str = "https://www.meteosource.com/api/v1/free"
    meteosource_location: str = "amsterdam"  # Default location

    # Application Settings
    timezone: str = "Europe/Amsterdam"
    data_dir: Path = Path(__file__).parent.parent.parent / "data"

    # Update Schedule
    daily_update_hour: int = 0
    daily_update_minute: int = 0

    # Forecasting Settings
    max_forecast_horizon_hours: int = 36
    default_lag_hours: int = 168  # 7 days
    min_training_samples: int = 720  # 30 days at hourly resolution (or 7.5 days at 15-min)

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


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
        "target_column": "price_eur_per_mwh",
        "target_type": "regression",
    },
    "imbalance_shortage": {
        "display_name": "Imbalance - Shortage Price",
        "description": "Price paid when system has power shortage (down-regulation). Reflects cost of activating reserves to increase generation.",
        "data_file": "imbalance_unified.csv",
        "target_column": "shortage_price",
        "target_type": "regression",
    },
    "imbalance_surplus": {
        "display_name": "Imbalance - Surplus Price",
        "description": "Price paid when system has power surplus (up-regulation). Reflects cost of reducing generation or increasing consumption.",
        "data_file": "imbalance_unified.csv",
        "target_column": "surplus_price",
        "target_type": "regression",
    },
    "regulation_state": {
        "display_name": "Regulation State",
        "description": "Categorical forecast of system regulation state: -1 (DOWN), 0 (BALANCED), 1 (UP), 2 (UP_AND_DOWN). Uses classification model.",
        "data_file": "imbalance_unified.csv",
        "target_column": "regulation_state",
        "target_type": "classification",
        "class_labels": {-1: "DOWN", 0: "BALANCED", 1: "UP", 2: "UP_AND_DOWN"},
    },
}

# Weather data configuration
WEATHER_CONFIG = {
    "data_file": "weather_data_2025.csv",
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
    "xgboost_classifier": {
        "display_name": "XGBoost Classifier",
        "description": "Gradient boosting classifier for multi-class prediction of regulation states. Uses advanced machine learning to predict categorical outcomes (UP, DOWN, BALANCED, UP_AND_DOWN).",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
    },
    "hist_gradient_boosting": {
        "display_name": "Histogram Gradient Boosting",
        "description": "Efficient gradient boosting regressor using histogram-based algorithm. Faster than traditional boosting with native support for missing values. Excellent for large datasets with non-linear patterns.",
        "loss": "squared_error",
        "learning_rate": 0.05,
        "max_iter": 500,
        "max_depth": 6,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 0.0,
        "random_state": 42,
    },
}
