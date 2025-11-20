"""
Forecast service for training models and generating predictions.

Orchestrates the entire forecasting pipeline:
1. Load and prepare data
2. Build features
3. Train models
4. Generate forecasts
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from app.core.config import MARKET_TYPES, MODEL_TYPES, settings
from app.core.logging import get_logger
from app.models.linear_regression_model import LinearRegressionPriceModel
from app.models.persistence_model import PersistencePriceModel
from app.models.random_forest_model import RandomForestPriceModel
from app.models.xgboost_classifier import RegulationStateXGBModel
from app.models.hist_gradient_boosting_model import HistGradientBoostingPriceModel
from app.services.data_store import load_market_data
from app.services.data_validation import validate_and_align_market_weather_data
from app.services.feature_engineering import (
    build_features_and_target,
    build_forecast_features,
)
from app.services.weather_data_manager import get_weather_manager
from app.services.entsoe_data_updater import check_and_update_day_ahead_data
from app.services.hyperparameter_service import load_best_params

logger = get_logger(__name__)


@dataclass
class TrainingDataConfig:
    """Configuration for which data sources to use in model training."""

    use_weather_data: bool = True
    use_time_features: bool = True
    # Granular weather feature selection
    use_temperature: bool = True
    use_wind_speed: bool = True
    use_cloud_cover: bool = True
    use_precipitation: bool = True
    # use_market_data is always True (required for price prediction)


class ForecastService:
    """
    Service for managing forecasting operations.

    Handles model training, prediction, and caching of trained models.
    """

    def __init__(self):
        """Initialize the forecast service."""
        self.trained_models: Dict[str, Dict[str, any]] = {}

    def _validate_model_market_compatibility(self, market_type: str, model_type: str) -> None:
        """
        Validate that the model type is compatible with the market type.

        Args:
            market_type: Type of market
            model_type: Type of model

        Raises:
            ValueError: If the model-market combination is invalid
        """
        if market_type not in MARKET_TYPES:
            raise ValueError(f"Unknown market type: {market_type}")

        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}")

        market_config = MARKET_TYPES[market_type]
        target_type = market_config.get("target_type", "regression")

        # Define which models support which target types
        regression_models = {"persistence", "linear_regression", "random_forest", "hist_gradient_boosting"}
        classification_models = {"xgboost_classifier"}

        if target_type == "classification":
            if model_type not in classification_models:
                raise ValueError(
                    f"Model '{model_type}' cannot be used for classification target '{market_type}'. "
                    f"Use one of: {', '.join(classification_models)}"
                )
        elif target_type == "regression":
            if model_type not in regression_models:
                raise ValueError(
                    f"Model '{model_type}' cannot be used for regression target '{market_type}'. "
                    f"Use one of: {', '.join(regression_models)}"
                )

    def _get_model_instance(self, model_type: str, market_type: str = None):
        """
        Create a model instance based on type.

        Automatically loads tuned hyperparameters if available.

        Args:
            model_type: Type of model
            market_type: Type of market (needed for loading tuned hyperparameters)

        Returns:
            Model instance
        """
        # Try to load tuned hyperparameters if market_type provided
        tuned_params = None
        if market_type and model_type != "persistence":
            tuned_params = load_best_params(market_type=market_type, model_type=model_type)
            if tuned_params:
                logger.info(f"Loaded tuned hyperparameters for {model_type} on {market_type}")

        if model_type == "persistence":
            return PersistencePriceModel(method="last")
        elif model_type == "linear_regression":
            if tuned_params:
                return LinearRegressionPriceModel(**tuned_params)
            return LinearRegressionPriceModel()
        elif model_type == "random_forest":
            if tuned_params:
                return RandomForestPriceModel(**tuned_params)
            return RandomForestPriceModel()
        elif model_type == "xgboost_classifier":
            if tuned_params:
                return RegulationStateXGBModel(**tuned_params)
            return RegulationStateXGBModel()
        elif model_type == "hist_gradient_boosting":
            if tuned_params:
                return HistGradientBoostingPriceModel(**tuned_params)
            return HistGradientBoostingPriceModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _get_model_key(
        self,
        market_type: str,
        model_type: str,
        training_config: Optional[TrainingDataConfig] = None,
    ) -> str:
        """
        Get cache key for a trained model.

        Includes training configuration to ensure models trained with different data sources are cached separately.
        """
        if training_config is None:
            training_config = TrainingDataConfig()

        # Create suffix from all config options
        config_suffix = (
            f"_w{int(training_config.use_weather_data)}_t{int(training_config.use_time_features)}"
        )
        if training_config.use_weather_data:
            # Add weather feature selections to cache key
            config_suffix += f"_T{int(training_config.use_temperature)}"
            config_suffix += f"_W{int(training_config.use_wind_speed)}"
            config_suffix += f"_C{int(training_config.use_cloud_cover)}"
            config_suffix += f"_P{int(training_config.use_precipitation)}"
        return f"{market_type}_{model_type}{config_suffix}"

    def train_model(
        self,
        market_type: str,
        model_type: str,
        force_retrain: bool = False,
        training_config: Optional[TrainingDataConfig] = None,
    ) -> bool:
        """
        Train a model for a specific market.

        Args:
            market_type: Type of market
            model_type: Type of model
            force_retrain: Force retraining even if model is cached
            training_config: Configuration for which data sources to use

        Returns:
            True if successful, False otherwise
        """
        if training_config is None:
            training_config = TrainingDataConfig()

        # Validate model-market compatibility
        try:
            self._validate_model_market_compatibility(market_type, model_type)
        except ValueError as e:
            logger.error(str(e))
            return False

        model_key = self._get_model_key(market_type, model_type, training_config)

        # Check if already trained
        if not force_retrain and model_key in self.trained_models:
            logger.info(f"Using cached model: {model_key}")
            return True

        logger.info(
            f"Training {model_type} for {market_type} (weather={training_config.use_weather_data})"
        )

        try:
            # Build weather features dict from config
            weather_features = None
            if training_config.use_weather_data:
                weather_features = {
                    "temperature": training_config.use_temperature,
                    "wind_speed": training_config.use_wind_speed,
                    "cloud_cover": training_config.use_cloud_cover,
                    "precipitation": training_config.use_precipitation,
                }

            # Build features and target with specified data sources
            X, y = build_features_and_target(
                market_type=market_type,
                include_weather=training_config.use_weather_data,
                weather_features=weather_features,
            )

            if X.empty or y.empty:
                logger.error(f"No data available for training {market_type}")
                return False

            # Check minimum samples required for training
            if len(X) < settings.min_training_samples:
                logger.warning(
                    f"Insufficient training samples for {market_type}: "
                    f"{len(X)} < {settings.min_training_samples}"
                )
                # Continue anyway to allow forecasting with limited data

            # Create and train model
            model = self._get_model_instance(model_type, market_type)
            
            # Prepare feature columns (exclude timestamp)
            feature_columns = [col for col in X.columns if col != "timestamp_utc"]
            
            # For XGBoost, set feature columns before training
            if model_type == "xgboost_classifier":
                model.feature_columns = feature_columns
            
            # Train model with only feature columns (no timestamp)
            X_train = X[feature_columns]
            model.fit(X_train, y)
            model_metadata = {
                "model": model,
                "trained_at": datetime.utcnow(),
                "n_samples": len(X),
                "feature_columns": feature_columns,
            }

            # Add model-specific metadata
            if model_type == "random_forest":
                # Extract feature importance
                if hasattr(model, "model_") and hasattr(model.model_, "feature_importances_"):
                    importances = model.model_.feature_importances_
                    # Sort by importance
                    indices = importances.argsort()[::-1]
                    model_metadata["feature_importance"] = {
                        "features": [feature_columns[i] for i in indices],
                        "importance": [float(importances[i]) for i in indices],
                    }

            elif model_type == "linear_regression":
                # Extract coefficients
                if hasattr(model, "model_"):
                    if hasattr(model.model_, "coef_"):
                        model_metadata["coefficients"] = [float(c) for c in model.model_.coef_]
                    if hasattr(model.model_, "intercept_"):
                        model_metadata["intercept"] = float(model.model_.intercept_)

            self.trained_models[model_key] = model_metadata

            logger.info(f"Successfully trained {model_key} with {len(X)} samples")
            return True

        except Exception as e:
            logger.error(f"Error training {model_key}: {e}")
            return False

    def generate_forecast(
        self,
        market_type: str,
        model_type: str,
        horizon_hours: int,
        forecast_start: Optional[datetime] = None,
        historical_window_hours: int = 0,
        training_config: Optional[TrainingDataConfig] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Generate a forecast for a specific market using a specific model.

        Args:
            market_type: Type of market
            model_type: Type of model
            horizon_hours: Number of hours to forecast ahead
            forecast_start: Starting timestamp for forecast (default: now)
            historical_window_hours: Number of hours to include before forecast_start (for RMSE calculation)
            training_config: Configuration for which data sources to use

        Returns:
            DataFrame with forecast or None if error
        """
        if training_config is None:
            training_config = TrainingDataConfig()

        # Update weather forecast if needed (smart caching - only updates if > 1 hour old)
        if training_config.use_weather_data:
            try:
                weather_manager = get_weather_manager()
                updated = weather_manager.update_weather_data(force=False)
                if updated:
                    logger.info("✓ Weather forecast updated with latest 24-hour data")
                else:
                    logger.debug("✓ Weather forecast is current, no update needed")
            except Exception as e:
                logger.warning(f"Could not update weather forecast: {e}")
                logger.warning("Proceeding with existing weather data")

        # Update day-ahead market data if needed (smart caching - only updates if > 12 hours old)
        if market_type == "day_ahead":
            try:
                updated = check_and_update_day_ahead_data(max_age_hours=12.0, force=False)
                if updated:
                    logger.info("✓ Day-ahead market data updated with latest ENTSO-E data")
                else:
                    logger.debug("✓ Day-ahead market data is current, no update needed")
            except Exception as e:
                logger.warning(f"Could not update day-ahead market data: {e}")
                logger.warning("Proceeding with existing market data")

        # Validate inputs
        if market_type not in MARKET_TYPES:
            logger.error(f"Invalid market type: {market_type}")
            return None

        if model_type not in MODEL_TYPES:
            logger.error(f"Invalid model type: {model_type}")
            return None

        if horizon_hours <= 0 or horizon_hours > settings.max_forecast_horizon_hours:
            logger.error(
                f"Invalid horizon: {horizon_hours} "
                f"(must be 1-{settings.max_forecast_horizon_hours})"
            )
            return None

        # Determine forecast start time
        if forecast_start is None:
            # Use the last available data point as the forecast start to avoid gaps
            df_market = load_market_data(market_type)
            if df_market is not None and not df_market.empty:
                # Get the most recent timestamp from actual data
                last_data_time = df_market["timestamp_utc"].max()
                # Forecast starts from the next interval (15 minutes after last data)
                forecast_start = last_data_time + pd.Timedelta(minutes=15)
                logger.info(
                    f"Using last available data point as forecast start: {last_data_time} -> {forecast_start}"
                )
            else:
                # Fallback to current time if no data available
                forecast_start = pd.Timestamp.utcnow().floor("15min")
                logger.warning("No historical data found, using current time as forecast start")
        else:
            forecast_start = pd.Timestamp(forecast_start)

        logger.info(
            f"Generating {horizon_hours}h forecast for {market_type} "
            f"using {model_type} starting from {forecast_start}"
            f"{f' with {historical_window_hours}h historical window' if historical_window_hours > 0 else ''}"
        )

        # Ensure model is trained with the specified configuration
        model_key = self._get_model_key(market_type, model_type, training_config)
        if model_key not in self.trained_models:
            logger.info(f"Model not cached, training {model_key}")
            success = self.train_model(market_type, model_type, training_config=training_config)
            if not success:
                return None

        # Get trained model
        model_info = self.trained_models[model_key]
        model = model_info["model"]
        feature_columns = model_info["feature_columns"]  # Get feature columns from cached model

        try:
            # Build weather features dict from config
            weather_features = None
            if training_config.use_weather_data:
                weather_features = {
                    "temperature": training_config.use_temperature,
                    "wind_speed": training_config.use_wind_speed,
                    "cloud_cover": training_config.use_cloud_cover,
                    "precipitation": training_config.use_precipitation,
                }

            all_forecasts = []

            # Generate historical window forecast if requested
            if historical_window_hours > 0:
                historical_start = forecast_start - pd.Timedelta(hours=historical_window_hours)
                X_historical = build_forecast_features(
                    market_type=market_type,
                    forecast_start=historical_start,
                    horizon_hours=historical_window_hours,
                    include_weather=training_config.use_weather_data,
                    weather_features=weather_features,
                )
                if not X_historical.empty:
                    # Extract timestamp and feature columns
                    timestamp_historical = X_historical["timestamp_utc"]
                    X_historical_features = X_historical[feature_columns]
                    
                    predictions_historical = model.predict(X_historical_features)
                    df_historical = pd.DataFrame(
                        {
                            "timestamp_utc": timestamp_historical,
                            "forecast_price_eur_per_mwh": predictions_historical,
                            "model_type": model_type,
                            "market_type": market_type,
                        }
                    )
                    all_forecasts.append(df_historical)

            # Build forecast features using same config as training
            X_forecast = build_forecast_features(
                market_type=market_type,
                forecast_start=forecast_start,
                horizon_hours=horizon_hours,
                include_weather=training_config.use_weather_data,
                weather_features=weather_features,
            )

            if X_forecast.empty:
                logger.error("Failed to build forecast features")
                return None

            # Extract timestamp and feature columns
            timestamp_forecast = X_forecast["timestamp_utc"]
            X_forecast_features = X_forecast[feature_columns]
            
            # Generate predictions
            predictions = model.predict(X_forecast_features)

            # Build result DataFrame for future forecast
            df_future = pd.DataFrame(
                {
                    "timestamp_utc": timestamp_forecast,
                    "forecast_price_eur_per_mwh": predictions,
                    "model_type": model_type,
                    "market_type": market_type,
                }
            )
            all_forecasts.append(df_future)

            # Combine historical and future forecasts
            result_df = pd.concat(all_forecasts, ignore_index=True)

            logger.info(f"Generated {len(result_df)} forecast values")

            return result_df

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return None

    def calculate_forecast_rmse(
        self,
        forecast_df: pd.DataFrame,
        market_type: str,
    ) -> Optional[tuple[float, int]]:
        """
        Calculate RMSE for a forecast against actual historical data.

        Args:
            forecast_df: DataFrame with forecast predictions
            market_type: Type of market to get actual data for

        Returns:
            Tuple of (RMSE value, number of data points) or None if no overlap with historical data
        """
        from app.services.data_store import load_market_data
        import numpy as np

        # Load actual market data
        df_actual = load_market_data(market_type)
        if df_actual is None or df_actual.empty:
            return None

        target_column = MARKET_TYPES[market_type]["target_column"]

        # Ensure timezone aware
        if df_actual["timestamp_utc"].dt.tz is None:
            df_actual["timestamp_utc"] = pd.to_datetime(df_actual["timestamp_utc"], utc=True)
        if forecast_df["timestamp_utc"].dt.tz is None:
            forecast_df["timestamp_utc"] = pd.to_datetime(forecast_df["timestamp_utc"], utc=True)

        # Merge on timestamp to find overlapping data
        merged = forecast_df.merge(
            df_actual[["timestamp_utc", target_column]], on="timestamp_utc", how="inner"
        )

        if merged.empty:
            return None

        # Calculate RMSE
        squared_errors = (merged["forecast_price_eur_per_mwh"] - merged[target_column]) ** 2
        rmse = np.sqrt(squared_errors.mean())
        n_points = len(merged)

        logger.info(f"RMSE calculated on {n_points} overlapping points: {rmse:.2f}")

        return (rmse, n_points)

    def compare_models(
        self,
        market_type: str,
        model_types: list,
        horizon_hours: int,
        forecast_start: Optional[datetime] = None,
        historical_window_hours: int = 0,
        training_config: Optional[TrainingDataConfig] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Generate forecasts using multiple models for comparison.

        Args:
            market_type: Type of market
            model_types: List of model types to compare
            horizon_hours: Number of hours to forecast
            forecast_start: Starting timestamp for forecast
            historical_window_hours: Number of hours to include before forecast_start (for RMSE calculation)
            training_config: Configuration for which data sources to use

        Returns:
            Combined DataFrame with forecasts from all models
        """
        all_forecasts = []

        for model_type in model_types:
            df_forecast = self.generate_forecast(
                market_type=market_type,
                model_type=model_type,
                horizon_hours=horizon_hours,
                forecast_start=forecast_start,
                historical_window_hours=historical_window_hours,
                training_config=training_config,
            )

            if df_forecast is not None:
                all_forecasts.append(df_forecast)

        if not all_forecasts:
            logger.error("No forecasts generated")
            return None

        # Combine all forecasts
        df_combined = pd.concat(all_forecasts, ignore_index=True)

        logger.info(f"Compared {len(model_types)} models with {horizon_hours}h horizon")

        return df_combined

    def get_model_info(
        self,
        market_type: str,
        model_type: str,
        training_config: Optional[TrainingDataConfig] = None,
    ) -> Optional[dict]:
        """
        Get information about a trained model.

        Args:
            market_type: Type of market
            model_type: Type of model
            training_config: Configuration used for training

        Returns:
            Dictionary with model information or None
        """
        model_key = self._get_model_key(market_type, model_type, training_config)

        if model_key not in self.trained_models:
            return None

        info = self.trained_models[model_key].copy()
        info.pop("model", None)  # Don't return the model object itself

        return info

    def get_training_data(
        self,
        market_type: str,
        model_type: str,
        training_config: Optional[TrainingDataConfig] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get the training data that was used (or would be used) for a model.

        Args:
            market_type: Type of market
            model_type: Type of model
            training_config: Configuration for which data sources to use

        Returns:
            DataFrame with training data (timestamp_utc and price) or None
        """
        if training_config is None:
            training_config = TrainingDataConfig()

        from app.services.data_store import load_market_data

        # Load market data
        df = load_market_data(market_type)
        if df is None or df.empty:
            return None

        # Get target column
        target_column = MARKET_TYPES[market_type]["target_column"]

        # Return timestamp and target only
        return df[["timestamp_utc", target_column]].copy()


# Global forecast service instance
_forecast_service: Optional[ForecastService] = None


def get_forecast_service() -> ForecastService:
    """
    Get the global forecast service instance.

    Returns:
        ForecastService instance
    """
    global _forecast_service

    if _forecast_service is None:
        _forecast_service = ForecastService()

    return _forecast_service
