# %% [markdown]
# # Power Market Forecasting - Interactive Exploration
#
# This notebook allows you to:
# - **Reuse the app's production forecasting code** (no duplicated logic)
# - **Load the same data** from `/data` directory
# - **Train and evaluate models** interactively in VS Code
# - **Visualize results** with matplotlib (no Streamlit needed)
# - **Experiment with different models, markets, and hyperparameters**
#
# All imports come directly from `app.services` and `app.models` - this notebook is a thin
# interactive wrapper around the production codebase.

# %% [markdown]
# ## 1. Environment Setup & Imports

# %%
# Core libraries
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up project root (adjust if running from different location)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# App imports - reuse production code
from app.core.config import settings, MARKET_TYPES, MODEL_TYPES
from app.services.data_store import load_market_data, load_weather_data
from app.services.feature_engineering import build_features_and_target, build_forecast_features
from app.services.forecast_service import ForecastService, TrainingDataConfig
from app.models.persistence_model import PersistencePriceModel
from app.models.linear_regression_model import LinearRegressionPriceModel
from app.models.random_forest_model import RandomForestPriceModel
from app.models.xgboost_classifier import RegulationStateXGBModel

# Sklearn utilities
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# Configure plotting
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
warnings.filterwarnings("ignore")

print("✓ Imports successful")
print(f"✓ Project root: {project_root}")
print(f"✓ Data directory: {settings.data_dir}")


# %%
print("=" * 60)
print("AVAILABLE MARKET TYPES")
print("=" * 60)
for market_key, market_config in MARKET_TYPES.items():
    target_type = market_config.get("target_type", "regression")
    data_file = market_config.get("data_file", "N/A")
    print(f"\n{market_key}:")
    print(f"  Target Type: {target_type}")
    print(f"  Data File: {data_file}")
    if "class_labels" in market_config:
        print(f"  Classes: {list(market_config['class_labels'].values())}")

print("\n" + "=" * 60)
print("AVAILABLE MODEL TYPES")
print("=" * 60)
for model_key, model_config in MODEL_TYPES.items():
    print(f"\n{model_key}:")
    print(f"  Name: {model_config.get('name', 'N/A')}")
    print(f"  Description: {model_config.get('description', 'N/A')}")

# %%
# Load all available datasets
print("Loading datasets...")

# Day-ahead market data
df_day_ahead = load_market_data("day_ahead")
print(f"\n✓ Day-ahead data: {len(df_day_ahead) if df_day_ahead is not None else 0} records")
if df_day_ahead is not None:
    print(
        f"  Date range: {df_day_ahead['timestamp_utc'].min()} to {df_day_ahead['timestamp_utc'].max()}"
    )
    target_col = MARKET_TYPES["day_ahead"]["target_column"]
    if target_col in df_day_ahead.columns:
        print(
            f"  Price range: {df_day_ahead[target_col].min():.2f} to {df_day_ahead[target_col].max():.2f} EUR/MWh"
        )

# Imbalance shortage (up-regulation price)
df_shortage = load_market_data("imbalance_shortage")
print(f"\n✓ Imbalance shortage data: {len(df_shortage) if df_shortage is not None else 0} records")
if df_shortage is not None:
    print(
        f"  Date range: {df_shortage['timestamp_utc'].min()} to {df_shortage['timestamp_utc'].max()}"
    )
    target_col = MARKET_TYPES["imbalance_shortage"]["target_column"]
    if target_col in df_shortage.columns:
        print(
            f"  Price range: {df_shortage[target_col].min():.2f} to {df_shortage[target_col].max():.2f} EUR/MWh"
        )

# Imbalance surplus (down-regulation price)
df_surplus = load_market_data("imbalance_surplus")
print(f"\n✓ Imbalance surplus data: {len(df_surplus) if df_surplus is not None else 0} records")
if df_surplus is not None:
    print(
        f"  Date range: {df_surplus['timestamp_utc'].min()} to {df_surplus['timestamp_utc'].max()}"
    )
    target_col = MARKET_TYPES["imbalance_surplus"]["target_column"]
    if target_col in df_surplus.columns:
        print(
            f"  Price range: {df_surplus[target_col].min():.2f} to {df_surplus[target_col].max():.2f} EUR/MWh"
        )

# Regulation state (classification target)
df_regulation = load_market_data("regulation_state")
print(
    f"\n✓ Regulation state data: {len(df_regulation) if df_regulation is not None else 0} records"
)
if df_regulation is not None:
    print(
        f"  Date range: {df_regulation['timestamp_utc'].min()} to {df_regulation['timestamp_utc'].max()}"
    )
    print(f"  State distribution:")
    target_col = MARKET_TYPES["regulation_state"]["target_column"]
    if target_col in df_regulation.columns:
        print(df_regulation[target_col].value_counts().sort_index())

# Weather data
df_weather = load_weather_data()
print(f"\n✓ Weather data: {len(df_weather) if df_weather is not None else 0} records")
if df_weather is not None:
    # Weather data uses timestamp as index, not as a column
    if isinstance(df_weather.index, pd.DatetimeIndex):
        print(f"  Date range: {df_weather.index.min()} to {df_weather.index.max()}")
    print(f"  Available features: {list(df_weather.columns)}")


# %%
def plot_timeseries(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str = None,
    figsize: Tuple[int, int] = (14, 5),
    sample: Optional[int] = None,
) -> None:
    """
    Plot a time series with clean formatting.

    Args:
        df: DataFrame with time series data
        x_col: Column name for x-axis (timestamp)
        y_col: Column name for y-axis (values)
        title: Plot title
        ylabel: Y-axis label (default: y_col)
        figsize: Figure size tuple
        sample: If provided, randomly sample this many points for faster plotting
    """
    plot_df = df.copy()
    if sample and len(plot_df) > sample:
        plot_df = plot_df.sample(n=sample).sort_values(x_col)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(plot_df[x_col], plot_df[y_col], linewidth=1.5, alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(ylabel or y_col, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_predictions_comparison(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    ylabel: str = "Value",
    sample: Optional[int] = None,
) -> None:
    """
    Plot actual vs predicted values over time.

    Args:
        timestamps: Timestamp series
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        ylabel: Y-axis label
        sample: If provided, sample this many points for faster plotting
    """
    df_plot = pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual": y_true,
            "predicted": y_pred,
        }
    ).sort_values("timestamp")

    if sample and len(df_plot) > sample:
        df_plot = df_plot.sample(n=sample).sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_plot["timestamp"], df_plot["actual"], label="Actual", linewidth=1.5, alpha=0.9)
    ax.plot(
        df_plot["timestamp"],
        df_plot["predicted"],
        label="Predicted",
        linewidth=1.5,
        alpha=0.7,
        linestyle="--",
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestamp", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 20,
) -> None:
    """
    Plot feature importance as horizontal bar chart.

    Args:
        feature_names: List of feature names
        importance_values: Array of importance values
        title: Plot title
        top_n: Number of top features to display
    """
    # Create DataFrame and sort
    df_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance_values,
            }
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(df_importance["feature"], df_importance["importance"])
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_classification_states(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Dict[int, str],
    title: str,
    sample: Optional[int] = 2000,
) -> None:
    """
    Plot classification states over time (scatter plot with colors).

    Args:
        timestamps: Timestamp series
        y_true: Actual class labels
        y_pred: Predicted class labels
        class_labels: Mapping from numeric labels to names
        title: Plot title
        sample: Number of points to sample
    """
    df_plot = pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual": y_true,
            "predicted": y_pred,
        }
    ).sort_values("timestamp")

    if sample and len(df_plot) > sample:
        df_plot = df_plot.sample(n=sample).sort_values("timestamp")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot actual states
    for state_num, state_name in class_labels.items():
        mask = df_plot["actual"] == state_num
        ax1.scatter(
            df_plot.loc[mask, "timestamp"],
            df_plot.loc[mask, "actual"],
            label=state_name,
            alpha=0.6,
            s=20,
        )
    ax1.set_ylabel("Actual State", fontsize=11)
    ax1.set_title(f"{title} - Actual", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot predicted states
    for state_num, state_name in class_labels.items():
        mask = df_plot["predicted"] == state_num
        ax2.scatter(
            df_plot.loc[mask, "timestamp"],
            df_plot.loc[mask, "predicted"],
            label=state_name,
            alpha=0.6,
            s=20,
        )
    ax2.set_ylabel("Predicted State", fontsize=11)
    ax2.set_xlabel("Timestamp", fontsize=11)
    ax2.set_title(f"{title} - Predicted", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
print("✓ Visualization helpers defined")

# Plot day-ahead prices (if available)
if df_day_ahead is not None and len(df_day_ahead) > 0:
    target_col = MARKET_TYPES["day_ahead"]["target_column"]
    if target_col in df_day_ahead.columns:
        plot_timeseries(
            df_day_ahead,
            x_col="timestamp_utc",
            y_col=target_col,
            title="Day-Ahead Market Prices (EUR/MWh)",
            ylabel="Price (EUR/MWh)",
            sample=5000,
        )

# %%
# Plot imbalance shortage prices (if available)
if df_shortage is not None and len(df_shortage) > 0:
    target_col = MARKET_TYPES["imbalance_shortage"]["target_column"]
    if target_col in df_shortage.columns:
        plot_timeseries(
            df_shortage,
            x_col="timestamp_utc",
            y_col=target_col,
            title="Imbalance Shortage Prices - Up-Regulation (EUR/MWh)",
            ylabel="Price (EUR/MWh)",
            sample=5000,
        )

# %%
# Plot imbalance surplus prices (if available)
if df_surplus is not None and len(df_surplus) > 0:
    target_col = MARKET_TYPES["imbalance_surplus"]["target_column"]
    if target_col in df_surplus.columns:
        plot_timeseries(
            df_surplus,
            x_col="timestamp_utc",
            y_col=target_col,
            title="Imbalance Surplus Prices - Down-Regulation (EUR/MWh)",
            ylabel="Price (EUR/MWh)",
            sample=5000,
        )

# %%
# Plot regulation state distribution over time (if available)
target_col = MARKET_TYPES["regulation_state"]["target_column"]
if df_regulation is not None and len(df_regulation) > 0 and target_col in df_regulation.columns:
    # Get class labels
    class_labels = MARKET_TYPES["regulation_state"].get("class_labels", {})

    fig, ax = plt.subplots(figsize=(14, 5))
    sample_df = df_regulation.sample(n=min(3000, len(df_regulation))).sort_values("timestamp_utc")

    for state_num, state_name in class_labels.items():
        mask = sample_df[target_col] == state_num
        ax.scatter(
            sample_df.loc[mask, "timestamp_utc"],
            sample_df.loc[mask, target_col],
            label=state_name,
            alpha=0.6,
            s=20,
        )

    ax.set_title("Regulation State Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestamp", fontsize=11)
    ax.set_ylabel("State", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %%
# ==============================================================================
# EXPERIMENT CONFIGURATION - EDIT THESE VALUES
# ==============================================================================

# Choose market type:
# - "day_ahead" → hourly day-ahead prices (regression)
# - "imbalance_shortage" → 15-min up-regulation prices (regression)
# - "imbalance_surplus" → 15-min down-regulation prices (regression)
# - "regulation_state" → 15-min regulation state classification (4 classes)
market_type = "imbalance_shortage"

# Choose model type:
# - "persistence" → naive baseline (last known value)
# - "linear_regression" → linear model (regression only)
# - "random_forest" → random forest (regression only)
# - "xgboost_classifier" → XGBoost classifier (regulation_state only)
model_type = "random_forest"

# Model hyperparameters (passed to model constructor)
# Examples:
# - RandomForest: {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5}
# - XGBoost: {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
model_hyperparams = {
    "n_estimators": 100,
    "max_depth": 8,
}

# Training data configuration
training_config = TrainingDataConfig(
    use_weather_data=True,  # Include weather features
    use_time_features=True,  # Include hour, day of week, etc.
    use_temperature=True,
    use_wind_speed=True,
    use_cloud_cover=True,
    use_precipitation=True,
)

# Time windows for training and testing
# Format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
train_start = "2024-10-01"
train_end = "2025-10-01"
test_start = "2025-10-01"
test_end = "2025-10-15"

# Forecast horizon (hours) - for service-based forecast
forecast_horizon_hours = 24

# ==============================================================================

print("=" * 60)
print("EXPERIMENT CONFIGURATION")
print("=" * 60)
print(f"Market Type: {market_type}")
print(f"Model Type: {model_type}")
print(f"Model Hyperparameters: {model_hyperparams}")
print(f"Training Period: {train_start} to {train_end}")
print(f"Test Period: {test_start} to {test_end}")
print(f"Weather Features: {training_config.use_weather_data}")
print(f"Time Features: {training_config.use_time_features}")
print("=" * 60)

# %%
print("Building features using production feature engineering code...")

# Build weather features dict from config
weather_features = None
if training_config.use_weather_data:
    weather_features = {
        "temperature": training_config.use_temperature,
        "wind_speed": training_config.use_wind_speed,
        "cloud_cover": training_config.use_cloud_cover,
        "precipitation": training_config.use_precipitation,
    }

# Build features and targets using production code
X, y = build_features_and_target(
    market_type=market_type,
    lag_hours=None,  # Uses production defaults: [24, 48, 168]
    include_weather=training_config.use_weather_data,
    weather_features=weather_features,
)

if X is None or y is None:
    raise ValueError(f"Failed to build features for market_type={market_type}")

print(f"\n✓ Features built successfully")
print(f"  Total samples: {len(X)}")
print(f"  Features: {X.shape[1]}")
print(f"  Feature columns: {list(X.columns)}")

# Extract timestamp for train/test split
if "timestamp_utc" not in X.columns:
    raise ValueError("timestamp_utc not found in features - cannot perform time-based split")

timestamps = pd.to_datetime(X["timestamp_utc"])

# Create train/test masks based on time windows
mask_train = (timestamps >= pd.Timestamp(train_start)) & (timestamps < pd.Timestamp(train_end))
mask_test = (timestamps >= pd.Timestamp(test_start)) & (timestamps < pd.Timestamp(test_end))

# Extract feature columns (exclude timestamp)
feature_columns = [col for col in X.columns if col != "timestamp_utc"]

# Split data
X_train = X.loc[mask_train, feature_columns]
y_train = y[mask_train]
timestamps_train = timestamps[mask_train]

X_test = X.loc[mask_test, feature_columns]
y_test = y[mask_test]
timestamps_test = timestamps[mask_test]

print(f"\n✓ Train/test split completed")
print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"  Training period: {timestamps_train.min()} to {timestamps_train.max()}")
print(f"  Test period: {timestamps_test.min()} to {timestamps_test.max()}")

# Check for data leakage
if timestamps_train.max() >= timestamps_test.min():
    print("\n⚠️  WARNING: Potential data leakage - train and test periods overlap!")

# %%
print(f"Training {model_type} model for {market_type}...")

# Determine if this is a classification or regression task
market_config = MARKET_TYPES[market_type]
is_classification = market_config.get("target_type") == "classification"

# Create model instance based on configuration
if is_classification:
    if model_type != "xgboost_classifier":
        raise ValueError(f"Model {model_type} cannot be used for classification task {market_type}")
    model = RegulationStateXGBModel(**model_hyperparams)
else:
    # Regression models
    if model_type == "persistence":
        model = PersistencePriceModel()
    elif model_type == "linear_regression":
        model = LinearRegressionPriceModel()
    elif model_type == "random_forest":
        model = RandomForestPriceModel(**model_hyperparams)
    else:
        raise ValueError(f"Unsupported model type for regression: {model_type}")

# Train the model
print(f"  Fitting model on {len(X_train)} training samples...")
model.fit(X_train, y_train)

# Generate predictions
print(f"  Generating predictions...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"\n✓ Model training completed")
print(f"  Training predictions shape: {y_pred_train.shape}")
print(f"  Test predictions shape: {y_pred_test.shape}")
print(f"  Sample train predictions: {y_pred_train[:5]}")
print(f"  Sample test predictions: {y_pred_test[:5]}")

# For classification, also get probability predictions
if is_classification:
    y_proba_train = model.predict_proba(X_train)
    y_proba_test = model.predict_proba(X_test)
    print(f"  Probability predictions shape: {y_proba_test.shape}")
    print(f"  Sample test probabilities:\n{y_proba_test[:3]}")
# %%
print("=" * 60)
print("MODEL EVALUATION METRICS")
print("=" * 60)

if is_classification:
    # Classification metrics
    print("\n--- TRAINING SET ---")
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"Accuracy: {train_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_train, y_pred_train, zero_division=0))

    print("\n--- TEST SET ---")
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix - Test Set", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    # Add class labels if available
    class_labels = market_config.get("class_labels", {})
    if class_labels:
        labels = [class_labels.get(i, str(i)) for i in sorted(class_labels.keys())]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    plt.tight_layout()
    plt.show()

else:
    # Regression metrics
    print("\n--- TRAINING SET ---")
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_rmse = np.sqrt(train_mse)  # Calculate RMSE manually for compatibility
    train_r2 = r2_score(y_train, y_pred_train)
    print(f"MAE:  {train_mae:.3f}")
    print(f"RMSE: {train_rmse:.3f}")
    print(f"R²:   {train_r2:.4f}")

    print("\n--- TEST SET ---")
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)  # Calculate RMSE manually for compatibility
    test_r2 = r2_score(y_test, y_pred_test)
    print(f"MAE:  {test_mae:.3f}")
    print(f"RMSE: {test_rmse:.3f}")
    print(f"R²:   {test_r2:.4f}")

    # Plot actual vs predicted scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training set
    ax1.scatter(y_train, y_pred_train, alpha=0.3, s=10)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2)
    ax1.set_xlabel("Actual", fontsize=11)
    ax1.set_ylabel("Predicted", fontsize=11)
    ax1.set_title(f"Training Set (R²={train_r2:.3f})", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Test set
    ax2.scatter(y_test, y_pred_test, alpha=0.3, s=10)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax2.set_xlabel("Actual", fontsize=11)
    ax2.set_ylabel("Predicted", fontsize=11)
    ax2.set_title(f"Test Set (R²={test_r2:.3f})", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

print("\n" + "=" * 60)

# %%
if is_classification:
    # Plot classification states over time
    class_labels = MARKET_TYPES[market_type].get("class_labels", {})

    plot_classification_states(
        timestamps=timestamps_test,
        y_true=y_test.values,
        y_pred=y_pred_test,
        class_labels=class_labels,
        title=f"{market_type} - {model_type} (Test Set)",
        sample=2000,
    )
else:
    # Plot regression predictions over time
    plot_predictions_comparison(
        timestamps=timestamps_test,
        y_true=y_test.values,
        y_pred=y_pred_test,
        title=f"{market_type} - {model_type} - Test Set Predictions",
        ylabel="Price (EUR/MWh)",
        sample=2000,
    )


# %%
# Extract feature importance (if model supports it)
try:
    if hasattr(model, "feature_importances_"):
        # Tree-based models (RandomForest, XGBoost)
        importances = model.feature_importances_
        plot_feature_importance(
            feature_names=feature_columns,
            importance_values=importances,
            title=f"Feature Importance - {model_type}",
            top_n=20,
        )
    elif hasattr(model, "coef_"):
        # Linear models
        coefficients = np.abs(model.coef_)
        plot_feature_importance(
            feature_names=feature_columns,
            importance_values=coefficients,
            title=f"Feature Coefficients (Absolute) - {model_type}",
            top_n=20,
        )
    else:
        print(f"Model {model_type} does not support feature importance extraction")
except Exception as e:
    print(f"Error extracting feature importance: {e}")


# %%
print("=" * 60)
print("GENERATING FORECAST USING PRODUCTION SERVICE")
print("=" * 60)

# Initialize the forecast service
forecast_service = ForecastService()

# Generate forecast using the same service as the Streamlit app
forecast_result = forecast_service.generate_forecast(
    market_type=market_type,
    model_type=model_type,
    horizon_hours=forecast_horizon_hours,
    forecast_start=None,  # Uses latest available data
    historical_window_hours=72,  # Include 72 hours of historical data
    training_config=training_config,
)

if forecast_result is not None:
    print(f"\n✓ Forecast generated successfully")
    print(f"  Forecast shape: {forecast_result.shape}")
    print(
        f"  Forecast period: {forecast_result['timestamp_utc'].min()} to {forecast_result['timestamp_utc'].max()}"
    )
    print(f"\nForecast head:")
    print(forecast_result.head(10))

    # Plot forecast
    if is_classification:
        # For classification, plot the predicted states
        if "prediction" in forecast_result.columns:
            class_labels = MARKET_TYPES[market_type].get("class_labels", {})

            fig, ax = plt.subplots(figsize=(14, 5))
            for state_num, state_name in class_labels.items():
                mask = forecast_result["prediction"] == state_num
                ax.scatter(
                    forecast_result.loc[mask, "timestamp_utc"],
                    forecast_result.loc[mask, "prediction"],
                    label=state_name,
                    alpha=0.7,
                    s=30,
                )
            ax.set_title(
                f"Forecast: {market_type} - {model_type} ({forecast_horizon_hours}h ahead)",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("Timestamp", fontsize=11)
            ax.set_ylabel("Predicted State", fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    else:
        # For regression, plot the predicted prices
        if "prediction" in forecast_result.columns:
            plot_timeseries(
                forecast_result,
                x_col="timestamp_utc",
                y_col="prediction",
                title=f"Forecast: {market_type} - {model_type} ({forecast_horizon_hours}h ahead)",
                ylabel="Predicted Price (EUR/MWh)",
            )
else:
    print("\n❌ Forecast generation failed")


# %%
# Add your custom analysis here
# Examples:
# - Compare multiple models
# - Test different hyperparameter combinations
# - Analyze error patterns
# - Cross-validation
# - Residual analysis
# - Feature engineering experiments

print("This cell is reserved for your custom experiments")
print(f"Available data:")
print(f"  - X_train, y_train ({len(X_train)} samples)")
print(f"  - X_test, y_test ({len(X_test)} samples)")
print(f"  - Trained model: {model}")
print(f"  - Feature columns: {len(feature_columns)} features")
print(f"\nReady for experimentation! 🚀")


# %%
# Create results directory if it doesn't exist
results_dir = project_root / "notebooks" / "results"
results_dir.mkdir(exist_ok=True)

# Export test predictions
output_df = pd.DataFrame(
    {
        "timestamp_utc": timestamps_test,
        "actual": y_test.values,
        "predicted": y_pred_test,
    }
)

if is_classification and "y_proba_test" in locals():
    # Add probability columns for classification
    class_labels = MARKET_TYPES[market_type].get("class_labels", {})
    for i, (state_num, state_name) in enumerate(sorted(class_labels.items())):
        output_df[f"prob_{state_name}"] = y_proba_test[:, i]

# Save to CSV
output_filename = (
    f"{market_type}_{model_type}_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
output_path = results_dir / output_filename
output_df.to_csv(output_path, index=False)

print(f"✓ Predictions saved to: {output_path}")
print(f"  Rows: {len(output_df)}")
print(f"  Columns: {list(output_df.columns)}")

# %% [markdown]
# ---
# # End of Notebook
#
# **Key Takeaways:**
# - All code reuses production modules from `app.services` and `app.models`
# - No duplicated feature engineering or model logic
# - Easy to experiment with different markets, models, and hyperparameters
# - Results can be exported for further analysis
#
# **Next Steps:**
# - Modify configuration in Section 6 to try different experiments
# - Add custom analysis in Section 13
# - Compare multiple models by running cells repeatedly with different configs
# - Use results for model selection and hyperparameter tuning
