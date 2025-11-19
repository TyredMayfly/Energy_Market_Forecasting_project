"""
Streamlit web application for power market forecasting - Redesigned UI.

Features a clean, top-aligned control panel with comprehensive forecasting configuration.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.config import MARKET_TYPES, MODEL_TYPES, settings
from app.services.data_store import get_data_summary, load_market_data, load_weather_data
from app.services.forecast_service import get_forecast_service, TrainingDataConfig

# Setup logger
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Energy Market Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def get_model_hyperparameters_ui(model_type: str) -> Dict:
    """
    Render model-specific hyperparameter controls and return their values.

    Args:
        model_type: Selected model type

    Returns:
        Dictionary of hyperparameter values
    """
    hyperparams = {}

    if model_type == "persistence":
        st.markdown("**Persistence Model Settings**")
        hyperparams["window_length"] = st.slider(
            "Lookback window (intervals)",
            min_value=1,
            max_value=96,  # 24 hours at 15-min resolution
            value=4,  # 1 hour default
            help="Number of recent intervals to use for persistence forecast",
        )

    elif model_type == "linear_regression":
        st.markdown("**Linear Regression Settings**")
        col1, col2 = st.columns(2)
        with col1:
            hyperparams["fit_intercept"] = st.checkbox(
                "Fit intercept",
                value=True,
                help="Whether to calculate the intercept for this model",
            )
        with col2:
            hyperparams["normalize"] = st.checkbox(
                "Normalize features",
                value=False,
                help="Normalize features before fitting (deprecated in newer sklearn)",
            )

    elif model_type == "random_forest":
        st.markdown("**Random Forest Settings**")
        col1, col2 = st.columns(2)
        with col1:
            hyperparams["n_estimators"] = st.slider(
                "Number of trees",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Number of trees in the forest",
            )
            hyperparams["min_samples_leaf"] = st.slider(
                "Min samples per leaf",
                min_value=1,
                max_value=20,
                value=1,
                help="Minimum number of samples required at a leaf node",
            )
        with col2:
            max_depth_enabled = st.checkbox(
                "Limit tree depth", value=False, help="Enable maximum depth constraint"
            )
            if max_depth_enabled:
                hyperparams["max_depth"] = st.slider(
                    "Max depth",
                    min_value=5,
                    max_value=50,
                    value=20,
                    help="Maximum depth of each tree",
                )
            else:
                hyperparams["max_depth"] = None

            hyperparams["random_state"] = st.number_input(
                "Random seed",
                min_value=0,
                max_value=10000,
                value=42,
                help="Random state for reproducibility",
            )

    elif model_type == "xgboost_classifier":
        st.markdown("**XGBoost Classifier Settings**")
        col1, col2 = st.columns(2)
        with col1:
            hyperparams["n_estimators"] = st.slider(
                "Number of boosting rounds",
                min_value=50,
                max_value=300,
                value=100,
                step=10,
                help="Number of gradient boosted trees",
            )
            hyperparams["max_depth"] = st.slider(
                "Max tree depth", min_value=3, max_value=15, value=6, help="Maximum depth of trees"
            )
        with col2:
            hyperparams["learning_rate"] = st.slider(
                "Learning rate",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Step size shrinkage to prevent overfitting",
            )
            hyperparams["random_state"] = st.number_input(
                "Random seed",
                min_value=0,
                max_value=10000,
                value=42,
                help="Random state for reproducibility",
            )

    return hyperparams


def get_forecast_configuration() -> Optional[Dict]:
    """
    Render the top-aligned forecast configuration panel and collect user inputs.

    Returns:
        Dictionary containing all configuration parameters, or None if "Run Forecast" not clicked
    """
    st.title("⚡ Energy Market Forecasting - Netherlands")

    st.markdown("---")

    # Configuration container
    with st.container():
        st.header("⚙️ Forecast Configuration")

        # === ROW 1: Market & Model Selection ===
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1️⃣ Energy Market")
            market_type = st.selectbox(
                "Select market type",
                options=list(MARKET_TYPES.keys()),
                format_func=lambda x: MARKET_TYPES[x]["display_name"],
                help="Choose which electricity market to forecast",
                key="market_select",
            )
            st.caption(f"ℹ️ {MARKET_TYPES[market_type]['description']}")

        with col2:
            st.subheader("2️⃣ Forecasting Model")

            # Filter models based on target type
            target_type = MARKET_TYPES[market_type].get("target_type", "regression")

            # Define which models support which target types
            regression_models = ["persistence", "linear_regression", "random_forest"]
            classification_models = ["xgboost_classifier"]

            if target_type == "classification":
                available_models = classification_models
            else:
                available_models = regression_models

            model_type = st.selectbox(
                "Select ML model",
                options=available_models,
                format_func=lambda x: MODEL_TYPES[x]["display_name"],
                help="Choose the forecasting algorithm (filtered by market type)",
                key="model_select",
            )
            st.caption(f"ℹ️ {MODEL_TYPES[model_type]['description']}")

        st.markdown("---")

        # === ROW 2: Model Configuration ===
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("3️⃣ Model Hyperparameters")
            hyperparameters = get_model_hyperparameters_ui(model_type)

        with col2:
            st.subheader("4️⃣ Training Data Sources")

            # Market data (always required)
            st.markdown("**Market Data** (always included)")
            use_market_data = True
            st.caption("✓ Historical price data and lag features")

            st.markdown("")  # Spacing

            # Time features
            use_time_features = st.checkbox(
                "Use time-based features",
                value=True,
                help="Include hour-of-day, day-of-week, weekend indicators, etc.",
                key="time_features",
            )

            # Weather data master toggle
            use_weather_data = st.checkbox(
                "Use weather data (KNMI)",
                value=True,
                help="Include meteorological features in model training",
                key="weather_master",
            )

            # Granular weather features (only if weather enabled)
            if use_weather_data:
                st.markdown("**Weather Features:**")
                wcol1, wcol2 = st.columns(2)
                with wcol1:
                    use_temperature = st.checkbox("🌡️ Temperature", value=True, key="temp")
                    use_cloud_cover = st.checkbox("☁️ Cloud Cover", value=True, key="cloud")
                with wcol2:
                    use_wind_speed = st.checkbox("💨 Wind Speed", value=True, key="wind")
                    use_precipitation = st.checkbox("🌧️ Precipitation", value=True, key="precip")
            else:
                use_temperature = use_wind_speed = use_cloud_cover = use_precipitation = False

            # Rolling statistics
            use_rolling_stats = st.checkbox(
                "Use rolling statistics",
                value=True,
                help="Include 24-hour rolling mean and standard deviation",
                key="rolling",
            )

        st.markdown("---")

        # === ROW 3: Forecast Settings & Display Options ===
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("5️⃣ Forecast Horizon")
            forecast_horizon_hours = st.slider(
                "Hours ahead",
                min_value=1,
                max_value=24,
                value=24,
                help="Number of hours to forecast into the future",
            )
            st.caption(f"📊 {forecast_horizon_hours * 4} intervals (15-min resolution)")

        with col2:
            st.subheader("6️⃣ Historical Context")
            historical_window_hours = st.slider(
                "Historical window (hours)",
                min_value=24,
                max_value=168,  # 7 days
                value=72,
                help="Hours of historical data to display on chart",
            )

        with col3:
            st.subheader("7️⃣ Chart Display Options")
            show_historical = st.checkbox(
                "📈 Show historical prices",
                value=True,
                help="Display recent historical data on chart",
            )
            show_training_data = st.checkbox(
                "🎯 Show training data", value=False, help="Highlight data used for model training"
            )
            show_weather_overlay = st.checkbox(
                "🌤️ Show weather overlay",
                value=False,
                help="Add weather variables to chart (secondary axis)",
            )
            show_feature_importance = st.checkbox(
                "📊 Show feature importance",
                value=False,
                help="Display feature importance (Random Forest only)",
                disabled=(model_type != "random_forest"),
            )

        st.markdown("---")

        # === Advanced Options ===
        with st.expander("🔧 Advanced Options"):
            compare_models = st.checkbox(
                "Compare all models",
                value=False,
                help="Generate forecasts from all three models for comparison",
            )

        # === Action Buttons ===
        st.markdown("")  # Spacing
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            run_forecast = st.button(
                "🚀 Run Forecast",
                type="primary",
                width="stretch",
                help="Generate forecast with current configuration",
            )

    # Return configuration if button clicked
    if run_forecast:
        training_data_options = {
            "use_market_data": use_market_data,
            "use_weather_data": use_weather_data,
            "use_time_features": use_time_features,
            "use_rolling_stats": use_rolling_stats,
            # Granular weather options
            "use_temperature": use_temperature,
            "use_wind_speed": use_wind_speed,
            "use_cloud_cover": use_cloud_cover,
            "use_precipitation": use_precipitation,
        }

        return {
            "market_type": market_type,
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "training_data_options": training_data_options,
            "forecast_horizon_hours": forecast_horizon_hours,
            "historical_window_hours": historical_window_hours,
            "show_historical": show_historical,
            "show_training_data": show_training_data,
            "show_weather_overlay": show_weather_overlay,
            "show_feature_importance": show_feature_importance,
            "compare_models": compare_models,
        }

    return None


def generate_forecast_with_config(
    config: Dict,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    """
    Generate forecast using the provided configuration.

    Args:
        config: Forecast configuration dictionary

    Returns:
        Tuple of (historical_df, training_df, forecast_df, metadata)
    """
    service = get_forecast_service()

    # Create training configuration
    training_config = TrainingDataConfig(
        use_weather_data=config["training_data_options"]["use_weather_data"],
        use_time_features=config["training_data_options"]["use_time_features"],
        use_temperature=config["training_data_options"]["use_temperature"],
        use_wind_speed=config["training_data_options"]["use_wind_speed"],
        use_cloud_cover=config["training_data_options"]["use_cloud_cover"],
        use_precipitation=config["training_data_options"]["use_precipitation"],
    )

    # Generate forecast(s)
    if config["compare_models"]:
        model_types_to_compare = list(MODEL_TYPES.keys())
        df_forecast = service.compare_models(
            market_type=config["market_type"],
            model_types=model_types_to_compare,
            horizon_hours=config["forecast_horizon_hours"],
            historical_window_hours=config["historical_window_hours"],
            training_config=training_config,
        )
    else:
        df_forecast = service.generate_forecast(
            market_type=config["market_type"],
            model_type=config["model_type"],
            horizon_hours=config["forecast_horizon_hours"],
            historical_window_hours=config["historical_window_hours"],
            training_config=training_config,
        )

    # Get training data if requested
    training_df = None
    if config["show_training_data"] and not config["compare_models"]:
        training_df = service.get_training_data(
            market_type=config["market_type"],
            model_type=config["model_type"],
            training_config=training_config,
        )

    # Get historical data if requested
    historical_df = None
    if config["show_historical"]:
        historical_df = load_market_data(config["market_type"])

    # Get metadata (model info, RMSE, etc.)
    metadata = {}
    if df_forecast is not None:
        if config["compare_models"]:
            metadata["models"] = {}
            for mtype in df_forecast["model_type"].unique():
                model_info = service.get_model_info(config["market_type"], mtype, training_config)
                if model_info:
                    metadata["models"][mtype] = model_info
        else:
            model_info = service.get_model_info(
                config["market_type"], config["model_type"], training_config
            )
            if model_info:
                metadata["model_info"] = model_info

        # Calculate RMSE
        rmse_result = service.calculate_forecast_rmse(df_forecast, config["market_type"])
        if rmse_result is not None:
            rmse, n_points = rmse_result
            metadata["rmse"] = rmse
            metadata["rmse_points"] = n_points

    return historical_df, training_df, df_forecast, metadata


def plot_forecast_results(
    forecast_df: pd.DataFrame,
    market_type: str,
    config: Dict,
    historical_df: Optional[pd.DataFrame] = None,
    training_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Create forecast plot with optional historical and training data overlays.
    Supports both regression (price) and classification (regulation_state) targets.

    Args:
        forecast_df: Forecast data
        market_type: Market type identifier
        config: Display configuration
        historical_df: Historical data (optional)
        training_df: Training data (optional)

    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    target_col = MARKET_TYPES[market_type]["target_column"]
    target_type = MARKET_TYPES[market_type].get("target_type", "regression")

    # For classification, use step plot instead of continuous lines
    is_classification = target_type == "classification"

    # === 1. Training Data (bottom layer) ===
    if config["show_training_data"] and training_df is not None and not training_df.empty:
        if training_df["timestamp_utc"].dt.tz is None:
            training_df["timestamp_utc"] = pd.to_datetime(training_df["timestamp_utc"], utc=True)

        logger.info(
            f"[PLOT] Training Data: {len(training_df)} points | "
            f"Start: {training_df['timestamp_utc'].min()} | "
            f"End: {training_df['timestamp_utc'].max()} | "
            f"Value range: {training_df[target_col].min():.2f}-{training_df[target_col].max():.2f}"
        )

        plot_mode = "lines" if not is_classification else "lines"
        plot_shape = "linear" if not is_classification else "hv"  # Step plot for classification

        fig.add_trace(
            go.Scatter(
                x=training_df["timestamp_utc"],
                y=training_df[target_col],
                mode=plot_mode,
                name="Training Data",
                line=dict(color="lightblue", width=2, dash="dot", shape=plot_shape),
                opacity=0.6,
            )
        )

    # === 2. Historical Data (middle layer) ===
    if config["show_historical"] and historical_df is not None and not historical_df.empty:
        if historical_df["timestamp_utc"].dt.tz is None:
            historical_df["timestamp_utc"] = pd.to_datetime(
                historical_df["timestamp_utc"], utc=True
            )

        # The forecast_df may include historical window data for RMSE calculation
        # Calculate the actual forecast start time (when historical data ends)
        # This is based on the most recent actual price data available
        forecast_start_time = historical_df["timestamp_utc"].max()
        cutoff_time = forecast_start_time - timedelta(hours=config["historical_window_hours"])

        # Get recent historical data
        df_recent = historical_df[
            (historical_df["timestamp_utc"] >= cutoff_time)
            & (historical_df["timestamp_utc"] <= forecast_start_time)
        ].copy()

        # Exclude training data to avoid overlap
        if config["show_training_data"] and training_df is not None:
            training_end = training_df["timestamp_utc"].max()
            df_recent = df_recent[df_recent["timestamp_utc"] > training_end]

        if not df_recent.empty:
            logger.info(
                f"[PLOT] Historical Data: {len(df_recent)} points | "
                f"Start: {df_recent['timestamp_utc'].min()} | "
                f"End: {df_recent['timestamp_utc'].max()} | "
                f"Window: {config['historical_window_hours']}h | "
                f"Value range: {df_recent[target_col].min():.2f}-{df_recent[target_col].max():.2f}"
            )

            plot_shape = "linear" if not is_classification else "hv"

            fig.add_trace(
                go.Scatter(
                    x=df_recent["timestamp_utc"],
                    y=df_recent[target_col],
                    mode="lines",
                    name="Historical",
                    line=dict(color="gray", width=2, shape=plot_shape),
                    opacity=0.7,
                )
            )

    # === 3. Forecast Data (top layer) ===
    # When showing historical window, forecast should overlay from the start of historical window
    # This allows comparing forecast vs actual for the historical period
    if config["show_historical"] and historical_df is not None and not historical_df.empty:
        # Show forecast from start of historical window up to end of forecast horizon
        forecast_start_time = historical_df["timestamp_utc"].max()
        cutoff_time = forecast_start_time - timedelta(hours=config["historical_window_hours"])
        forecast_end_time = forecast_start_time + timedelta(hours=config["forecast_horizon_hours"])

        # Filter forecast to match the exact time range
        # Use >= on lower bound to include the cutoff time itself
        forecast_df_display = forecast_df[
            (forecast_df["timestamp_utc"] >= cutoff_time)
            & (forecast_df["timestamp_utc"] <= forecast_end_time)
        ].copy()

        # If forecast doesn't start at cutoff_time, it means we need to use what's available
        if (
            not forecast_df_display.empty
            and forecast_df_display["timestamp_utc"].min() > cutoff_time
        ):
            # Forecast starts later than expected, use the earliest available
            earliest_available = forecast_df["timestamp_utc"].min()
            logger.warning(
                f"[PLOT] WARNING: Forecast starts later than expected cutoff. "
                f"Expected: {cutoff_time}, Actual: {earliest_available}"
            )
            forecast_df_display = forecast_df[
                (forecast_df["timestamp_utc"] >= earliest_available)
                & (forecast_df["timestamp_utc"] <= forecast_end_time)
            ].copy()
    else:
        # No historical data, show all forecast
        forecast_df_display = forecast_df.copy()

    if not forecast_df_display.empty:
        logger.info(
            f"[PLOT] Forecast Data: {len(forecast_df_display)} points | "
            f"Start: {forecast_df_display['timestamp_utc'].min()} | "
            f"End: {forecast_df_display['timestamp_utc'].max()} | "
            f"Horizon: {config['forecast_horizon_hours']}h | "
            f"Price range: {forecast_df_display['forecast_price_eur_per_mwh'].min():.2f}-{forecast_df_display['forecast_price_eur_per_mwh'].max():.2f} EUR/MWh"
        )

    if config["compare_models"]:
        colors = {"persistence": "blue", "linear_regression": "green", "random_forest": "red"}

        for mtype in forecast_df_display["model_type"].unique():
            df_model = forecast_df_display[forecast_df_display["model_type"] == mtype]
            fig.add_trace(
                go.Scatter(
                    x=df_model["timestamp_utc"],
                    y=df_model["forecast_price_eur_per_mwh"],
                    mode="lines+markers",
                    name=MODEL_TYPES[mtype]["display_name"],
                    line=dict(color=colors.get(mtype, "orange"), width=3),
                    marker=dict(size=6),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=forecast_df_display["timestamp_utc"],
                y=forecast_df_display["forecast_price_eur_per_mwh"],
                mode="lines+markers",
                name=f"{MODEL_TYPES[config['model_type']]['display_name']} Forecast",
                line=dict(color="blue", width=3),
                marker=dict(size=8),
            )
        )

    # === Layout ===
    y_axis_title = "Price (EUR/MWh)" if not is_classification else "Regulation State"

    fig.update_layout(
        title=f"{MARKET_TYPES[market_type]['display_name']} Forecast",
        xaxis_title="Time (UTC)",
        yaxis_title=y_axis_title,
        hovermode="x unified",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    # For classification, add discrete y-axis ticks with labels
    if is_classification and "class_labels" in MARKET_TYPES[market_type]:
        class_labels = MARKET_TYPES[market_type]["class_labels"]
        fig.update_yaxis(
            tickmode="array",
            tickvals=list(class_labels.keys()),
            ticktext=list(class_labels.values()),
        )

    # === 4. Weather Overlay (if requested) ===
    if config["show_weather_overlay"]:
        df_weather = load_weather_data()

        if df_weather is not None and not df_weather.empty:
            # Weather data uses timestamp as index
            weather_timestamps = df_weather.index
            if weather_timestamps.tz is None:
                df_weather.index = pd.to_datetime(df_weather.index, utc=True)

            # Filter to chart time range - should match historical data range
            if config["show_historical"] and historical_df is not None and not historical_df.empty:
                # Show weather for historical window + forecast period
                forecast_start_time = historical_df["timestamp_utc"].max()
                cutoff_time = forecast_start_time - timedelta(
                    hours=config["historical_window_hours"]
                )
                forecast_end = forecast_start_time + timedelta(
                    hours=config["forecast_horizon_hours"]
                )
            else:
                # No historical data - show weather for forecast period only
                cutoff_time = forecast_df_display["timestamp_utc"].min()
                forecast_end = forecast_df_display["timestamp_utc"].max()

            df_weather_filtered = df_weather[
                (df_weather.index >= cutoff_time) & (df_weather.index <= forecast_end)
            ].copy()

            if not df_weather_filtered.empty:
                logger.info(
                    f"[PLOT] Weather Data: {len(df_weather_filtered)} points | "
                    f"Start: {df_weather_filtered.index.min()} | "
                    f"End: {df_weather_filtered.index.max()} | "
                    f"Temp: {df_weather_filtered['temperature_deg_c'].min():.1f}-{df_weather_filtered['temperature_deg_c'].max():.1f}C | "
                    f"Wind: {df_weather_filtered['wind_speed_m_per_s'].min():.1f}-{df_weather_filtered['wind_speed_m_per_s'].max():.1f} m/s | "
                    f"Radiation: {df_weather_filtered['global_radiation_w_per_m2'].min():.0f}-{df_weather_filtered['global_radiation_w_per_m2'].max():.0f} W/m2 | "
                    f"Clouds: {df_weather_filtered['cloud_cover_pct'].min():.0f}-{df_weather_filtered['cloud_cover_pct'].max():.0f}% | "
                    f"Precip: {df_weather_filtered['precipitation_mm'].min():.1f}-{df_weather_filtered['precipitation_mm'].max():.1f} mm"
                )

                # Create single secondary y-axis for all weather variables
                fig.update_layout(
                    yaxis2=dict(
                        title="Weather Variables", overlaying="y", side="right", showgrid=False
                    ),
                    height=600,
                )

                # Normalize weather variables to 0-100 scale for better visualization
                # Temperature: normalize to roughly 0-100 (shift and scale)
                if "temperature_deg_c" in df_weather_filtered.columns:
                    temp_normalized = ((df_weather_filtered["temperature_deg_c"] + 20) / 60) * 100
                    temp_normalized = temp_normalized.clip(0, 100)
                    fig.add_trace(
                        go.Scatter(
                            x=df_weather_filtered.index,
                            y=temp_normalized,
                            mode="lines",
                            name="🌡️ Temperature",
                            line=dict(color="#FF6B6B", width=2),
                            opacity=0.6,
                            yaxis="y2",
                            hovertemplate="Temp: %{customdata:.1f}°C<extra></extra>",
                            customdata=df_weather_filtered["temperature_deg_c"],
                        )
                    )

                # Wind Speed: normalize to 0-100 (assuming max ~25 m/s)
                if "wind_speed_m_per_s" in df_weather_filtered.columns:
                    wind_normalized = (df_weather_filtered["wind_speed_m_per_s"] / 25) * 100
                    wind_normalized = wind_normalized.clip(0, 100)
                    fig.add_trace(
                        go.Scatter(
                            x=df_weather_filtered.index,
                            y=wind_normalized,
                            mode="lines",
                            name="💨 Wind Speed",
                            line=dict(color="#4ECDC4", width=2),
                            opacity=0.6,
                            yaxis="y2",
                            hovertemplate="Wind: %{customdata:.1f} m/s<extra></extra>",
                            customdata=df_weather_filtered["wind_speed_m_per_s"],
                        )
                    )

                # Solar Radiation: normalize to 0-100 (assuming max ~1000 W/m²)
                if "global_radiation_w_per_m2" in df_weather_filtered.columns:
                    radiation_normalized = (
                        df_weather_filtered["global_radiation_w_per_m2"] / 1000
                    ) * 100
                    radiation_normalized = radiation_normalized.clip(0, 100)
                    fig.add_trace(
                        go.Scatter(
                            x=df_weather_filtered.index,
                            y=radiation_normalized,
                            mode="lines",
                            name="☀️ Solar Radiation",
                            line=dict(color="#FFD93D", width=2),
                            opacity=0.6,
                            yaxis="y2",
                            fill="tozeroy",
                            fillcolor="rgba(255, 217, 61, 0.1)",
                            hovertemplate="Radiation: %{customdata:.0f} W/m²<extra></extra>",
                            customdata=df_weather_filtered["global_radiation_w_per_m2"],
                        )
                    )

                # Cloud Cover - already 0-100%
                if "cloud_cover_pct" in df_weather_filtered.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_weather_filtered.index,
                            y=df_weather_filtered["cloud_cover_pct"],
                            mode="lines",
                            name="☁️ Cloud Cover",
                            line=dict(color="#95A5A6", width=1.5),
                            opacity=0.5,
                            yaxis="y2",
                            fill="tozeroy",
                            fillcolor="rgba(149, 165, 166, 0.1)",
                            hovertemplate="Clouds: %{y:.0f}%<extra></extra>",
                        )
                    )

                # Precipitation: scale to 0-100 (assuming max ~10mm)
                if "precipitation_mm" in df_weather_filtered.columns:
                    # Only show if there's actual precipitation
                    if df_weather_filtered["precipitation_mm"].max() > 0:
                        precip_normalized = (df_weather_filtered["precipitation_mm"] / 10) * 100
                        precip_normalized = precip_normalized.clip(0, 100)
                        fig.add_trace(
                            go.Bar(
                                x=df_weather_filtered.index,
                                y=precip_normalized,
                                name="🌧️ Precipitation",
                                marker=dict(color="#3498DB", opacity=0.3),
                                yaxis="y2",
                                showlegend=True,
                                hovertemplate="Precip: %{customdata:.1f} mm<extra></extra>",
                                customdata=df_weather_filtered["precipitation_mm"],
                            )
                        )

    return fig


def render_results_section(
    config: Dict,
    historical_df: Optional[pd.DataFrame],
    training_df: Optional[pd.DataFrame],
    forecast_df: Optional[pd.DataFrame],
    metadata: Dict,
):
    """
    Render the forecast results section below the configuration panel.

    Args:
        config: Forecast configuration
        historical_df: Historical data
        training_df: Training data
        forecast_df: Forecast results
        metadata: Model metadata and metrics
    """
    st.markdown("---")
    st.header("📊 Forecast Results")

    if forecast_df is None or forecast_df.empty:
        st.error(
            "❌ Failed to generate forecast. Please check that data is available and API keys are configured."
        )
        return

    # Success message
    st.success("✅ Forecast generated successfully!")

    # === Configuration Summary ===
    with st.expander("📋 Configuration Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Market:** {MARKET_TYPES[config['market_type']]['display_name']}")
            if config["compare_models"]:
                st.markdown(f"**Models:** All 3 models")
            else:
                st.markdown(f"**Model:** {MODEL_TYPES[config['model_type']]['display_name']}")
        with col2:
            st.markdown(f"**Horizon:** {config['forecast_horizon_hours']} hours")
            st.markdown(f"**Resolution:** 15-minute intervals")
        with col3:
            features_used = []
            if config["training_data_options"]["use_time_features"]:
                features_used.append("Time")
            if config["training_data_options"]["use_weather_data"]:
                features_used.append("Weather")
            st.markdown(f"**Features:** {', '.join(features_used)}")

    # === Main Forecast Chart ===
    st.subheader("📈 Price Forecast")
    fig = plot_forecast_results(
        forecast_df, config["market_type"], config, historical_df, training_df
    )
    st.plotly_chart(fig, width="stretch")

    # === Forecast Statistics ===
    st.subheader("📊 Forecast Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mean_price = forecast_df["forecast_price_eur_per_mwh"].mean()
        st.metric("Mean Price", f"{mean_price:.2f}", help="EUR/MWh")

    with col2:
        min_price = forecast_df["forecast_price_eur_per_mwh"].min()
        st.metric("Min Price", f"{min_price:.2f}", help="EUR/MWh")

    with col3:
        max_price = forecast_df["forecast_price_eur_per_mwh"].max()
        st.metric("Max Price", f"{max_price:.2f}", help="EUR/MWh")

    with col4:
        if "rmse" in metadata and "rmse_points" in metadata:
            rmse = metadata["rmse"]
            n_points = metadata["rmse_points"]
            hours_covered = n_points / 4
            st.metric(
                "RMSE",
                f"{rmse:.2f}",
                help=f"Root Mean Squared Error (EUR/MWh) on {n_points} points ({hours_covered:.1f}h)",
            )
        else:
            st.metric("RMSE", "N/A", help="No historical data for comparison")

    # === Tabs for Additional Info ===
    tab1, tab2, tab3 = st.tabs(["📋 Data Table", "ℹ️ Model Info", "📊 Feature Importance"])

    with tab1:
        df_display = forecast_df.copy()
        df_display["timestamp_utc"] = df_display["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M")
        df_display["forecast_price_eur_per_mwh"] = df_display["forecast_price_eur_per_mwh"].round(2)

        st.dataframe(df_display, width="stretch", hide_index=True)

        # Download button
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"forecast_{config['market_type']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    with tab2:
        if config["compare_models"]:
            for mtype, model_info in metadata.get("models", {}).items():
                with st.expander(f"{MODEL_TYPES[mtype]['display_name']}", expanded=False):
                    st.write(f"**Trained at:** {model_info['trained_at']}")
                    st.write(f"**Training samples:** {model_info['n_samples']}")
                    st.write(f"**Features used:** {len(model_info['feature_columns'])}")

                    # Show model-specific information
                    if mtype == "random_forest" and "feature_importance" in model_info:
                        st.markdown("**Top 10 Most Important Features:**")
                        importance_data = model_info["feature_importance"]
                        import_df = pd.DataFrame(
                            {
                                "Feature": importance_data["features"][:10],
                                "Importance": importance_data["importance"][:10],
                            }
                        )
                        st.dataframe(import_df, hide_index=True)
                    elif mtype == "linear_regression" and "coefficients" in model_info:
                        st.markdown("**Model Coefficients:**")
                        st.write(f"Intercept: {model_info['intercept']:.4f}")
                        coef_df = pd.DataFrame(
                            {
                                "Feature": model_info["feature_columns"][:10],
                                "Coefficient": model_info["coefficients"][:10],
                            }
                        )
                        st.dataframe(coef_df, hide_index=True)
        else:
            model_info = metadata.get("model_info")
            if model_info:
                st.write(f"**Model:** {MODEL_TYPES[config['model_type']]['display_name']}")
                st.write(f"**Trained at:** {model_info['trained_at']}")
                st.write(f"**Training samples:** {model_info['n_samples']}")
                st.write(f"**Features used:** {len(model_info['feature_columns'])}")

                # Model-specific statistics
                if config["model_type"] == "random_forest" and "feature_importance" in model_info:
                    st.markdown("---")
                    st.markdown("### Feature Importance")
                    importance_data = model_info["feature_importance"]

                    # Create bar chart
                    top_n = min(15, len(importance_data["features"]))
                    fig_importance = go.Figure(
                        [
                            go.Bar(
                                x=importance_data["importance"][:top_n],
                                y=importance_data["features"][:top_n],
                                orientation="h",
                                marker=dict(color="#1f77b4"),
                            )
                        ]
                    )
                    fig_importance.update_layout(
                        title=f"Top {top_n} Most Important Features",
                        xaxis_title="Importance Score",
                        yaxis_title="Feature",
                        height=400,
                        yaxis={"categoryorder": "total ascending"},
                    )
                    st.plotly_chart(fig_importance, width="stretch")

                    # Show numerical values
                    with st.expander("View All Feature Importance Values"):
                        import_df = pd.DataFrame(
                            {
                                "Feature": importance_data["features"],
                                "Importance": [
                                    f"{imp:.6f}" for imp in importance_data["importance"]
                                ],
                            }
                        )
                        st.dataframe(import_df, hide_index=True, height=400)

                elif config["model_type"] == "linear_regression" and "coefficients" in model_info:
                    st.markdown("---")
                    st.markdown("### Linear Regression Equation")

                    # Display equation
                    intercept = model_info.get("intercept", 0.0)
                    coefficients = model_info.get("coefficients", [])
                    features = model_info.get("feature_columns", [])

                    st.write(f"**Intercept (β₀):** {intercept:.4f}")
                    st.markdown("**Equation:**")

                    # Build equation string
                    eq_parts = [f"{intercept:.4f}"]
                    for i, (feat, coef) in enumerate(zip(features[:10], coefficients[:10])):
                        sign = "+" if coef >= 0 else "-"
                        eq_parts.append(f"{sign} {abs(coef):.4f} × {feat}")

                    equation = "Price = " + " ".join(eq_parts)
                    if len(features) > 10:
                        equation += f" + ... ({len(features)-10} more features)"

                    st.code(equation, language="")

                    # Show top coefficients
                    st.markdown("**Top 10 Coefficients by Magnitude:**")
                    coef_abs = [
                        (feat, coef, abs(coef)) for feat, coef in zip(features, coefficients)
                    ]
                    coef_abs.sort(key=lambda x: x[2], reverse=True)

                    coef_df = pd.DataFrame(
                        {
                            "Feature": [x[0] for x in coef_abs[:10]],
                            "Coefficient": [f"{x[1]:.6f}" for x in coef_abs[:10]],
                            "Magnitude": [f"{x[2]:.6f}" for x in coef_abs[:10]],
                        }
                    )
                    st.dataframe(coef_df, hide_index=True)

                    # Full coefficients in expander
                    with st.expander("View All Coefficients"):
                        all_coef_df = pd.DataFrame(
                            {"Feature": features, "Coefficient": [f"{c:.6f}" for c in coefficients]}
                        )
                        st.dataframe(all_coef_df, hide_index=True, height=400)

                with st.expander("Feature Details"):
                    st.write(model_info["feature_columns"])

    with tab3:
        # This tab now shows additional feature importance details
        if config["model_type"] == "random_forest":
            model_info = (
                metadata.get("model_info")
                if not config["compare_models"]
                else metadata.get("models", {}).get("random_forest")
            )

            if model_info and "feature_importance" in model_info:
                st.markdown("### Feature Importance Analysis")

                importance_data = model_info["feature_importance"]
                features = importance_data["features"]
                importance = importance_data["importance"]

                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Features", len(features))
                with col2:
                    st.metric("Top Feature", features[0])
                with col3:
                    st.metric("Top Importance", f"{importance[0]:.4f}")

                # Cumulative importance
                cumsum = [sum(importance[: i + 1]) for i in range(len(importance))]

                st.markdown("**Cumulative Feature Importance**")
                st.write(f"Top 5 features explain: {cumsum[4]:.1%} of variance")
                st.write(f"Top 10 features explain: {cumsum[9]:.1%} of variance")
                st.write(
                    f"Top 20 features explain: {cumsum[min(19, len(cumsum)-1)]:.1%} of variance"
                )

                # Detailed table
                st.markdown("**Complete Feature Rankings**")
                detail_df = pd.DataFrame(
                    {
                        "Rank": range(1, len(features) + 1),
                        "Feature": features,
                        "Importance": [f"{imp:.6f}" for imp in importance],
                        "Cumulative": [f"{cum:.2%}" for cum in cumsum],
                    }
                )
                st.dataframe(detail_df, hide_index=True, height=500)
            else:
                st.info(
                    "Feature importance data not available. Generate a new forecast to see importance scores."
                )

        elif config["model_type"] == "linear_regression":
            model_info = (
                metadata.get("model_info")
                if not config["compare_models"]
                else metadata.get("models", {}).get("linear_regression")
            )

            if model_info and "coefficients" in model_info:
                st.markdown("### Coefficient Analysis")

                features = model_info["feature_columns"]
                coefficients = model_info["coefficients"]

                # Create visualization of coefficients
                coef_abs = [(feat, coef, abs(coef)) for feat, coef in zip(features, coefficients)]
                coef_abs.sort(key=lambda x: x[2], reverse=True)

                top_n = min(20, len(coef_abs))
                top_features = [x[0] for x in coef_abs[:top_n]]
                top_coefs = [x[1] for x in coef_abs[:top_n]]

                fig_coef = go.Figure(
                    [
                        go.Bar(
                            x=top_coefs,
                            y=top_features,
                            orientation="h",
                            marker=dict(color=top_coefs, colorscale="RdBu", cmid=0),
                        )
                    ]
                )
                fig_coef.update_layout(
                    title=f"Top {top_n} Coefficients by Magnitude",
                    xaxis_title="Coefficient Value",
                    yaxis_title="Feature",
                    height=500,
                    yaxis={"categoryorder": "total ascending"},
                )
                st.plotly_chart(fig_coef, width="stretch")

                # Statistics
                st.markdown("**Coefficient Statistics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive Coefficients", sum(1 for c in coefficients if c > 0))
                with col2:
                    st.metric("Negative Coefficients", sum(1 for c in coefficients if c < 0))
                with col3:
                    st.metric("Near-Zero (|x|<0.01)", sum(1 for c in coefficients if abs(c) < 0.01))
            else:
                st.info(
                    "Coefficient data not available. Generate a new forecast to see regression coefficients."
                )

        elif config["model_type"] == "persistence":
            st.info(
                "The Persistence model uses the most recent price as the forecast. No feature importance or coefficients are applicable."
            )

        else:
            st.info("Select a model type and generate a forecast to view feature analysis.")


def render_welcome_screen():
    """Render the welcome screen when no forecast has been run."""
    st.markdown("---")
    st.header("👋 Welcome to Energy Market Forecasting")

    st.markdown(
        """
        ### Get Started
        
        1. **Select your market**: Choose from Day-Ahead or Imbalance markets (shortage/surplus prices, regulation state)
        2. **Pick a model**: Try Persistence, Linear Regression, Random Forest, or XGBoost Classifier (for regulation state)
        3. **Configure settings**: Adjust hyperparameters and training data sources
        4. **Set your horizon**: Forecast 1-36 hours ahead
        5. **Click "Run Forecast"**: Generate and visualize predictions
        
        ### Data Sources
        
        This application uses live data from:
        - **ENTSO-E Transparency Platform**: Day-ahead prices and imbalance data
        - **KNMI Weather API**: Temperature, wind, radiation, and precipitation
        - **Automatic updates**: Data refreshes automatically to stay current
        """
    )

    # Data summary
    st.subheader("📊 Current Data Status")
    summary = get_data_summary()

    if any(info["records"] > 0 for info in summary["markets"].values()):
        for mkt_type, info in summary["markets"].items():
            if info["records"] > 0:
                st.write(
                    f"**{MARKET_TYPES[mkt_type]['display_name']}:** "
                    f"{info['records']:,} records "
                    f"({info['start_date'][:10]} to {info['end_date'][:10]})"
                )

        if summary["weather"]["records"] > 0:
            st.write(
                f"**Weather Data:** {summary['weather']['records']:,} records "
                f"({summary['weather']['start_date'][:10]} to {summary['weather']['end_date'][:10]})"
            )


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """Main application entry point."""

    # Get configuration from UI (returns None if button not clicked)
    config = get_forecast_configuration()

    # If forecast requested, generate and display results
    if config is not None:
        with st.spinner("🔄 Generating forecast..."):
            historical_df, training_df, forecast_df, metadata = generate_forecast_with_config(
                config
            )

        render_results_section(config, historical_df, training_df, forecast_df, metadata)

    else:
        # Show welcome screen
        render_welcome_screen()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.9em;'>
            <p>Energy Market Forecasting for the Netherlands | Data: ENTSO-E & KNMI</p>
            <p>
                <a href="https://energymarketforecastingproject.streamlit.app/" target="_blank">Live Demo</a> | 
                <a href="https://github.com/TyredMayfly/Energy_Market_Forecasting_project" target="_blank">GitHub</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
