"""
Streamlit web application for power market forecasting - Redesigned UI.

Features a clean, top-aligned control panel with comprehensive forecasting configuration.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.config import MARKET_TYPES, MODEL_TYPES, settings
from app.services.data_store import get_data_summary, load_market_data, load_weather_data
from app.services.forecast_service import get_forecast_service, TrainingDataConfig


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
            help="Number of recent intervals to use for persistence forecast"
        )
        
    elif model_type == "linear_regression":
        st.markdown("**Linear Regression Settings**")
        col1, col2 = st.columns(2)
        with col1:
            hyperparams["fit_intercept"] = st.checkbox(
                "Fit intercept",
                value=True,
                help="Whether to calculate the intercept for this model"
            )
        with col2:
            hyperparams["normalize"] = st.checkbox(
                "Normalize features",
                value=False,
                help="Normalize features before fitting (deprecated in newer sklearn)"
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
                help="Number of trees in the forest"
            )
            hyperparams["min_samples_leaf"] = st.slider(
                "Min samples per leaf",
                min_value=1,
                max_value=20,
                value=1,
                help="Minimum number of samples required at a leaf node"
            )
        with col2:
            max_depth_enabled = st.checkbox(
                "Limit tree depth",
                value=False,
                help="Enable maximum depth constraint"
            )
            if max_depth_enabled:
                hyperparams["max_depth"] = st.slider(
                    "Max depth",
                    min_value=5,
                    max_value=50,
                    value=20,
                    help="Maximum depth of each tree"
                )
            else:
                hyperparams["max_depth"] = None
            
            hyperparams["random_state"] = st.number_input(
                "Random seed",
                min_value=0,
                max_value=10000,
                value=42,
                help="Random state for reproducibility"
            )
    
    return hyperparams


def get_forecast_configuration() -> Optional[Dict]:
    """
    Render the top-aligned forecast configuration panel and collect user inputs.
    
    Returns:
        Dictionary containing all configuration parameters, or None if "Run Forecast" not clicked
    """
    st.title("⚡ Energy Market Forecasting - Netherlands")
    
    # Demo mode banner
    if settings.demo_mode:
        st.info(
            "🧪 **Demo Mode Active** - Using sample data (Jan 1 - Nov 18, 2025) with "
            "reduced lag features. No API keys required."
        )
    
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
                key="market_select"
            )
            st.caption(f"ℹ️ {MARKET_TYPES[market_type]['description']}")
        
        with col2:
            st.subheader("2️⃣ Forecasting Model")
            model_type = st.selectbox(
                "Select ML model",
                options=list(MODEL_TYPES.keys()),
                format_func=lambda x: MODEL_TYPES[x]["display_name"],
                help="Choose the forecasting algorithm",
                key="model_select"
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
                key="time_features"
            )
            
            # Weather data master toggle
            use_weather_data = st.checkbox(
                "Use weather data (KNMI)",
                value=True,
                help="Include meteorological features in model training",
                key="weather_master"
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
                key="rolling"
            )
        
        st.markdown("---")
        
        # === ROW 3: Forecast Settings & Display Options ===
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("5️⃣ Forecast Horizon")
            forecast_horizon_hours = st.slider(
                "Hours ahead",
                min_value=1,
                max_value=36,
                value=24,
                help="Number of hours to forecast into the future"
            )
            st.caption(f"📊 {forecast_horizon_hours * 4} intervals (15-min resolution)")
        
        with col2:
            st.subheader("6️⃣ Historical Context")
            historical_window_hours = st.slider(
                "Historical window (hours)",
                min_value=24,
                max_value=168,  # 7 days
                value=72,
                help="Hours of historical data to display on chart"
            )
        
        with col3:
            st.subheader("7️⃣ Chart Display Options")
            show_historical = st.checkbox(
                "📈 Show historical prices",
                value=True,
                help="Display recent historical data on chart"
            )
            show_training_data = st.checkbox(
                "🎯 Show training data",
                value=False,
                help="Highlight data used for model training"
            )
            show_weather_overlay = st.checkbox(
                "🌤️ Show weather overlay",
                value=False,
                help="Add weather variables to chart (secondary axis)"
            )
            show_feature_importance = st.checkbox(
                "📊 Show feature importance",
                value=False,
                help="Display feature importance (Random Forest only)",
                disabled=(model_type != "random_forest")
            )
        
        st.markdown("---")
        
        # === Advanced Options ===
        with st.expander("🔧 Advanced Options"):
            compare_models = st.checkbox(
                "Compare all models",
                value=False,
                help="Generate forecasts from all three models for comparison"
            )
        
        # === Action Buttons ===
        st.markdown("")  # Spacing
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            run_forecast = st.button(
                "🚀 Run Forecast",
                type="primary",
                width="stretch",
                help="Generate forecast with current configuration"
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


def generate_forecast_with_config(config: Dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
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
                model_info = service.get_model_info(
                    config["market_type"],
                    mtype,
                    training_config
                )
                if model_info:
                    metadata["models"][mtype] = model_info
        else:
            model_info = service.get_model_info(
                config["market_type"],
                config["model_type"],
                training_config
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
    price_col = MARKET_TYPES[market_type]["price_column"]
    
    # === 1. Training Data (bottom layer) ===
    if config["show_training_data"] and training_df is not None and not training_df.empty:
        if training_df["timestamp_utc"].dt.tz is None:
            training_df["timestamp_utc"] = pd.to_datetime(training_df["timestamp_utc"], utc=True)
        
        fig.add_trace(
            go.Scatter(
                x=training_df["timestamp_utc"],
                y=training_df[price_col],
                mode="lines",
                name="Training Data",
                line=dict(color="lightblue", width=2, dash="dot"),
                opacity=0.6,
            )
        )
    
    # === 2. Historical Data (middle layer) ===
    if config["show_historical"] and historical_df is not None and not historical_df.empty:
        if historical_df["timestamp_utc"].dt.tz is None:
            historical_df["timestamp_utc"] = pd.to_datetime(historical_df["timestamp_utc"], utc=True)
        
        # Calculate forecast start time
        forecast_start_time = forecast_df["timestamp_utc"].min() + timedelta(
            hours=config["historical_window_hours"]
        )
        
        # Get recent historical data
        cutoff_time = forecast_start_time - timedelta(hours=config["historical_window_hours"])
        df_recent = historical_df[
            (historical_df["timestamp_utc"] >= cutoff_time) &
            (historical_df["timestamp_utc"] < forecast_start_time)
        ].copy()
        
        # Exclude training data to avoid overlap
        if config["show_training_data"] and training_df is not None:
            training_end = training_df["timestamp_utc"].max()
            df_recent = df_recent[df_recent["timestamp_utc"] > training_end]
        
        if not df_recent.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_recent["timestamp_utc"],
                    y=df_recent[price_col],
                    mode="lines",
                    name="Historical",
                    line=dict(color="gray", width=2),
                    opacity=0.7,
                )
            )
    
    # === 3. Forecast Data (top layer) ===
    if config["compare_models"]:
        colors = {
            "persistence": "blue",
            "linear_regression": "green",
            "random_forest": "red"
        }
        
        for mtype in forecast_df["model_type"].unique():
            df_model = forecast_df[forecast_df["model_type"] == mtype]
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
                x=forecast_df["timestamp_utc"],
                y=forecast_df["forecast_price_eur_per_mwh"],
                mode="lines+markers",
                name=f"{MODEL_TYPES[config['model_type']]['display_name']} Forecast",
                line=dict(color="blue", width=3),
                marker=dict(size=8),
            )
        )
    
    # === Layout ===
    fig.update_layout(
        title=f"{MARKET_TYPES[market_type]['display_name']} Price Forecast",
        xaxis_title="Time (UTC)",
        yaxis_title="Price (EUR/MWh)",
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
    
    # === 4. Weather Overlay (if requested) ===
    if config["show_weather_overlay"]:
        df_weather = load_weather_data()
        
        if df_weather is not None and not df_weather.empty:
            if df_weather["timestamp_utc"].dt.tz is None:
                df_weather["timestamp_utc"] = pd.to_datetime(df_weather["timestamp_utc"], utc=True)
            
            # Filter to chart time range
            if config["show_historical"]:
                cutoff_time = forecast_df["timestamp_utc"].min() - timedelta(
                    hours=config["historical_window_hours"]
                )
            else:
                cutoff_time = forecast_df["timestamp_utc"].min()
            
            forecast_end = forecast_df["timestamp_utc"].max()
            df_weather_filtered = df_weather[
                (df_weather["timestamp_utc"] >= cutoff_time) &
                (df_weather["timestamp_utc"] <= forecast_end)
            ].copy()
            
            if not df_weather_filtered.empty:
                # Add secondary y-axis
                fig.update_layout(
                    yaxis2=dict(
                        title="Weather Variables",
                        overlaying="y",
                        side="right"
                    ),
                    height=550
                )
                
                # Add weather traces
                if "temperature_deg_c" in df_weather_filtered.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_weather_filtered["timestamp_utc"],
                            y=df_weather_filtered["temperature_deg_c"],
                            mode="lines",
                            name="Temperature (°C)",
                            line=dict(color="orange", width=1, dash="dash"),
                            opacity=0.6,
                            yaxis="y2",
                        )
                    )
                
                if "wind_speed_m_per_s" in df_weather_filtered.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_weather_filtered["timestamp_utc"],
                            y=df_weather_filtered["wind_speed_m_per_s"],
                            mode="lines",
                            name="Wind Speed (m/s)",
                            line=dict(color="cyan", width=1, dash="dash"),
                            opacity=0.6,
                            yaxis="y2",
                        )
                    )
    
    return fig


def render_results_section(
    config: Dict,
    historical_df: Optional[pd.DataFrame],
    training_df: Optional[pd.DataFrame],
    forecast_df: Optional[pd.DataFrame],
    metadata: Dict
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
            "❌ Failed to generate forecast. Please ensure demo data is available. "
            "In demo mode, sample data should be loaded automatically from the data/ directory."
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
        forecast_df,
        config["market_type"],
        config,
        historical_df,
        training_df
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
                help=f"Root Mean Squared Error (EUR/MWh) on {n_points} points ({hours_covered:.1f}h)"
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
        else:
            model_info = metadata.get("model_info")
            if model_info:
                st.write(f"**Model:** {MODEL_TYPES[config['model_type']]['display_name']}")
                st.write(f"**Trained at:** {model_info['trained_at']}")
                st.write(f"**Training samples:** {model_info['n_samples']}")
                st.write(f"**Features used:** {len(model_info['feature_columns'])}")
                
                with st.expander("Feature Details"):
                    st.write(model_info["feature_columns"])
    
    with tab3:
        if config["show_feature_importance"] and config["model_type"] == "random_forest":
            st.info("Feature importance visualization - Coming soon!")
        elif config["model_type"] != "random_forest":
            st.warning("Feature importance is only available for Random Forest models.")
        else:
            st.info("Enable 'Show feature importance' in the configuration panel to view this.")


def render_welcome_screen():
    """Render the welcome screen when no forecast has been run."""
    st.markdown("---")
    st.header("👋 Welcome to Energy Market Forecasting")
    
    st.markdown(
        """
        ### Get Started
        
        1. **Select your market**: Choose from Day-Ahead, Intraday, or Imbalance markets
        2. **Pick a model**: Try Persistence (baseline), Linear Regression, or Random Forest
        3. **Configure settings**: Adjust hyperparameters and training data sources
        4. **Set your horizon**: Forecast 1-36 hours ahead
        5. **Click "Run Forecast"**: Generate and visualize predictions
        
        ### About Demo Mode
        
        This application is running in **demo mode** with pre-loaded sample data covering 
        January 1 - November 18, 2025. No API keys are required.
        
        - **30,912+ market records** at 15-minute resolution
        - **31,056+ weather observations** from KNMI
        - **3 forecasting models** to compare
        """
    )
    
    # Data summary
    st.subheader("📊 Available Demo Data")
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
            historical_df, training_df, forecast_df, metadata = generate_forecast_with_config(config)
        
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
