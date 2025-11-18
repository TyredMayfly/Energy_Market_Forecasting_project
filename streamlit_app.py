"""
Streamlit web application for power market forecasting.

Interactive UI for selecting markets, models, and horizons, and visualizing forecasts.
"""

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app.core.config import MARKET_TYPES, MODEL_TYPES, settings
from app.services.data_store import get_data_summary, load_market_data
from app.services.forecast_service import get_forecast_service, TrainingDataConfig

# Page configuration
st.set_page_config(
    page_title="Market Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Title and description
st.title("⚡ Power Market Forecasting for the Netherlands")

# Show demo mode banner
if settings.demo_mode:
    st.info(
        "🧪 **Demo Mode Active** - Using sample data with reduced lag features. "
        "No API keys required. Set `DEMO_MODE=false` in `.env` to use real-time data."
    )

st.markdown(
    """
An educational web application for learning about electricity market forecasting.
Compare different forecasting models and explore day-ahead, intraday, and imbalance markets.
"""
)

# Sidebar for configuration
st.sidebar.header("Forecast Configuration")

# Market selection
market_type = st.sidebar.selectbox(
    "Market Type",
    options=list(MARKET_TYPES.keys()),
    format_func=lambda x: MARKET_TYPES[x]["display_name"],
    help="Select the electricity market to forecast",
)

# Display market description
st.sidebar.markdown(f"*{MARKET_TYPES[market_type]['description']}*")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    options=list(MODEL_TYPES.keys()),
    format_func=lambda x: MODEL_TYPES[x]["display_name"],
    help="Choose the forecasting algorithm to use",
)

# Display model description
st.sidebar.markdown(f"**{MODEL_TYPES[model_type]['display_name']}:** {MODEL_TYPES[model_type]['description']}")

# Forecast horizon
horizon_hours = st.sidebar.slider(
    "Forecast Horizon (hours)",
    min_value=1,
    max_value=36,
    value=24,
    help="Number of hours to forecast ahead",
)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    st.subheader("Model Comparison")
    compare_models = st.checkbox(
        "Compare Multiple Models",
        value=False,
        help="Show forecasts from all available models",
    )
    
    st.subheader("Model Training")
    use_weather_data = st.checkbox(
        "Use Weather Data",
        value=True,
        help="Include weather features in model training",
    )
    
    # Granular weather feature selection (only shown if weather data is enabled)
    if use_weather_data:
        st.markdown("**Select Weather Features:**")
        use_temperature = st.checkbox(
            "Temperature",
            value=True,
            help="Include temperature data",
        )
        use_wind_speed = st.checkbox(
            "Wind Speed",
            value=True,
            help="Include wind speed data",
        )
        use_cloud_cover = st.checkbox(
            "Cloud Cover",
            value=True,
            help="Include cloud cover data",
        )
        use_precipitation = st.checkbox(
            "Precipitation",
            value=True,
            help="Include precipitation data",
        )
    else:
        # Default values when weather is disabled
        use_temperature = False
        use_wind_speed = False
        use_cloud_cover = False
        use_precipitation = False
    
    st.subheader("Chart Display")
    show_historical = st.checkbox(
        "Show Historical Prices",
        value=True,
        help="Display recent historical prices on the forecast chart",
    )

    historical_hours = st.slider(
        "Historical Window (hours)",
        min_value=24,
        max_value=168,
        value=72,
        help="Number of hours of historical data to include in forecast (for chart display and RMSE calculation)",
    )
    
    show_training_data = st.checkbox(
        "Show Training Data",
        value=False,
        help="Highlight the historical data that was used to train the model",
    )
    
    show_weather_data = st.checkbox(
        "Show Weather Data",
        value=False,
        help="Display weather variables (temperature, wind speed, cloud cover, precipitation) on the chart",
    )

# Forecast button
run_forecast = st.sidebar.button("🚀 Run Forecast", type="primary", width="stretch")

# Data summary in sidebar
with st.sidebar.expander("📊 Data Summary"):
    summary = get_data_summary()

    for mkt_type, info in summary["markets"].items():
        if info["records"] > 0:
            st.metric(
                label=MARKET_TYPES[mkt_type]["display_name"],
                value=f"{info['records']} records",
                delta=None,
            )

    if summary["weather"]["records"] > 0:
        st.metric(
            label="Weather Data",
            value=f"{summary['weather']['records']} records",
            delta=None,
        )


# Main content area
if run_forecast:
    # Display configuration summary
    st.subheader("📋 Forecast Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Market:** {MARKET_TYPES[market_type]['display_name']}")
        st.caption(MARKET_TYPES[market_type]['description'])
    
    with col2:
        if compare_models:
            st.markdown(f"**Models:** Comparing {len(MODEL_TYPES)} models")
            st.caption("Persistence, Linear Regression, and Random Forest")
        else:
            st.markdown(f"**Model:** {MODEL_TYPES[model_type]['display_name']}")
            st.caption(MODEL_TYPES[model_type]['description'])
    
    with col3:
        weather_features_list = []
        if use_weather_data:
            if use_temperature:
                weather_features_list.append("Temperature")
            if use_wind_speed:
                weather_features_list.append("Wind")
            if use_cloud_cover:
                weather_features_list.append("Cloud")
            if use_precipitation:
                weather_features_list.append("Precipitation")
        
        if weather_features_list:
            st.markdown(f"**Weather Features:** {', '.join(weather_features_list)}")
            st.caption(f"Using {len(weather_features_list)} weather inputs")
        else:
            st.markdown("**Weather Features:** None")
            st.caption("Using only time and price lag features")
    
    st.divider()
    
    with st.spinner("Generating forecast..."):
        # Get forecast service
        service = get_forecast_service()
        
        # Create training configuration
        training_config = TrainingDataConfig(
            use_weather_data=use_weather_data,
            use_time_features=True,  # Always use time features
            use_temperature=use_temperature,
            use_wind_speed=use_wind_speed,
            use_cloud_cover=use_cloud_cover,
            use_precipitation=use_precipitation,
        )

        # Generate forecast(s)
        if compare_models:
            # Compare all models
            model_types_to_compare = list(MODEL_TYPES.keys())
            df_forecast = service.compare_models(
                market_type=market_type,
                model_types=model_types_to_compare,
                horizon_hours=horizon_hours,
                historical_window_hours=historical_hours,
                training_config=training_config,
            )
        else:
            # Single model forecast
            df_forecast = service.generate_forecast(
                market_type=market_type,
                model_type=model_type,
                horizon_hours=horizon_hours,
                historical_window_hours=historical_hours,
                training_config=training_config,
            )

        if df_forecast is not None and not df_forecast.empty:
            st.success("✅ Forecast generated successfully!")

            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📈 Forecast Chart", "📋 Data Table", "ℹ️ Model Info"])

            with tab1:
                # Create plot
                fig = go.Figure()
                
                price_col = MARKET_TYPES[market_type]["price_column"]

                # Get training data if requested
                df_training = None
                if show_training_data:
                    df_training = service.get_training_data(
                        market_type=market_type,
                        model_type=model_type if not compare_models else "persistence",
                        training_config=training_config,
                    )
                
                # Add training data first (so it appears behind other lines)
                if show_training_data and df_training is not None and not df_training.empty:
                    # Ensure timezone-aware
                    if df_training["timestamp_utc"].dt.tz is None:
                        df_training["timestamp_utc"] = pd.to_datetime(
                            df_training["timestamp_utc"], utc=True
                        )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df_training["timestamp_utc"],
                            y=df_training[price_col],
                            mode="lines",
                            name="Training Data",
                            line=dict(color="lightblue", width=2, dash="dot"),
                            opacity=0.6,
                        )
                    )

                # Add historical data if toggle is enabled (but not overlapping with training)
                if show_historical:
                    df_historical = load_market_data(market_type)
                    if df_historical is not None and not df_historical.empty:
                        # Ensure timezone-aware for comparison
                        if df_historical["timestamp_utc"].dt.tz is None:
                            df_historical["timestamp_utc"] = pd.to_datetime(
                                df_historical["timestamp_utc"], utc=True
                            )
                        
                        # Calculate actual forecast start time (df_forecast.min is already historical_hours back)
                        # So the forecast start is historical_hours forward from df_forecast.min
                        forecast_start_time = df_forecast["timestamp_utc"].min() + timedelta(
                            hours=historical_hours
                        )
                        
                        # Get recent historical data (before forecast start)
                        cutoff_time = forecast_start_time - timedelta(
                            hours=historical_hours
                        )
                        df_recent = df_historical[
                            (df_historical["timestamp_utc"] >= cutoff_time) &
                            (df_historical["timestamp_utc"] < forecast_start_time)
                        ].copy()
                        
                        # If showing training data, exclude it from historical to avoid overlap
                        if show_training_data and df_training is not None:
                            training_end = df_training["timestamp_utc"].max()
                            df_recent = df_recent[df_recent["timestamp_utc"] > training_end]

                        if not df_recent.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_recent["timestamp_utc"],
                                    y=df_recent[price_col],
                                    mode="lines",
                                    name="Historical (non-training)",
                                    line=dict(color="gray", width=2),
                                    opacity=0.7,
                                )
                            )

                # Add forecast(s)
                if compare_models:
                    # Multiple models
                    colors = {"persistence": "blue", "linear_regression": "green", "random_forest": "red"}

                    for mtype in df_forecast["model_type"].unique():
                        df_model = df_forecast[df_forecast["model_type"] == mtype]

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
                    # Single model
                    fig.add_trace(
                        go.Scatter(
                            x=df_forecast["timestamp_utc"],
                            y=df_forecast["forecast_price_eur_per_mwh"],
                            mode="lines+markers",
                            name=f"{MODEL_TYPES[model_type]['display_name']} Forecast",
                            line=dict(color="blue", width=3),
                            marker=dict(size=8),
                        )
                    )

                # Update layout
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
                
                # Add weather data if requested
                if show_weather_data:
                    from app.services.data_store import load_weather_data
                    df_weather = load_weather_data()
                    
                    if df_weather is not None and not df_weather.empty:
                        # Ensure timezone-aware
                        if df_weather["timestamp_utc"].dt.tz is None:
                            df_weather["timestamp_utc"] = pd.to_datetime(
                                df_weather["timestamp_utc"], utc=True
                            )
                        
                        # Filter to same time range as chart
                        if show_historical:
                            cutoff_time = df_forecast["timestamp_utc"].min() - timedelta(
                                hours=historical_hours
                            )
                        else:
                            cutoff_time = df_forecast["timestamp_utc"].min()
                        
                        forecast_end = df_forecast["timestamp_utc"].max()
                        df_weather_filtered = df_weather[
                            (df_weather["timestamp_utc"] >= cutoff_time) &
                            (df_weather["timestamp_utc"] <= forecast_end)
                        ].copy()
                        
                        if not df_weather_filtered.empty:
                            # Create secondary y-axis for weather data
                            fig.update_layout(
                                yaxis2=dict(
                                    title="Weather Variables",
                                    overlaying="y",
                                    side="right"
                                ),
                                height=550  # Slightly taller to accommodate legend
                            )
                            
                            # Add temperature
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
                            
                            # Add wind speed
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
                            
                            # Add cloud cover
                            if "cloud_cover_oktas" in df_weather_filtered.columns:
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_weather_filtered["timestamp_utc"],
                                        y=df_weather_filtered["cloud_cover_oktas"],
                                        mode="lines",
                                        name="Cloud Cover (oktas)",
                                        line=dict(color="purple", width=1, dash="dash"),
                                        opacity=0.6,
                                        yaxis="y2",
                                    )
                                )
                            
                            # Add precipitation
                            if "precipitation_mm" in df_weather_filtered.columns:
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_weather_filtered["timestamp_utc"],
                                        y=df_weather_filtered["precipitation_mm"],
                                        mode="lines",
                                        name="Precipitation (mm)",
                                        line=dict(color="blue", width=1, dash="dot"),
                                        opacity=0.6,
                                        yaxis="y2",
                                        fill="tozeroy",
                                        fillcolor="rgba(0, 0, 255, 0.1)",
                                )
                            )

                st.plotly_chart(fig, width="stretch")

                # Statistics
                st.subheader("Forecast Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    mean_price = df_forecast["forecast_price_eur_per_mwh"].mean()
                    st.metric("Mean Price", f"{mean_price:.2f}", help="EUR/MWh")

                with col2:
                    min_price = df_forecast["forecast_price_eur_per_mwh"].min()
                    st.metric("Min Price", f"{min_price:.2f}", help="EUR/MWh")

                with col3:
                    max_price = df_forecast["forecast_price_eur_per_mwh"].max()
                    st.metric("Max Price", f"{max_price:.2f}", help="EUR/MWh")
                    
                with col4:
                    # Calculate RMSE if historical data is available
                    rmse_result = service.calculate_forecast_rmse(df_forecast, market_type)
                    if rmse_result is not None:
                        rmse, n_points = rmse_result
                        # Calculate hours covered by actual overlapping data
                        hours_covered = n_points / 4
                        st.metric("RMSE", f"{rmse:.2f}", 
                                 help=f"Root Mean Squared Error (EUR/MWh) calculated on {n_points} data points ({hours_covered:.1f} hours at 15-min intervals). Primarily based on {historical_hours}h historical window where actual data exists for comparison.")
                    else:
                        st.metric("RMSE", "N/A", 
                                 help="No historical data available for comparison")

            with tab2:
                # Display forecast data table
                st.subheader("Forecast Data")

                # Format the dataframe for display
                df_display = df_forecast.copy()
                df_display["timestamp_utc"] = df_display["timestamp_utc"].dt.strftime(
                    "%Y-%m-%d %H:%M"
                )
                df_display["forecast_price_eur_per_mwh"] = df_display[
                    "forecast_price_eur_per_mwh"
                ].round(2)

                st.dataframe(
                    df_display,
                    width="stretch",
                    hide_index=True,
                )

                # Download button
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"forecast_{market_type}_{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            with tab3:
                # Model information
                st.subheader("Model Information")

                if compare_models:
                    for mtype in model_types_to_compare:
                        model_info = service.get_model_info(market_type, mtype, training_config)
                        if model_info:
                            with st.expander(f"{MODEL_TYPES[mtype]['display_name']}"):
                                st.write(f"**Description:** {MODEL_TYPES[mtype]['description']}")
                                st.write(f"**Trained at:** {model_info['trained_at']}")
                                st.write(f"**Training samples:** {model_info['n_samples']}")
                                st.write(f"**Features used:** {len(model_info['feature_columns'])}")
                else:
                    model_info = service.get_model_info(market_type, model_type, training_config)
                    if model_info:
                        st.write(f"**Model:** {MODEL_TYPES[model_type]['display_name']}")
                        st.write(f"**Description:** {MODEL_TYPES[model_type]['description']}")
                        st.write(f"**Trained at:** {model_info['trained_at']}")
                        st.write(f"**Training samples:** {model_info['n_samples']}")
                        st.write(f"**Features used:** {len(model_info['feature_columns'])}")

                        with st.expander("Feature Details"):
                            st.write(model_info["feature_columns"])

        else:
            st.error(
                "❌ Failed to generate forecast. Please ensure data is initialized. "
                "Run: `python -m app.services.data_update_service --init`"
            )

else:
    # Instructions
    st.info(
        """
    👈 **Get Started:**
    1. Select a market type (Day-Ahead, Intraday, or Imbalance)
    2. Choose a forecasting model
    3. Set your forecast horizon (1-36 hours)
    4. Click **Run Forecast** to generate predictions
    
    **Note:** Make sure to initialize data first by running:
    ```
    python -m app.services.data_update_service --init
    ```
    """
    )

    # Show example data summary
    st.subheader("📊 Available Data")

    summary = get_data_summary()

    if any(info["records"] > 0 for info in summary["markets"].values()):
        for mkt_type, info in summary["markets"].items():
            if info["records"] > 0:
                st.write(
                    f"**{MARKET_TYPES[mkt_type]['display_name']}:** "
                    f"{info['records']} records "
                    f"({info['start_date'][:10]} to {info['end_date'][:10]})"
                )
    else:
        st.warning(
            "⚠️ No market data found. Please initialize the data first:\n"
            "```\npython -m app.services.data_update_service --init\n```"
        )


# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Educational Power Market Forecasting | Data from ENTSO-E & KNMI</p>
    <p>Built with Streamlit, FastAPI, and scikit-learn</p>
</div>
""",
    unsafe_allow_html=True,
)
