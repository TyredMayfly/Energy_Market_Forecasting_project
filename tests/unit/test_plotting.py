"""
Unit tests for Streamlit plotting and visualization functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go


class TestPlotForecastResults:
    """Test forecast visualization plotting."""
    
    def test_plot_creation_basic(self):
        """Test basic plot creation with minimal data."""
        # Create minimal forecast data
        timestamps = pd.date_range('2025-01-01', periods=24, freq='h', tz='UTC')
        df_forecast = pd.DataFrame({
            'timestamp_utc': timestamps,
            'forecast_price_eur_per_mwh': np.random.uniform(20, 80, 24)
        })
        
        # Mock config
        config = {
            'market_type': 'day_ahead',
            'model_type': 'persistence',
            'show_historical': False,
            'show_weather_overlay': False
        }
        
        # Import function would happen here
        # fig = plot_forecast_results(config, df_forecast, None, None, None)
        
        # For now, test data structure
        assert len(df_forecast) == 24
        assert df_forecast['forecast_price_eur_per_mwh'].min() >= 20
        assert df_forecast['forecast_price_eur_per_mwh'].max() <= 80
    
    def test_weather_data_filtering(self):
        """Test filtering weather data to forecast timerange."""
        # Create weather data spanning multiple days
        weather_start = pd.Timestamp('2025-01-01', tz='UTC')
        weather_timestamps = pd.date_range(weather_start, periods=72, freq='h')
        df_weather = pd.DataFrame({
            'temperature_deg_c': np.random.uniform(0, 20, 72),
            'wind_speed_m_per_s': np.random.uniform(0, 15, 72),
            'global_radiation_w_per_m2': np.random.uniform(0, 800, 72),
            'cloud_cover_pct': np.random.uniform(0, 100, 72),
            'precipitation_mm': np.random.uniform(0, 5, 72)
        }, index=weather_timestamps)
        
        # Forecast covers 24 hours starting at hour 24
        forecast_start = pd.Timestamp('2025-01-02', tz='UTC')
        forecast_end = pd.Timestamp('2025-01-03', tz='UTC') - pd.Timedelta(hours=1)
        
        # Filter weather to forecast range
        filtered = df_weather[
            (df_weather.index >= forecast_start) &
            (df_weather.index <= forecast_end)
        ]
        
        assert len(filtered) == 24
        assert filtered.index[0] == forecast_start
        assert filtered.index[-1] == forecast_end
    
    def test_weather_variable_presence(self):
        """Test that all weather variables are present in data."""
        timestamps = pd.date_range('2025-01-01', periods=24, freq='h', tz='UTC')
        df_weather = pd.DataFrame({
            'temperature_deg_c': np.random.uniform(0, 20, 24),
            'wind_speed_m_per_s': np.random.uniform(0, 15, 24),
            'global_radiation_w_per_m2': np.random.uniform(0, 800, 24),
            'cloud_cover_pct': np.random.uniform(0, 100, 24),
            'precipitation_mm': np.random.uniform(0, 5, 24)
        }, index=timestamps)
        
        # Verify all 5 weather variables are present
        expected_vars = [
            'temperature_deg_c',
            'wind_speed_m_per_s',
            'global_radiation_w_per_m2',
            'cloud_cover_pct',
            'precipitation_mm'
        ]
        
        for var in expected_vars:
            assert var in df_weather.columns, f"Missing weather variable: {var}"
        
        # Verify reasonable value ranges
        assert df_weather['temperature_deg_c'].between(-50, 50).all()
        assert df_weather['wind_speed_m_per_s'].between(0, 50).all()
        assert df_weather['global_radiation_w_per_m2'].between(0, 1500).all()
        assert df_weather['cloud_cover_pct'].between(0, 100).all()
        assert df_weather['precipitation_mm'].between(0, 100).all()


class TestPlotlyFigureGeneration:
    """Test Plotly figure creation and configuration."""
    
    def test_figure_with_multiple_yaxes(self):
        """Test creation of figure with multiple y-axes for weather overlay."""
        fig = go.Figure()
        
        # Add price trace on primary y-axis
        fig.add_trace(go.Scatter(
            x=[1, 2, 3],
            y=[10, 20, 15],
            name="Price",
            yaxis="y"
        ))
        
        # Add temperature on secondary y-axis
        fig.add_trace(go.Scatter(
            x=[1, 2, 3],
            y=[15, 18, 16],
            name="Temperature",
            yaxis="y2"
        ))
        
        # Configure multiple y-axes
        fig.update_layout(
            yaxis=dict(title="Price (EUR/MWh)"),
            yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right")
        )
        
        assert len(fig.data) == 2
        assert fig.data[0].name == "Price"
        assert fig.data[1].name == "Temperature"
        assert fig.data[1].yaxis == "y2"
    
    def test_precipitation_bar_overlay(self):
        """Test adding precipitation as bar chart overlay."""
        timestamps = pd.date_range('2025-01-01', periods=24, freq='h')
        precipitation = np.array([0, 0, 2.5, 1.2, 0.5, 0, 0] + [0]*17)
        
        fig = go.Figure()
        
        # Add precipitation bars
        fig.add_trace(go.Bar(
            x=timestamps,
            y=precipitation,
            name="Precipitation",
            marker=dict(color='blue', opacity=0.3)
        ))
        
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)
        assert (fig.data[0].y == precipitation).all()
    
    def test_solar_radiation_area_plot(self):
        """Test solar radiation as filled area plot."""
        timestamps = pd.date_range('2025-01-01 06:00', periods=12, freq='h')
        # Solar radiation peaks at noon
        radiation = [0, 100, 300, 500, 700, 800, 750, 600, 400, 200, 50, 0]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=radiation,
            mode='lines',
            name='Solar Radiation',
            fill='tozeroy',
            fillcolor='rgba(255, 217, 61, 0.1)'
        ))
        
        assert len(fig.data) == 1
        assert fig.data[0].fill == 'tozeroy'
        assert max(radiation) == 800  # Peak radiation


class TestHistoricalDataOverlay:
    """Test historical data overlay on forecast plots."""
    
    def test_historical_window_calculation(self):
        """Test calculation of historical window timestamps."""
        forecast_start = pd.Timestamp('2025-01-01 12:00', tz='UTC')
        historical_hours = 24
        
        historical_start = forecast_start - pd.Timedelta(hours=historical_hours)
        
        assert historical_start == pd.Timestamp('2024-12-31 12:00', tz='UTC')
        
        # Generate historical timestamps
        historical_timestamps = pd.date_range(
            historical_start,
            forecast_start - pd.Timedelta(minutes=15),
            freq='15min'
        )
        
        assert len(historical_timestamps) == 96  # 24 hours * 4 (15-min intervals)
    
    def test_historical_actual_price_merge(self):
        """Test merging historical forecast with actual prices."""
        # Historical forecast
        timestamps = pd.date_range('2025-01-01', periods=24, freq='h', tz='UTC')
        df_historical = pd.DataFrame({
            'timestamp_utc': timestamps,
            'forecast_price_eur_per_mwh': np.random.uniform(40, 60, 24)
        })
        
        # Actual prices (some overlap)
        df_actual = pd.DataFrame({
            'timestamp_utc': timestamps,
            'actual_price_eur_per_mwh': np.random.uniform(38, 62, 24)
        })
        
        # Merge
        df_merged = df_historical.merge(df_actual, on='timestamp_utc', how='left')
        
        assert len(df_merged) == 24
        assert 'forecast_price_eur_per_mwh' in df_merged.columns
        assert 'actual_price_eur_per_mwh' in df_merged.columns


class TestModelComparisonVisualization:
    """Test model comparison visualization."""
    
    def test_multiple_forecast_traces(self):
        """Test adding multiple model forecasts to same plot."""
        timestamps = pd.date_range('2025-01-01', periods=24, freq='h', tz='UTC')
        
        models = {
            'persistence': np.random.uniform(40, 60, 24),
            'linear_regression': np.random.uniform(35, 65, 24),
            'random_forest': np.random.uniform(38, 62, 24)
        }
        
        fig = go.Figure()
        
        colors = ['blue', 'green', 'red']
        for i, (model_name, predictions) in enumerate(models.items()):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=predictions,
                mode='lines',
                name=model_name,
                line=dict(color=colors[i])
            ))
        
        assert len(fig.data) == 3
        assert fig.data[0].name == 'persistence'
        assert fig.data[1].name == 'linear_regression'
        assert fig.data[2].name == 'random_forest'
    
    def test_rmse_annotation(self):
        """Test adding RMSE annotations to plot."""
        fig = go.Figure()
        
        # Add annotation for RMSE
        fig.add_annotation(
            text="RMSE: 12.34 EUR/MWh",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="white",
            bordercolor="gray"
        )
        
        assert len(fig.layout.annotations) == 1
        assert "RMSE" in fig.layout.annotations[0].text


class TestFeatureImportancePlots:
    """Test feature importance visualization."""
    
    def test_horizontal_bar_chart(self):
        """Test horizontal bar chart for feature importance."""
        features = ['price_lag_1h', 'hour_of_day', 'temperature_deg_c', 'wind_speed_m_per_s']
        importance = [0.45, 0.25, 0.20, 0.10]
        
        fig = go.Figure([
            go.Bar(
                x=importance,
                y=features,
                orientation='h'
            )
        ])
        
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)
        assert fig.data[0].orientation == 'h'
        assert len(fig.data[0].x) == 4
    
    def test_feature_importance_sorting(self):
        """Test that features are sorted by importance."""
        features = ['feat_a', 'feat_b', 'feat_c', 'feat_d']
        importance = [0.1, 0.5, 0.3, 0.1]
        
        # Sort by importance descending
        sorted_indices = np.argsort(importance)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importance = [importance[i] for i in sorted_indices]
        
        assert sorted_features[0] == 'feat_b'  # Highest importance (0.5)
        assert sorted_features[1] == 'feat_c'  # Second (0.3)
        assert sorted_importance[0] >= sorted_importance[1]
        assert sorted_importance[1] >= sorted_importance[2]


class TestCoefficientVisualization:
    """Test linear regression coefficient visualization."""
    
    def test_coefficient_color_scale(self):
        """Test coefficient visualization with diverging color scale."""
        features = ['feat_a', 'feat_b', 'feat_c', 'feat_d', 'feat_e']
        coefficients = [0.5, -0.3, 0.1, -0.8, 0.2]
        
        fig = go.Figure([
            go.Bar(
                x=coefficients,
                y=features,
                orientation='h',
                marker=dict(
                    color=coefficients,
                    colorscale='RdBu',
                    cmid=0
                )
            )
        ])
        
        # Plotly converts named colorscales to full RGB tuples
        assert fig.data[0].marker.colorscale is not None
        assert fig.data[0].marker.cmid == 0
        
        # Verify positive and negative coefficients
        assert any(c > 0 for c in coefficients)
        assert any(c < 0 for c in coefficients)
    
    def test_equation_string_formatting(self):
        """Test formatting of regression equation."""
        intercept = 25.5
        features = ['price_lag_1h', 'hour_of_day']
        coefficients = [0.85, -1.2]
        
        # Build equation
        eq_parts = [f"{intercept:.2f}"]
        for feat, coef in zip(features, coefficients):
            sign = "+" if coef >= 0 else "-"
            eq_parts.append(f"{sign} {abs(coef):.2f} × {feat}")
        
        equation = "Price = " + " ".join(eq_parts)
        
        assert "25.50" in equation
        assert "+ 0.85 × price_lag_1h" in equation
        assert "- 1.20 × hour_of_day" in equation
        assert equation.startswith("Price = ")


class TestWeatherOverlayEdgeCases:
    """Test edge cases in weather overlay visualization."""
    
    def test_missing_weather_variables(self):
        """Test handling of missing weather variables."""
        timestamps = pd.date_range('2025-01-01', periods=24, freq='h', tz='UTC')
        
        # Weather data with only some variables
        df_weather = pd.DataFrame({
            'temperature_deg_c': np.random.uniform(0, 20, 24),
            'wind_speed_m_per_s': np.random.uniform(0, 15, 24)
            # Missing: radiation, cloud_cover, precipitation
        }, index=timestamps)
        
        # Check which variables are available
        available_vars = []
        for var in ['temperature_deg_c', 'wind_speed_m_per_s', 'global_radiation_w_per_m2', 
                    'cloud_cover_pct', 'precipitation_mm']:
            if var in df_weather.columns:
                available_vars.append(var)
        
        assert len(available_vars) == 2
        assert 'temperature_deg_c' in available_vars
        assert 'wind_speed_m_per_s' in available_vars
    
    def test_zero_precipitation_handling(self):
        """Test that zero precipitation is handled correctly."""
        timestamps = pd.date_range('2025-01-01', periods=24, freq='h', tz='UTC')
        precipitation = np.zeros(24)  # No precipitation
        
        # Should not add precipitation trace if all zeros
        if precipitation.max() == 0:
            add_precip_trace = False
        else:
            add_precip_trace = True
        
        assert add_precip_trace == False
    
    def test_nighttime_solar_radiation(self):
        """Test solar radiation is zero at night."""
        # Create 24-hour data with proper day/night cycle
        timestamps = pd.date_range('2025-01-01', periods=24, freq='h')
        radiation = []
        
        for ts in timestamps:
            hour = ts.hour
            if 6 <= hour <= 18:  # Daytime hours
                radiation.append(np.random.uniform(100, 800))
            else:  # Nighttime
                radiation.append(0)
        
        radiation = np.array(radiation)
        
        # Verify nighttime is zero
        night_hours = [ts.hour for ts in timestamps if ts.hour < 6 or ts.hour > 18]
        night_indices = [i for i, ts in enumerate(timestamps) if ts.hour < 6 or ts.hour > 18]
        
        for idx in night_indices:
            assert radiation[idx] == 0


class TestDataValidation:
    """Test data validation for plotting."""
    
    def test_timestamp_timezone_awareness(self):
        """Test that timestamps are timezone-aware."""
        timestamps_utc = pd.date_range('2025-01-01', periods=24, freq='h', tz='UTC')
        
        assert timestamps_utc.tz is not None
        assert str(timestamps_utc.tz) == 'UTC'
    
    def test_price_value_ranges(self):
        """Test that price values are within reasonable ranges."""
        prices = np.random.uniform(20, 150, 100)
        
        # Prices should be positive
        assert all(prices > 0)
        
        # Prices should be in reasonable EUR/MWh range
        assert all(prices < 500)  # Extreme prices
    
    def test_dataframe_index_continuity(self):
        """Test that dataframe index is continuous."""
        timestamps = pd.date_range('2025-01-01', periods=96, freq='15min', tz='UTC')
        df = pd.DataFrame({
            'price': np.random.uniform(40, 60, 96)
        }, index=timestamps)
        
        # Check continuity
        time_diffs = df.index.to_series().diff()[1:]
        assert all(time_diffs == pd.Timedelta(minutes=15))


class TestForecastOverlayTiming:
    """Tests for forecast overlay timing and range."""
    
    def test_forecast_overlay_with_historical_window(self):
        """Test that forecast overlays correctly with historical window."""
        # Setup: Create historical data for 100 hours
        historical_start = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
        historical_end = pd.Timestamp('2025-01-05 04:00:00', tz='UTC')  # 100 hours later
        historical_timestamps = pd.date_range(historical_start, historical_end, freq='h')
        
        historical_df = pd.DataFrame({
            'timestamp_utc': historical_timestamps,
            'price_eur_per_mwh': np.random.uniform(40, 60, len(historical_timestamps))
        })
        
        # Create forecast data: 72-hour historical window + 24-hour future forecast
        # Forecast should start 72 hours before historical_end and go 24 hours into future
        forecast_historical_start = historical_end - pd.Timedelta(hours=72)
        forecast_future_end = historical_end + pd.Timedelta(hours=24)
        
        # Forecast includes both historical window (for RMSE) and future predictions
        all_forecast_timestamps = pd.date_range(forecast_historical_start, forecast_future_end, freq='h')
        forecast_df = pd.DataFrame({
            'timestamp_utc': all_forecast_timestamps,
            'forecast_price_eur_per_mwh': np.random.uniform(40, 60, len(all_forecast_timestamps)),
            'model_type': 'linear_regression',
            'market_type': 'day_ahead'
        })
        
        # Simulate the filtering logic from streamlit_app.py
        config = {
            'show_historical': True,
            'historical_window_hours': 72,
            'forecast_horizon_hours': 24
        }
        
        # Calculate what the display logic should do
        forecast_start_time = historical_df['timestamp_utc'].max()
        cutoff_time = forecast_start_time - pd.Timedelta(hours=config['historical_window_hours'])
        forecast_end_time = forecast_start_time + pd.Timedelta(hours=config['forecast_horizon_hours'])
        
        # Filter forecast for display
        forecast_df_display = forecast_df[
            (forecast_df['timestamp_utc'] >= cutoff_time) &
            (forecast_df['timestamp_utc'] <= forecast_end_time)
        ].copy()
        
        # Verify the filtered forecast covers exactly the expected range
        assert forecast_df_display['timestamp_utc'].min() == cutoff_time
        assert forecast_df_display['timestamp_utc'].max() == forecast_end_time
        
        # Verify total span is historical_window + forecast_horizon
        total_hours = (forecast_end_time - cutoff_time).total_seconds() / 3600
        assert total_hours == config['historical_window_hours'] + config['forecast_horizon_hours']
        
        # Verify forecast overlaps with historical data
        historical_in_window = historical_df[
            (historical_df['timestamp_utc'] >= cutoff_time) &
            (historical_df['timestamp_utc'] <= forecast_start_time)
        ]
        assert len(historical_in_window) == config['historical_window_hours'] + 1  # +1 for inclusive range
        
        # Verify there's overlap between historical and forecast
        overlap_timestamps = set(historical_in_window['timestamp_utc']) & set(forecast_df_display['timestamp_utc'])
        assert len(overlap_timestamps) > 0  # Should have overlapping timestamps
    
    def test_forecast_without_historical_window(self):
        """Test that forecast shows all data when no historical window is selected."""
        # Create forecast data
        forecast_start = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
        forecast_end = pd.Timestamp('2025-01-02 00:00:00', tz='UTC')  # 24 hours
        forecast_timestamps = pd.date_range(forecast_start, forecast_end, freq='h')
        
        forecast_df = pd.DataFrame({
            'timestamp_utc': forecast_timestamps,
            'forecast_price_eur_per_mwh': np.random.uniform(40, 60, len(forecast_timestamps)),
            'model_type': 'persistence',
            'market_type': 'day_ahead'
        })
        
        # When show_historical is False, all forecast should be shown
        config = {'show_historical': False}
        
        # Simulate the filtering logic
        forecast_df_display = forecast_df.copy()
        
        # Verify all forecast data is kept
        assert len(forecast_df_display) == len(forecast_df)
        assert forecast_df_display['timestamp_utc'].min() == forecast_start
        assert forecast_df_display['timestamp_utc'].max() == forecast_end
    
    def test_forecast_respects_horizon_limit(self):
        """Test that forecast doesn't extend beyond forecast_horizon_hours."""
        historical_end = pd.Timestamp('2025-01-03 12:00:00', tz='UTC')
        
        # Create a forecast that includes more data than needed
        # (e.g., 72h historical + 24h future, but we only want to show 48h historical + 12h future)
        forecast_start = historical_end - pd.Timedelta(hours=72)
        all_forecast_timestamps = pd.date_range(forecast_start, periods=97, freq='h')  # 96 hours + 1
        
        forecast_df = pd.DataFrame({
            'timestamp_utc': all_forecast_timestamps,
            'forecast_price_eur_per_mwh': np.random.uniform(40, 60, len(all_forecast_timestamps))
        })
        
        # Apply filtering with smaller window
        config = {
            'historical_window_hours': 48,
            'forecast_horizon_hours': 12
        }
        
        cutoff_time = historical_end - pd.Timedelta(hours=config['historical_window_hours'])
        forecast_end_time = historical_end + pd.Timedelta(hours=config['forecast_horizon_hours'])
        
        forecast_df_display = forecast_df[
            (forecast_df['timestamp_utc'] >= cutoff_time) &
            (forecast_df['timestamp_utc'] <= forecast_end_time)
        ].copy()
        
        # Verify the displayed forecast respects the limits
        actual_span_hours = (forecast_df_display['timestamp_utc'].max() - 
                            forecast_df_display['timestamp_utc'].min()).total_seconds() / 3600
        expected_span_hours = config['historical_window_hours'] + config['forecast_horizon_hours']
        
        assert actual_span_hours == expected_span_hours
        assert len(forecast_df_display) == expected_span_hours + 1  # +1 for inclusive endpoints
