"""
Quick test to verify 15-minute forecast resolution and RMSE calculation.
"""

from datetime import datetime
from app.services.forecast_service import ForecastService, TrainingDataConfig

# Initialize service
service = ForecastService()
training_config = TrainingDataConfig(use_weather_data=True)

# Generate a 24-hour forecast
print("Generating 24-hour forecast with 15-minute resolution...")
forecast_start = datetime(2025, 11, 18, 0, 0, 0)

df_forecast = service.generate_forecast(
    market_type="day_ahead",
    model_type="linear_regression",
    horizon_hours=24,
    forecast_start=forecast_start,
    training_config=training_config
)

if df_forecast is not None:
    print(f"\n✅ Forecast generated successfully!")
    print(f"   Number of forecast points: {len(df_forecast)}")
    print(f"   Expected (24h × 4): 96")
    print(f"   First timestamp: {df_forecast['timestamp_utc'].iloc[0]}")
    print(f"   Last timestamp: {df_forecast['timestamp_utc'].iloc[-1]}")
    print(f"   Time difference: {df_forecast['timestamp_utc'].iloc[-1] - df_forecast['timestamp_utc'].iloc[0]}")
    
    # Check interval between rows
    time_diff = df_forecast['timestamp_utc'].iloc[1] - df_forecast['timestamp_utc'].iloc[0]
    print(f"   Interval between points: {time_diff}")
    
    # Calculate RMSE
    print("\nCalculating RMSE...")
    rmse = service.calculate_forecast_rmse(df_forecast, "day_ahead")
    
    if rmse is not None:
        print(f"   ✅ RMSE: {rmse:.2f} EUR/MWh")
    else:
        print("   ⚠️ RMSE: N/A (no overlapping historical data)")
    
    # Show sample forecasts
    print("\nSample forecast values:")
    print(df_forecast[['timestamp_utc', 'forecast_price_eur_per_mwh']].head(10).to_string(index=False))
    
else:
    print("❌ Forecast generation failed!")

print("\n" + "="*60)
print("Test complete!")
