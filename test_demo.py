"""Quick test of demo mode forecast generation."""
from app.services.forecast_service import get_forecast_service, TrainingDataConfig
from datetime import datetime, timezone

fs = get_forecast_service()

# Test with weather data
config_with_weather = TrainingDataConfig(use_weather_data=True)
result_with_weather = fs.generate_forecast(
    market_type='day_ahead',
    model_type='persistence', 
    horizon_hours=24,
    forecast_start=datetime(2025, 11, 18, 20, 0, tzinfo=timezone.utc),
    training_config=config_with_weather,
)

if result_with_weather is not None:
    print(f"✅ Forecast with weather generated successfully!")
    print(f"   Shape: {result_with_weather.shape}")
    print(f"   Mean forecast: {result_with_weather['forecast_price_eur_per_mwh'].mean():.2f} EUR/MWh")

# Test without weather data
config_without_weather = TrainingDataConfig(use_weather_data=False)
result_without_weather = fs.generate_forecast(
    market_type='day_ahead',
    model_type='linear_regression', 
    horizon_hours=24,
    forecast_start=datetime(2025, 11, 18, 20, 0, tzinfo=timezone.utc),
    training_config=config_without_weather,
)

if result_without_weather is not None:
    print(f"\n✅ Forecast without weather generated successfully!")
    print(f"   Shape: {result_without_weather.shape}")
    print(f"   Mean forecast: {result_without_weather['forecast_price_eur_per_mwh'].mean():.2f} EUR/MWh")

# Test training data retrieval
training_data = fs.get_training_data(
    market_type='day_ahead',
    model_type='persistence',
    training_config=config_with_weather,
)

if training_data is not None:
    print(f"\n✅ Training data retrieved successfully!")
    print(f"   Shape: {training_data.shape}")
    print(f"   Date range: {training_data['timestamp_utc'].min()} to {training_data['timestamp_utc'].max()}")
else:
    print("❌ Failed to retrieve training data")
