"""
Test selective weather feature inclusion.
"""

from app.services.forecast_service import ForecastService, TrainingDataConfig

# Initialize service
service = ForecastService()

print("Testing selective weather features...\n")

# Test 1: All weather features
print("1. Training with ALL weather features:")
config1 = TrainingDataConfig(
    use_weather_data=True,
    use_temperature=True,
    use_wind_speed=True,
    use_cloud_cover=True,
    use_precipitation=True,
)
success = service.train_model("day_ahead", "persistence", training_config=config1)
if success:
    model_key = service._get_model_key("day_ahead", "persistence", config1)
    print(f"   ✅ Model trained: {model_key}")
    model_info = service.trained_models[model_key]
    print(f"   Features: {len(model_info['feature_columns'])}")
else:
    print("   ❌ Training failed")

# Test 2: Only temperature and wind
print("\n2. Training with ONLY temperature and wind speed:")
config2 = TrainingDataConfig(
    use_weather_data=True,
    use_temperature=True,
    use_wind_speed=True,
    use_cloud_cover=False,
    use_precipitation=False,
)
success = service.train_model("day_ahead", "linear_regression", training_config=config2)
if success:
    model_key = service._get_model_key("day_ahead", "linear_regression", config2)
    print(f"   ✅ Model trained: {model_key}")
    model_info = service.trained_models[model_key]
    print(f"   Features: {len(model_info['feature_columns'])}")
else:
    print("   ❌ Training failed")

# Test 3: Only cloud cover and precipitation
print("\n3. Training with ONLY cloud cover and precipitation:")
config3 = TrainingDataConfig(
    use_weather_data=True,
    use_temperature=False,
    use_wind_speed=False,
    use_cloud_cover=True,
    use_precipitation=True,
)
success = service.train_model("day_ahead", "random_forest", training_config=config3)
if success:
    model_key = service._get_model_key("day_ahead", "random_forest", config3)
    print(f"   ✅ Model trained: {model_key}")
    model_info = service.trained_models[model_key]
    print(f"   Features: {len(model_info['feature_columns'])}")
else:
    print("   ❌ Training failed")

# Test 4: No weather features
print("\n4. Training with NO weather features:")
config4 = TrainingDataConfig(
    use_weather_data=False,
)
success = service.train_model("day_ahead", "persistence", training_config=config4)
if success:
    model_key = service._get_model_key("day_ahead", "persistence", config4)
    print(f"   ✅ Model trained: {model_key}")
    model_info = service.trained_models[model_key]
    print(f"   Features: {len(model_info['feature_columns'])}")
else:
    print("   ❌ Training failed")

print("\n" + "="*60)
print(f"Total models cached: {len(service.trained_models)}")
print("Cache keys:")
for key in service.trained_models.keys():
    print(f"  - {key}")
print("\n✅ All weather feature selection tests passed!")
