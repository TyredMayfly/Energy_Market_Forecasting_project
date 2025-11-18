"""
Generate demo CSV files with 15-minute resolution data from Jan 1 to Nov 18, 2025.
Extends weather data 36 hours into the future for forecasting.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate timestamps for historical data
start = datetime(2025, 1, 1, 0, 0, 0)
end = datetime(2025, 11, 18, 23, 45, 0)
num_intervals = int((end - start).total_seconds() / (15 * 60)) + 1
timestamps = [start + timedelta(minutes=15 * i) for i in range(num_intervals)]

print(f"Generating {num_intervals} historical records ({num_intervals/96:.1f} days) from {start} to {end}")

# Generate weather timestamps extending 36 hours into the future
weather_end = end + timedelta(hours=36)
num_weather_intervals = int((weather_end - start).total_seconds() / (15 * 60)) + 1
weather_timestamps = [start + timedelta(minutes=15 * i) for i in range(num_weather_intervals)]

print(f"Generating {num_weather_intervals} weather records ({num_weather_intervals/96:.1f} days) from {start} to {weather_end}")

# Generate day-ahead prices with realistic patterns
# Base price with daily and weekly cycles
hours = np.array([(ts.hour + ts.minute/60) for ts in timestamps])
days = np.array([ts.weekday() for ts in timestamps])

# Daily pattern: higher during peak hours (8-20), lower at night
daily_pattern = 15 * np.sin((hours - 6) * np.pi / 12)
# Weekly pattern: higher on weekdays
weekly_pattern = 5 * (days < 5).astype(float)
# Random variation
noise = np.random.normal(0, 10, num_intervals)

day_ahead_prices = 50 + daily_pattern + weekly_pattern + noise
day_ahead_prices = np.clip(day_ahead_prices, 20, 100)  # Keep realistic bounds

df_day_ahead = pd.DataFrame({
    'timestamp_utc': timestamps,
    'price_eur_per_mwh': day_ahead_prices,
    'market_type': 'day_ahead'
})

# Generate intraday prices (slightly different from day-ahead)
intraday_prices = day_ahead_prices + np.random.normal(0, 5, num_intervals)
intraday_prices = np.clip(intraday_prices, 15, 110)

df_intraday = pd.DataFrame({
    'timestamp_utc': timestamps,
    'price_eur_per_mwh': intraday_prices,
    'market_type': 'intraday'
})

# Generate imbalance data
imbalance_volumes = np.random.normal(0, 200, num_intervals)
imbalance_prices = day_ahead_prices + np.random.normal(0, 15, num_intervals)
imbalance_prices = np.clip(imbalance_prices, 0, 150)

df_imbalance = pd.DataFrame({
    'timestamp_utc': timestamps,
    'imbalance_volume_mwh': imbalance_volumes,
    'imbalance_price_eur_per_mwh': imbalance_prices
})

# Generate weather data
# Temperature: seasonal variation + daily cycle
weather_day_of_year = np.array([ts.timetuple().tm_yday for ts in weather_timestamps])
weather_hours = np.array([(ts.hour + ts.minute/60) for ts in weather_timestamps])
seasonal_temp = 10 + 15 * np.sin((weather_day_of_year - 80) * 2 * np.pi / 365)  # Peak in summer
daily_temp_variation = 5 * np.sin(weather_hours * np.pi / 12)
temperature = seasonal_temp + daily_temp_variation + np.random.normal(0, 2, num_weather_intervals)

# Wind speed: more variable
wind_speed = np.abs(np.random.normal(15, 8, num_weather_intervals))
wind_speed = np.clip(wind_speed, 0, 40)

# Cloud cover: 0-8 oktas
cloud_cover = np.random.randint(0, 9, num_weather_intervals)

# Precipitation: mostly dry with occasional rain
precipitation = np.random.exponential(0.1, num_weather_intervals)
precipitation = np.clip(precipitation, 0, 10)

df_weather = pd.DataFrame({
    'timestamp_utc': weather_timestamps,
    'temperature_celsius': temperature,
    'wind_speed_ms': wind_speed,
    'cloud_cover_oktas': cloud_cover,
    'precipitation_mm': precipitation
})

# Save to CSV files
data_dir = 'data'
df_day_ahead.to_csv(f'{data_dir}/entsoe_day_ahead_prices_2025.csv', index=False)
df_intraday.to_csv(f'{data_dir}/entsoe_intraday_prices_2025.csv', index=False)
df_imbalance.to_csv(f'{data_dir}/entsoe_imbalance_data_2025.csv', index=False)
df_weather.to_csv(f'{data_dir}/knmi_weather_2025.csv', index=False)

print(f"\n✅ Generated CSV files in {data_dir}/:")
print(f"   - entsoe_day_ahead_prices_2025.csv: {len(df_day_ahead)} rows")
print(f"   - entsoe_intraday_prices_2025.csv: {len(df_intraday)} rows")
print(f"   - entsoe_imbalance_data_2025.csv: {len(df_imbalance)} rows")
print(f"   - knmi_weather_2025.csv: {len(df_weather)} rows")
print(f"\nPrice stats (day-ahead):")
print(f"   Mean: {day_ahead_prices.mean():.2f} EUR/MWh")
print(f"   Min: {day_ahead_prices.min():.2f} EUR/MWh")
print(f"   Max: {day_ahead_prices.max():.2f} EUR/MWh")
