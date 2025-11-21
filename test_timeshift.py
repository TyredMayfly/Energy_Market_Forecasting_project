import pandas as pd

# Simulate the time-shift logic
df = pd.DataFrame({
    'Q': [360, 720],
    'timestamp': pd.to_datetime(['2025-01-01 01:00', '2025-01-01 12:00'])
})

df_cum = pd.DataFrame()
df_cum['timestamp'] = df['timestamp']
df_cum['global_radiation_w_per_m2'] = df['Q'] * 10000 / 3600

print("Original data:")
print(df_cum)

df_cum = df_cum.set_index('timestamp')
print("\nAfter set_index:")
print(df_cum)

df_cum.index = df_cum.index - pd.Timedelta(minutes=30)
print("\nAfter time-shift (-30min):")
print(df_cum)

df_interp = df_cum.resample('1h').interpolate(method='linear')
print("\nAfter resample + interpolate:")
print(df_interp)
print(f"\nShape: {df_interp.shape}")
print(f"Non-null values: {df_interp['global_radiation_w_per_m2'].notna().sum()}")
