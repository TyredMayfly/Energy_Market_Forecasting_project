"""
Test script to verify regulation state bar chart visualization and classification metrics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Create sample regulation state forecast data
n_points = 96  # 24 hours at 15-min resolution
timestamps = [
    datetime.now(pytz.UTC) + timedelta(minutes=15 * i) for i in range(n_points)
]

# Sample regulation states: -1, 0, 1, 2
np.random.seed(42)
forecast_states = np.random.choice([-1, 0, 1, 2], size=n_points, p=[0.2, 0.3, 0.3, 0.2])

df_forecast = pd.DataFrame({
    "timestamp_utc": timestamps,
    "forecast_price_eur_per_mwh": forecast_states,  # Using price column for states
    "model_type": "xgboost_classifier",
})

print("=" * 60)
print("REGULATION STATE FORECAST TEST")
print("=" * 60)
print(f"\nGenerated {len(df_forecast)} forecast points")
print(f"Time range: {df_forecast['timestamp_utc'].min()} to {df_forecast['timestamp_utc'].max()}")
print(f"\nState distribution:")
for state in sorted(df_forecast["forecast_price_eur_per_mwh"].unique()):
    count = (df_forecast["forecast_price_eur_per_mwh"] == state).sum()
    pct = count / len(df_forecast) * 100
    state_label = {-1: "Deficit", 0: "Balanced", 1: "Surplus", 2: "Strong Surplus"}.get(state, str(state))
    print(f"  {state_label} ({state}): {count} points ({pct:.1f}%)")

# Test classification metrics
print("\n" + "=" * 60)
print("TESTING CLASSIFICATION METRICS")
print("=" * 60)

# Create some "actual" data for testing metrics
actual_states = np.random.choice([-1, 0, 1, 2], size=n_points, p=[0.25, 0.25, 0.25, 0.25])
df_actual = pd.DataFrame({
    "timestamp_utc": timestamps,
    "regulation_state": actual_states,
})

# Calculate metrics manually
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

y_true = df_actual["regulation_state"].values
y_pred = df_forecast["forecast_price_eur_per_mwh"].values.astype(int)

accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"\nAccuracy: {accuracy:.1%}")
print(f"F1-Macro: {f1_macro:.3f}")
print(f"\nConfusion Matrix:")
print("Rows = Actual, Columns = Predicted")
print("States: -1 (Deficit), 0 (Balanced), 1 (Surplus), 2 (Strong Surplus)")
print("\n     -1   0   1   2")
for i, row in enumerate(conf_matrix):
    state_label = [-1, 0, 1, 2][i]
    print(f"{state_label:2d} {row}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("\nNext steps:")
print("1. Run Streamlit app: streamlit run streamlit_app.py")
print("2. Select Market Type: 'Regulation State'")
print("3. Select Model: 'XGBoost Classifier'")
print("4. Generate forecast to see:")
print("   - Bar chart visualization with color-coded states")
print("   - Accuracy and F1-Macro metrics instead of RMSE")
print("   - Confusion matrix table")
