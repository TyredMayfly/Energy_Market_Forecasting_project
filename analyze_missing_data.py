"""
Analyze missing data in imbalance_unified.csv
"""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/imbalance_unified.csv')

print("=" * 70)
print("IMBALANCE DATA ANALYSIS")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(df.isnull().sum())

# Calculate percentages
total = len(df)
shortage_missing = df['shortage_price'].isnull().sum()
surplus_missing = df['surplus_price'].isnull().sum()

print(f"\nMissing shortage_price: {shortage_missing} / {total} ({shortage_missing/total*100:.2f}%)")
print(f"Missing surplus_price: {surplus_missing} / {total} ({surplus_missing/total*100:.2f}%)")

print("\n" + "=" * 70)
print("SAMPLE OF ROWS WITH MISSING SHORTAGE_PRICE")
print("=" * 70)
missing_shortage_df = df[df['shortage_price'].isnull()]
if not missing_shortage_df.empty:
    print(missing_shortage_df.head(10))
    print(f"\nDate range: {missing_shortage_df['timestamp_utc'].min()} to {missing_shortage_df['timestamp_utc'].max()}")

print("\n" + "=" * 70)
print("SAMPLE OF ROWS WITH MISSING SURPLUS_PRICE")
print("=" * 70)
missing_surplus_df = df[df['surplus_price'].isnull()]
if not missing_surplus_df.empty:
    print(missing_surplus_df.head(10))
    print(f"\nDate range: {missing_surplus_df['timestamp_utc'].min()} to {missing_surplus_df['timestamp_utc'].max()}")

print("\n" + "=" * 70)
print("VALID DATA ANALYSIS")
print("=" * 70)
valid_shortage = df[df['shortage_price'].notnull()]
valid_surplus = df[df['surplus_price'].notnull()]

print(f"\nValid shortage price rows: {len(valid_shortage)}")
print(f"Valid surplus price rows: {len(valid_surplus)}")

if not valid_shortage.empty:
    print(f"\nShortage price range: {valid_shortage['shortage_price'].min():.2f} to {valid_shortage['shortage_price'].max():.2f}")
    print(f"Shortage price mean: {valid_shortage['shortage_price'].mean():.2f}")

if not valid_surplus.empty:
    print(f"\nSurplus price range: {valid_surplus['surplus_price'].min():.2f} to {valid_surplus['surplus_price'].max():.2f}")
    print(f"Surplus price mean: {valid_surplus['surplus_price'].mean():.2f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("\nRoot cause: Raw imbalance CSV data contains NULL/empty values")
print("These NULLs are preserved during data loading and cause gaps in graphs")
print("\nRecommended fix:")
print("1. Forward-fill missing values (use last known price)")
print("2. Or interpolate linearly between known values")
print("3. Or drop rows with missing data (not recommended - loses temporal continuity)")
