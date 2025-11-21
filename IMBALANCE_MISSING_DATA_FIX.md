# Imbalance Price Missing Data - Root Cause Analysis and Fix

## Executive Summary

**Problem**: Imbalance shortage/surplus price graphs show missing data points (gaps in visualization).

**Root Cause**: Raw imbalance CSV data contains ~3% NULL/empty values (1,170 missing shortage prices, 1,164 missing surplus prices out of 39,740 total records).

**Solution**: Implemented forward-fill imputation in `imbalance_data_loader.py` to handle missing values while preserving temporal continuity.

**Status**: ✅ **FIXED** - All missing values imputed, all tests passing (432/432)

---

## Investigation Process

### 1. Data Flow Analysis

Traced imbalance data through the full pipeline:

```
Raw CSV Files (imbalance_data/*.csv)
    ↓
imbalance_data_loader.py → load_and_combine_imbalance_csvs()
    ↓
Combined DataFrame with 4 columns:
  - timestamp_utc
  - shortage_price  ← MISSING VALUES HERE
  - surplus_price   ← MISSING VALUES HERE
  - regulation_state
    ↓
Saved to: data/imbalance_unified.csv
    ↓
Loaded by: data_store.py → load_market_data()
    ↓
Used by: feature_engineering.py → build_features_and_target()
    ↓
Displayed in: Streamlit UI / FastAPI responses
```

### 2. Missing Data Discovery

**Analysis Script**: `analyze_missing_data.py`

**Findings**:
- **Dataset size**: 39,740 records (Oct 1, 2024 → Nov 18, 2025)
- **Missing shortage_price**: 1,170 rows (2.94%)
- **Missing surplus_price**: 1,164 rows (2.93%)
- **Missing data period**: November 1-18, 2025 (recent data)
- **Pattern**: Missing values appear in clusters, suggesting source data quality issues

**Sample Missing Data**:
```
timestamp_utc       shortage_price  surplus_price  regulation_state
2025-11-01 00:00:00      NaN            NaN              -1
2025-11-01 00:15:00      NaN            NaN               2
2025-11-01 00:30:00      NaN           -4.0               2
2025-11-01 00:45:00      NaN            NaN              -1
```

### 3. Impact Analysis

**Visualization Impact**:
- Plotly line graphs skip NaN values, creating gaps
- Bar charts (for regulation state) appear discontinuous
- Users perceive incomplete data

**Feature Engineering Impact**:
- Lag features propagate NaNs (e.g., lag_1_price, lag_24_price)
- Rolling statistics (mean, std) affected by missing windows
- Model training drops rows with NaN features → reduced training data

**RMSE Calculation Impact**:
- `calculate_forecast_rmse()` merges on timestamp
- Missing actual prices → fewer overlap points
- RMSE calculated on incomplete comparison

---

## Solution Implemented

### Imputation Strategy: Forward-Fill with Back-Fill Fallback

**Rationale**:
1. **Temporal Continuity**: Imbalance prices at 15-minute resolution don't change drastically
2. **Last Known Value**: Forward-fill assumes price stability between intervals
3. **Start-of-Dataset Handling**: Back-fill handles any NaNs at the beginning

**Implementation** (`app/services/imbalance_data_loader.py`):

```python
# Convert prices to numeric, handling any non-numeric values
for price_col in ["shortage_price", "surplus_price"]:
    combined_df[price_col] = pd.to_numeric(combined_df[price_col], errors="coerce")

# Handle missing price values using forward-fill strategy
missing_shortage_before = combined_df["shortage_price"].isnull().sum()
missing_surplus_before = combined_df["surplus_price"].isnull().sum()

if missing_shortage_before > 0 or missing_surplus_before > 0:
    logger.info(
        f"Found {missing_shortage_before} missing shortage prices, "
        f"{missing_surplus_before} missing surplus prices"
    )
    logger.info("Applying forward-fill imputation to handle missing values...")

    # Forward-fill: use last known price to fill gaps
    combined_df["shortage_price"] = combined_df["shortage_price"].ffill()
    combined_df["surplus_price"] = combined_df["surplus_price"].ffill()

    # Back-fill any remaining NaNs at the start of the dataset
    combined_df["shortage_price"] = combined_df["shortage_price"].bfill()
    combined_df["surplus_price"] = combined_df["surplus_price"].bfill()

    missing_shortage_after = combined_df["shortage_price"].isnull().sum()
    missing_surplus_after = combined_df["surplus_price"].isnull().sum()

    logger.info(
        f"After imputation: {missing_shortage_after} missing shortage, "
        f"{missing_surplus_after} missing surplus"
    )
```

### Verification Results

**Before Fix**:
- Missing shortage_price: 1,170 rows (2.94%)
- Missing surplus_price: 1,164 rows (2.93%)

**After Fix**:
- Missing shortage_price: **0 rows (0.00%)**
- Missing surplus_price: **0 rows (0.00%)**

**Rebuild Command**:
```bash
python -m app.services.imbalance_data_loader
```

**Output**:
```
Found 1170 missing shortage prices, 1164 missing surplus prices
Applying forward-fill imputation to handle missing values...
After imputation: 0 missing shortage, 0 missing surplus

✅ Unified dataset created: data/imbalance_unified.csv
Dataset shape: (39740, 4)
Shortage price range: -1500.00 to 5500.00 EUR/MWh
Surplus price range: -1507.88 to 5500.00 EUR/MWh
```

---

## Testing

### Updated Tests

**File**: `tests/services/test_imbalance_data_loader.py`

**Modified Test**: `test_load_converts_prices_to_numeric`

**Before** (expected NaN preservation):
```python
assert pd.isna(result["shortage_price"].iloc[1])
```

**After** (expects forward-fill):
```python
assert result["shortage_price"].iloc[1] == 50.5  # forward-filled from row 0
assert result["shortage_price"].isnull().sum() == 0  # no NaNs remain
```

### Test Results

```
tests/services/test_imbalance_data_loader.py::TestLoadAndCombineImbalanceCSVs::test_load_single_csv PASSED
tests/services/test_imbalance_data_loader.py::TestLoadAndCombineImbalanceCSVs::test_load_multiple_csvs PASSED
tests/services/test_imbalance_data_loader.py::TestLoadAndCombineImbalanceCSVs::test_load_removes_duplicates PASSED
tests/services/test_imbalance_data_loader.py::TestLoadAndCombineImbalanceCSVs::test_load_converts_prices_to_numeric PASSED
... (10/10 tests passed)

Full Test Suite: 432 passed in 102.31s
```

---

## Impact Assessment

### ✅ **What Changed**
1. **Data Loading**: `imbalance_data_loader.py` now imputes missing values
2. **Unified Dataset**: `data/imbalance_unified.csv` rebuilt with 0 missing values
3. **Visualization**: Graphs now show continuous lines without gaps
4. **Feature Engineering**: No NaN propagation from price columns
5. **Model Training**: Full dataset available (no rows dropped due to NaN features)

### ✅ **What Stayed the Same**
1. **Data Structure**: Same 4 columns, same timestamp range
2. **Price Ranges**: Min/max prices unchanged (-1500 to 5500 EUR/MWh)
3. **Regulation States**: No changes (states are never missing)
4. **API Contracts**: No breaking changes to endpoints
5. **Existing Forecasts**: Price forecast logic unaffected

### ⚠️ **Tradeoffs**

**Pros**:
- ✅ Complete visualizations (no gaps)
- ✅ More training data (no NaN-based row drops)
- ✅ Stable feature engineering (no NaN propagation)
- ✅ Better RMSE calculations (more overlap points)

**Cons**:
- ⚠️ Imputed values != actual missing values
- ⚠️ Forward-fill assumes price stability (may not hold during volatile periods)
- ⚠️ Slightly higher mean/variance in imputed regions

**Risk Mitigation**:
- Missing data is only ~3% of total dataset
- Missing values concentrated in November 2025 (recent period)
- 15-minute resolution makes forward-fill reasonable assumption
- Alternative (interpolation) would be more complex with similar limitations

---

## Alternative Solutions Considered

### 1. **Linear Interpolation**
```python
combined_df["shortage_price"] = combined_df["shortage_price"].interpolate(method="linear")
```
**Pros**: Smoother transitions between known values  
**Cons**: Assumes linear price changes (unrealistic for energy markets)  
**Verdict**: Forward-fill more appropriate for step-like price behavior

### 2. **Drop Rows with Missing Data**
```python
combined_df = combined_df.dropna(subset=["shortage_price", "surplus_price"])
```
**Pros**: No imputed values, only real data  
**Cons**: Breaks temporal continuity, reduces dataset by ~3%, complicates lag features  
**Verdict**: Unacceptable loss of temporal structure

### 3. **Use API to Fetch Missing Data**
```python
# Fetch missing timestamps from TenneT API
```
**Pros**: Real data instead of imputed  
**Cons**: Missing data may not exist in API either, adds complexity, API rate limits  
**Verdict**: Not viable if source data is incomplete

### 4. **Flag Imputed Values**
```python
combined_df["shortage_price_imputed"] = combined_df["shortage_price"].isnull()
```
**Pros**: Transparency about which values are imputed  
**Cons**: Adds complexity to downstream code, models would need to handle flags  
**Verdict**: Over-engineering for ~3% of data

---

## Files Modified

1. **app/services/imbalance_data_loader.py**
   - Added forward-fill/back-fill imputation logic
   - Updated pandas API (ffill()/bfill() instead of deprecated fillna(method=...))
   - Added logging for imputation process

2. **tests/services/test_imbalance_data_loader.py**
   - Updated `test_load_converts_prices_to_numeric` to expect imputed values
   - Added assertions to verify no NaNs remain after imputation

3. **data/imbalance_unified.csv**
   - Rebuilt with imputed values (all 39,740 rows now have valid prices)

4. **analyze_missing_data.py** (new diagnostic script)
   - Analysis tool to identify missing data patterns
   - Useful for future data quality monitoring

---

## Deployment Steps

### 1. Rebuild Unified Dataset
```bash
cd C:/Users/thdeb/Projects/Market_Forecasting_example
python -m app.services.imbalance_data_loader
```

### 2. Verify No Missing Data
```bash
python analyze_missing_data.py
```

**Expected Output**:
```
Missing shortage_price: 0 / 39740 (0.00%)
Missing surplus_price: 0 / 39740 (0.00%)
```

### 3. Run Full Test Suite
```bash
pytest --tb=short -q
```

**Expected**: `432 passed`

### 4. Test Streamlit UI
```bash
streamlit run streamlit_app.py
```

**Steps**:
1. Select Market Type: "Imbalance - Shortage Price"
2. Generate Forecast
3. **Verify**: Graph shows continuous line (no gaps)
4. Repeat for "Imbalance - Surplus Price"

---

## Future Recommendations

### 1. **Data Quality Monitoring**
- Add periodic checks for missing data in incoming CSVs
- Alert when missing data > 5% threshold
- Log missing data statistics in data refresh pipeline

### 2. **Upstream Data Source Investigation**
- Contact TenneT to understand why ~3% of November 2025 data is missing
- Determine if missing data is systematic (e.g., maintenance periods)
- Establish data quality SLA with provider

### 3. **Advanced Imputation (Optional)**
- For critical forecasts, consider time-series-aware imputation (e.g., ARIMA-based)
- Use multiple imputation to quantify uncertainty from imputed values
- Implement separate models for imputed vs. real data

### 4. **Documentation**
- Add data quality section to `ARCHITECTURE_AND_DESIGN_OVERVIEW.md`
- Document imputation strategy in data pipeline documentation
- Include imputation as part of data refresh service logs

---

## Conclusion

The missing data issue in imbalance shortage/surplus price graphs was caused by **NULL values in the raw source CSV files** (~3% of data). The fix implements **forward-fill imputation** to preserve temporal continuity while maintaining data completeness.

**Key Outcomes**:
- ✅ **0% missing data** (down from 2.94%)
- ✅ **Continuous visualizations** (no gaps)
- ✅ **All tests passing** (432/432)
- ✅ **No breaking changes** to existing functionality
- ✅ **Simple, maintainable solution** (~30 lines of code)

The fix is production-ready and follows best practices for time-series data imputation in energy market forecasting applications.
