# KNMI Time-Shift and Interpolation Implementation

## Executive Summary

This document explains the time-shift correction and interpolation applied to KNMI (Royal Netherlands Meteorological Institute) hourly weather data to ensure accurate alignment with market data timestamps.

**Problem**: KNMI cumulative measurements (radiation, precipitation) represent values collected over the PREVIOUS hour, but are timestamped at the END of that interval.

**Solution**: Shift cumulative field timestamps backward by 30 minutes to place values at the center of their measurement interval, then interpolate to create a smooth hourly time series.

**Impact**: Improved temporal accuracy of weather features in forecasting models, especially for solar radiation which directly affects electricity generation.

---

## Background: KNMI Data Format

### Hourly Data Structure

KNMI provides hourly weather observations with the following characteristics:

1. **Hour Numbering**: Uses hours 1-24 (not 0-23)
   - Hour 1 = 01:00 (after midnight)
   - Hour 24 = 00:00 (midnight of next day)

2. **Timestamp Meaning**: 
   - For **instantaneous fields** (temperature, humidity, pressure): Value AT that timestamp
   - For **cumulative fields** (radiation, precipitation): Total accumulated during PREVIOUS hour

### Example: Global Radiation

```
KNMI Timestamp  | Meaning for Radiation (Q)
----------------|------------------------------------------
2025-01-01 01:00 | Total solar energy from 00:00 to 01:00
2025-01-01 02:00 | Total solar energy from 01:00 to 02:00
2025-01-01 12:00 | Total solar energy from 11:00 to 12:00
```

**Problem**: The timestamp 12:00 suggests "radiation at noon", but it actually represents radiation collected between 11:00 and 12:00. The midpoint of that interval is 11:30, not 12:00.

**Why This Matters**: 
- Solar radiation peaks around solar noon (≈13:00 in Netherlands)
- Without correction, radiation data appears to peak 30 minutes later than reality
- This temporal misalignment degrades forecast accuracy for solar-dependent electricity generation

---

## Field Classification

### Cumulative Fields (Require Time-Shift)

| Field | Description | Unit | KNMI Code |
|-------|-------------|------|-----------|
| Global Radiation | Solar energy over previous hour | J/cm² | Q |
| Precipitation Duration | Hours of rain over previous hour | 0.1 hour | RH |
| Precipitation Amount | Total rainfall over previous hour | 0.1 mm | DR |
| Sunshine Duration | Hours of sunshine over previous hour | 0.1 hour | SQ |
| Evapotranspiration | Potential evaporation over previous 24h | 0.1 mm | EV24 |

**Common characteristic**: All represent ACCUMULATED or SUMMED quantities measured during a past interval.

### Instantaneous/Averaged Fields (No Time-Shift)

| Field | Description | Unit | KNMI Code |
|-------|-------------|------|-----------|
| Temperature | Air temperature at timestamp | 0.1 °C | T |
| Dew Point | Dew point temperature at timestamp | 0.1 °C | TD |
| Wind Speed | Average wind speed during previous hour | 0.1 m/s | FH |
| Wind Direction | Average wind direction during previous hour | degrees | DD |
| Cloud Cover | Sky coverage at timestamp | oktas (0-8) | N |
| Humidity | Relative humidity at timestamp | % | U |
| Pressure | Air pressure at timestamp | 0.1 hPa | P |
| Visibility | Horizontal visibility at timestamp | coded | VV |

**Note**: Wind speed/direction are AVERAGED over the previous hour but treated as instantaneous because the averaging is inherent to the meteorological measurement process, not a data accumulation.

---

## Implementation Details

### Code Location

**Module**: `app/services/knmi_historical_client.py`  
**Function**: `convert_to_standard_format()`  
**Lines**: ~237-310

### Algorithm Steps

#### 1. Field Separation

```python
df_instantaneous = pd.DataFrame()
df_cumulative = pd.DataFrame()

# Instantaneous fields (temperature, wind, cloud cover)
if "T" in df.columns:
    df_instantaneous["temperature_deg_c"] = df["T"] / 10.0

# Cumulative fields (radiation, precipitation)
if "Q" in df.columns:
    df_cumulative["global_radiation_w_per_m2"] = df["Q"] * 10000 / 3600
```

#### 2. Time-Shift for Cumulative Fields

```python
# Shift timestamps backward by 30 minutes
df_cumulative.index = df_cumulative.index - pd.Timedelta(minutes=30)
```

**Effect**:
```
Before shift:        After shift:
01:00 → 1000 W/m²    00:30 → 1000 W/m²  (center of 00:00-01:00)
02:00 → 1200 W/m²    01:30 → 1200 W/m²  (center of 01:00-02:00)
12:00 → 2000 W/m²    11:30 → 2000 W/m²  (center of 11:00-12:00)
```

#### 3. Interpolation to Hourly Grid

**Challenge**: After shifting, cumulative data is at :30 timestamps (00:30, 01:30, ...), but we need hourly data (00:00, 01:00, ...) to match market data resolution.

**Solution**: Linear interpolation

```python
# Create hourly grid covering shifted data range
start_hour = df_cumulative.index.min().floor("1h")  # Round down to hour
end_hour = df_cumulative.index.max().ceil("1h")      # Round up to hour
hourly_index = pd.date_range(start=start_hour, end=end_hour, freq="1h")

# Combine shifted data with hourly grid
combined_index = df_cumulative.index.union(hourly_index).sort_values()
df_cumulative = df_cumulative.reindex(combined_index)

# Interpolate linearly between :30 timestamps to fill :00 timestamps
df_cumulative = df_cumulative.interpolate(method="linear", limit_direction="both")

# Keep only hourly grid points
df_cumulative = df_cumulative.reindex(hourly_index)
```

**Example**:
```
Shifted data:       Combined index:     After interpolation:   Final hourly:
00:30 → 1000        00:00 → NaN         00:00 → 917            00:00 → 917
01:30 → 1200        00:30 → 1000        00:30 → 1000           01:00 → 1100
                    01:00 → NaN         01:00 → 1100           02:00 → 1300
                    01:30 → 1200        01:30 → 1200           
                    02:00 → NaN         02:00 → 1300
```

**Rationale**: Linear interpolation assumes radiation/precipitation changes smoothly between hourly samples, which is reasonable for averaged meteorological quantities.

#### 4. Merging Instantaneous and Cumulative Fields

```python
# Outer join to combine all timestamps
result = df_instantaneous.join(df_cumulative, how="outer")

# Forward-fill and back-fill to handle edge cases
result = result.ffill().bfill()
```

**Result**: Complete hourly DataFrame with:
- Temperature, wind, cloud cover at original KNMI timestamps (no shift)
- Radiation, precipitation at interpolated hourly values (shifted + interpolated)

---

## Testing Strategy

### Unit Tests

**File**: `tests/unit/test_knmi_client.py`

#### Test 1: Time-Shift Verification (`test_cumulative_field_time_shift`)

Verifies that:
- Cumulative fields (Q, RH) are shifted by -30 minutes
- Instantaneous fields (T, N) are NOT shifted
- Both field types coexist correctly in final DataFrame

#### Test 2: Interpolation Validation (`test_interpolation_creates_hourly_series`)

Verifies that:
- Sparse cumulative data (e.g., 3 points) generates full hourly series
- Interpolated values exist between original sparse points
- Final output has hourly frequency

#### Test 3: Radiation Conversion (`test_radiation_conversion`)

Verifies that:
- J/cm²/h → W/m² conversion is correct
- Time-shift + interpolation preserves value ranges
- Interpolated values are physically reasonable

### Integration Testing

**Verification**: Compare KNMI-derived radiation with Meteosource forecasts to ensure temporal alignment of solar peak.

---

## Impact on Forecasting

### Before Time-Shift Correction

```
Market Hour  | Actual Radiation Peak | KNMI Timestamp (Wrong) | Feature Value
-------------|----------------------|------------------------|---------------
11:00-12:00  | Peak at 11:30        | Stored at 12:00        | Uses 12:00 value
12:00-13:00  | Declining            | Stored at 13:00        | Uses 13:00 value
```

**Problem**: Radiation feature lags behind reality by 30 minutes, causing model to learn incorrect relationships.

### After Time-Shift Correction

```
Market Hour  | Actual Radiation Peak | KNMI Timestamp (Corrected) | Feature Value
-------------|----------------------|----------------------------|---------------
11:00-12:00  | Peak at 11:30        | Interpolated at 11:00      | Uses 11:00 value
12:00-13:00  | Declining            | Interpolated at 12:00      | Uses 12:00 value
```

**Benefit**: Radiation feature correctly represents the midpoint of each market hour, improving model accuracy for solar-dependent generation forecasts.

---

## Alternative Approaches Considered

### 1. No Correction (Original Implementation)

**Pros**: Simple, no data manipulation  
**Cons**: Systematic 30-minute temporal misalignment, especially problematic for solar radiation

### 2. Shift by Full Hour

**Pros**: Aligns cumulative values to START of interval  
**Cons**: Doesn't represent the center of measurement period; even less accurate than no shift

### 3. Use Shifted Values Without Interpolation

**Pros**: No assumptions about intermediate values  
**Cons**: Leaves data at :30 timestamps, incompatible with hourly market data grid

### 4. Drop Cumulative Fields Entirely

**Pros**: Eliminates complexity  
**Cons**: Loses valuable solar radiation data, which is critical for renewable energy forecasting

### Selected Approach: 30-Minute Shift + Linear Interpolation

**Rationale**:
- **30-minute shift**: Places values at interval center (physically most representative)
- **Linear interpolation**: Creates hourly grid with minimal assumptions
- **Preserves data**: Retains solar radiation information critical for electricity generation forecasting
- **Simple**: Easy to understand, test, and maintain

---

## Maintenance and Future Considerations

### Monitoring

- **Data Quality**: Check for NaN values after interpolation (should be 0 with current implementation)
- **Value Ranges**: Validate radiation never exceeds solar constant (≈1367 W/m² at top of atmosphere, ≈1000 W/m² at surface)
- **Temporal Alignment**: Compare KNMI radiation peaks with Meteosource forecasts to detect drift

### Potential Enhancements

1. **15-Minute Interpolation**: If market data moves to 15-min resolution, modify to `resample("15min")`
2. **Non-Linear Interpolation**: Consider cubic spline for smoother transitions (may overfit with hourly data)
3. **Multiple Stations**: Average data from multiple KNMI stations (240=Schiphol, 260=De Bilt) to reduce measurement noise
4. **Validation Against Satellite Data**: Compare interpolated radiation with satellite-derived solar irradiance

### Breaking Changes to Avoid

- **Changing shift magnitude**: 30 minutes is meteorologically justified; other values would require revalidation
- **Removing interpolation**: Would break hourly grid alignment with market data
- **Switching to different interpolation**: Could affect model training if historical features change

---

## References

- **KNMI Data Documentation**: https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script
- **KNMI Station Metadata**: https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/uurgegevens/uurstations.txt
- **Meteorological Measurement Standards**: WMO (World Meteorological Organization) guidelines on hourly observations

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2025-11-21 | GitHub Copilot | Initial implementation of time-shift and interpolation for cumulative KNMI fields |
| 2025-11-21 | GitHub Copilot | Added comprehensive documentation and unit tests |

---

## Summary

The KNMI time-shift implementation corrects a fundamental temporal misalignment in cumulative meteorological measurements (radiation, precipitation) by:

1. **Classifying fields** into cumulative vs instantaneous based on KNMI documentation
2. **Shifting cumulative timestamps** backward by 30 minutes to represent interval centers
3. **Interpolating to hourly grid** to align with market data resolution
4. **Preserving instantaneous fields** at their original timestamps

This ensures weather features accurately represent the time periods they describe, improving forecast model accuracy for solar-dependent electricity generation.

**Key Takeaway**: Meteorological cumulative data requires careful temporal handling—timestamps represent when measurements ENDED, not when phenomena OCCURRED. The 30-minute shift corrects this systematic bias.
