# Architecture and Design Overview

## 1. Purpose and Scope

This application forecasts electricity market prices for the Netherlands using machine learning models trained on historical market data and weather features. It addresses three main problems:

1. **Price Forecasting**: Predicts day-ahead, imbalance shortage/surplus prices up to 36 hours ahead
2. **Model Optimization**: Automated hyperparameter search across model × market combinations
3. **Data Pipeline Management**: Automated fetching, validation, and storage of market and weather data

**Out of Scope**: Real-time trading decisions, financial advice, non-Netherlands markets, sub-15-minute resolution forecasting.

---

## 2. High-Level Architecture

### Components

- **Data Layer**: CSV storage (`data/`), external API clients (ENTSO-E, TenneT, KNMI, Meteosource)
- **Services Layer**: Data validation, feature engineering, forecast orchestration, hyperparameter tuning
- **Model Layer**: Persistence, Linear Regression, Random Forest, XGBoost, HistGradientBoosting
- **API Layer**: FastAPI REST endpoints with automatic data refresh scheduling
- **UI Layer**: Streamlit interactive web application
- **Artifacts**: Hyperparameter search results stored in `hyperparameters/`

### Data Flow

```
External APIs → Raw Data → Validation → CSV Storage
                                           ↓
                                    Feature Engineering
                                    (time + lag + weather)
                                           ↓
                                    Model Training ←─ Hyperparameters
                                           ↓
                                    Cached Models → Predictions → API/UI
```

**Typical Request Flow**:
1. User requests forecast via API/UI
2. System checks data freshness and updates if needed (< 1 hour old)
3. Service loads historical market + weather data from CSV
4. Feature engineering creates time features, lags, rolling stats
5. Model trains (or loads from cache) with optional hyperparameters
6. Future features built using weather forecasts
7. Model predicts prices/states for forecast horizon
8. **Evaluation metrics calculated** (if actual data overlaps with forecast period):
   - **Regression markets** (prices): RMSE (Root Mean Squared Error)
   - **Classification markets** (regulation_state): Accuracy, F1-Macro, Confusion Matrix

---

## 3. Data Sources and Pipelines

### Data Sources

| Source | Type | Resolution | Storage | Update Frequency |
|--------|------|------------|---------|------------------|
| **ENTSO-E Day-Ahead** | Market prices | 15-min | `entsoe_day_ahead_prices_2025.csv` | Daily via scheduler |
| **TenneT Imbalance** | Settlement prices | 15-min | `imbalance_unified.csv` | On-demand (15-min freshness) |
| **KNMI Weather** | Historical observations | 1-hour | `weather_data_2025.csv` | Daily via scheduler |
| **Meteosource Weather** | Forecasts | 1-hour | In-memory only | On-demand for predictions |

### Pipeline Details

**Automatic Data Refresh** (`app/services/data_refresh_service.py`):
- **Trigger**: Every forecast request (API or UI)
- **Freshness Threshold**: 1 hour (configurable)
- **Behavior**: Checks latest timestamp in each dataset; if age > 1 hour, fetches and appends new data
- **Idempotent**: Safe to call frequently; returns quickly if data is already fresh
- **Update Strategy**: Always appends to existing CSV files with deduplication and sorting

**Day-Ahead Prices** (`app/services/entsoe_client.py`):
- Fetches from ENTSO-E Transparency Platform
- Default range: Oct 1, 2024 → present
- Document type: A44 (day-ahead prices)
- Columns: `timestamp_utc`, `price_eur_per_mwh`, `market_type`

**Imbalance Data** (`app/services/settlement_price_service.py`):
- Hybrid: Local CSV + TenneT API fallback
- Freshness threshold: 15 minutes
- Columns: `timestamp_utc`, `shortage_price`, `surplus_price`, `regulation_state`
- States: UP (1), DOWN (-1), UP_AND_DOWN (2), BALANCED (0)
- **Data Quality**: ~3% of raw data contains missing values; forward-fill imputation applied during loading to ensure continuity
- **Imputation Strategy**: Forward-fill (last known value) + back-fill (for start-of-dataset gaps)
  - See `IMBALANCE_MISSING_DATA_FIX.md` for details

**Weather Data**:
- Historical: KNMI De Bilt station (260)
- Forecast: Meteosource Amsterdam (52.3676, 4.9041)
- Columns: `temperature_deg_c`, `wind_speed_m_per_s`, `global_radiation_w_per_m2`, `cloud_cover_pct`, `precipitation_mm`
- Note: Weather is hourly, forward-filled to match 15-min market data

### Time Index and Timezone Conventions

**Canonical Timezone**: All data stored in **UTC**. Local display uses `Europe/Amsterdam`.

**Validation** (`app/services/data_validation.py`):
- All timestamps must be timezone-aware (no naive datetimes)
- Start/end dates validated across data sources with configurable tolerance
- Functions: `validate_timezones()`, `validate_start_end_dates()`, `validate_and_align_market_weather_data()`

**Lag Features**:
- Default lags: [1, 2, 24, 168] periods (actual hours depends on data resolution)
- Calculated via `df.shift()` assuming chronologically sorted data
- Rolling windows: 24-hour mean/std using `rolling(window=24)`

---

## 4. Domain Assumptions and Design Choices

### Key Assumptions

1. **Data Resolution**: Market data is 15-minute intervals; weather is hourly (forward-filled)
   - *Rationale*: ENTSO-E provides 15-min data; KNMI only provides hourly
   
2. **Missing Data**: Dropped after feature engineering (NaN from lags/rolling stats)
   - *Rationale*: Most ML models can't handle NaN; early data periods naturally lack lag values
   
3. **DST Handling**: UTC storage eliminates DST ambiguity; local timezone conversion only for display
   - *Rationale*: Market operations use UTC; user interface shows local time

4. **Market Resolution Mapping**:
   - Day-ahead: 15-min → 96 intervals/day
   - Imbalance: 15-min → 96 intervals/day
   - Intraday: 15-min → 96 intervals/day

5. **Training/Test Split**: Time-based (no shuffling) to prevent data leakage
   - *Rationale*: Financial time series require chronological integrity

6. **Model Caching**: In-memory only, keyed by (market_type, model_type, training_config)
   - *Rationale*: Retraining is fast (<10s); persistence across restarts not critical

### Known Limitations

- **Weather Forecast Horizon**: Meteosource free tier = 24 hours (limits forecast accuracy beyond this)
- **ENTSO-E API Rate Limits**: No built-in rate limiting; may fail under heavy concurrent requests
- **Imbalance Historical Data**: ENTSO-E API unreliable for Q4 2024; using local CSV files instead
- **KNMI Time Resolution**: Hourly resolution only (interpolated to hourly grid after 30-min time-shift for cumulative fields)
- **No Database**: CSV files are single-threaded; concurrent writes could corrupt data

---

## 5. Models and Hyperparameter Search

### Supported Models

| Model | Type | Use Case | Hyperparameters | Training Time |
|-------|------|----------|-----------------|---------------|
| **Persistence** | Baseline | Naive forecast (last value or mean) | `method: last/mean` | <1s |
| **Linear Regression** | Regression | Linear relationships | `fit_intercept` | 1-2s |
| **Random Forest** | Regression | Non-linear, robust | `n_estimators`, `max_depth`, `min_samples_split`, etc. | 5-10s |
| **HistGradientBoosting** | Regression | Fast gradient boosting | `learning_rate`, `max_depth`, `max_iter`, etc. | 3-5s |
| **XGBoost Classifier** | Classification | Regulation state prediction | `n_estimators`, `max_depth`, `learning_rate`, etc. | 3-5s |

### Model Factory

Models instantiated via `ForecastService._get_model_instance()`:
- Loads tuned hyperparameters from `hyperparameters/{market_type}/{model_type}_*.json` if available
- Falls back to defaults if no tuned params found
- User can override with custom hyperparameters

### Hyperparameter Search Pipeline

**Entry Point**: `scripts/run_full_hyperparam_search.py`

**Process**:
1. Get all valid model × market combinations via `get_model_market_combinations()`
2. For each combination:
   - Load and prepare training data
   - Define parameter grid via `get_param_grid(model_type)`
   - Run `RandomizedSearchCV` with `TimeSeriesSplit` (5-fold CV)
   - Save results to `hyperparameters/{market_type}/{model_type}_{timestamp}.json`
3. Aggregate results into `hyperparameters/summary.csv`

**Key Functions** (`app/services/hyperparameter_service.py`):
- `get_param_grid()`: Returns param distributions for RandomizedSearchCV
- `run_hyperparameter_search()`: Executes search with cross-validation
- `save_search_results_with_trials()`: Persists best params + all CV scores
- `load_best_params()`: Retrieves tuned params for inference

**Output Structure**:
```
hyperparameters/
├── day_ahead/
│   ├── linear_regression_20251120_110925.json
│   ├── random_forest_20251120_111234.json
│   └── hist_gradient_boosting_20251120_112045.json
├── imbalance_shortage/
│   └── ...
├── summary.csv  # Consolidated results across all searches
└── hyperparams_defaults.json  # Fallback defaults
```

---

## 6. External Integrations (APIs)

### ENTSO-E Transparency Platform

**Module**: `app/services/entsoe_client.py`  
**Endpoints**: 
- Day-ahead prices: Document type A44
- Imbalance prices: Document types A85 (system price), A86 (positive/negative imbalance)

**Configuration**:
- API key: `ENTSOE_API_KEY` environment variable
- Base URL: `https://web-api.tp.entsoe.eu/api`
- Area code: `10YNL----------L` (Netherlands)

**Error Handling**: Returns empty DataFrame on API errors (logged but doesn't crash)

### TenneT Settlement Prices API

**Module**: `app/services/tennet_client.py`  
**Purpose**: Real-time imbalance price updates  
**Freshness Policy**: Only call if local data >15 minutes old (`SettlementPriceService.FRESHNESS_THRESHOLD`)  
**Fallback**: Uses local CSV if API fails

**Configuration**:
- API key: `TENNET_API_KEY` (optional)
- Base URL: `https://api.tennet.eu/settlement-prices/v1`
- Timeout: 10 seconds

### KNMI Weather Data

**Module**: `app/services/knmi_historical_client.py`  
**Endpoint**: `https://www.daggegevens.knmi.nl/klimatologie/uurgegevens`  
**Station**: De Bilt (260) or Schiphol (240)  
**No API Key Required**: Public open data

**Data Processing and Time-Shift Correction**:

KNMI hourly data requires special handling for **cumulative measurements**:

**Cumulative vs Instantaneous Fields**:
- **Cumulative** (measured over PREVIOUS hour): `Q` (global radiation), `RH` (precipitation duration), `DR` (precipitation amount), `SQ` (sunshine duration)
- **Instantaneous/Averaged** (measured AT timestamp): `T` (temperature), `FH` (wind speed), `N` (cloud cover), `U` (humidity), `P` (pressure)

**Time-Shift Rationale**:
- KNMI timestamps with hour=HH represent the END of a measurement interval
- For cumulative variables, timestamp HH:00 contains the total measured from (HH-1):00 to HH:00
- Example: Timestamp 13:00 contains solar radiation energy accumulated between 12:00 and 13:00
- **Problem**: The timestamp doesn't represent WHEN the energy was collected (the previous hour), but when the measurement completed
- **Solution**: Shift cumulative field timestamps backward by 30 minutes to place the value at the CENTER of the measurement interval

**Implementation** (`convert_to_standard_format`):
1. **Separate fields**: Split DataFrame into cumulative vs instantaneous fields
2. **Time-shift cumulative**: Subtract 30 minutes from timestamps: `df.index - pd.Timedelta(minutes=30)`
3. **Interpolation**: Create hourly grid from shifted data:
   - Original shifted times (e.g., 00:30, 01:30, ...) are preserved
   - Hourly grid points (00:00, 01:00, ...) are created via `pd.date_range()`
   - Combined index includes both shifted values and hourly grid
   - Linear interpolation fills hourly grid points
   - Final DataFrame contains only hourly values (00:00, 01:00, 02:00, ...)
4. **Merge**: Join instantaneous fields (no shift) with interpolated cumulative fields via outer join
5. **Fill gaps**: Forward-fill and back-fill to ensure complete coverage

**Unit Conversions**:
- Temperature: 0.1°C → °C (`value / 10`)
- Radiation: J/cm²/h → W/m² (`value * 10000 / 3600`)
- Cloud cover: oktas (0-9) → % (`value * 100 / 8`, 9=invisible→NaN)
- Precipitation: 0.1 mm → mm (`value / 10`, -1→0 for trace amounts)
- Wind speed: 0.1 m/s → m/s (`value / 10`)

**Hour 24 Handling**: KNMI uses hours 1-24; hour 24 converted to midnight of next day

**Assumptions**:
- 30-minute shift places cumulative values at interval center (reasonable for physics: average radiation/precipitation occurs at midpoint)
- Linear interpolation between hourly shifted values creates smooth time series suitable for ML features
- Forward-fill for non-cumulative fields preserves last known temperature/pressure/humidity between hourly readings

**Testing**: See `tests/unit/test_knmi_client.py` for verification of time-shift, interpolation, and unit conversions

### Meteosource Weather Forecasts

**Module**: `app/services/meteosource_client.py`  
**Purpose**: Future weather for forecast horizon  
**Limitation**: Free tier = 24-hour forecast only

**Configuration**:
- API key: `METEOSOURCE_API_KEY` environment variable
- Coordinates: (52.3676, 4.9041) Amsterdam

---

## 7. Code Structure and Key Modules

### Directory Layout

```
app/
├── core/
│   ├── config.py              # Settings (Pydantic), market/model type definitions
│   └── logging.py             # Centralized logger setup
├── models/                    # ML model implementations (all inherit from BaseModel)
│   ├── persistence_model.py
│   ├── linear_regression_model.py
│   ├── random_forest_model.py
│   ├── hist_gradient_boosting_model.py
│   └── xgboost_classifier.py
├── services/
│   ├── data_store.py          # CSV I/O with deduplication and sorting
│   ├── data_validation.py     # Timezone and date range validation
│   ├── feature_engineering.py # Time features, lags, rolling stats, weather merge
│   ├── forecast_service.py    # Model training orchestration and prediction
│   ├── hyperparameter_service.py # Param grids, search execution, results storage
│   ├── entsoe_client.py       # ENTSO-E API wrapper
│   ├── tennet_client.py       # TenneT API wrapper
│   ├── knmi_historical_client.py # KNMI weather fetcher
│   ├── meteosource_client.py  # Meteosource forecast fetcher
│   ├── settlement_price_service.py # Hybrid local+API imbalance prices
│   ├── weather_data_manager.py # Weather forecast management
│   ├── data_update_service.py  # Daily update scheduler
│   └── entsoe_data_updater.py  # Market data update logic
└── api/
    ├── main.py                # FastAPI app factory with lifespan management
    └── forecast.py            # REST endpoints (/forecast, /markets, /models, etc.)

scripts/
└── run_full_hyperparam_search.py  # Hyperparameter search orchestrator

tests/
├── unit/                      # Component-level tests
├── services/                  # Service integration tests
└── api/                       # API endpoint tests
```

### Key Entry Points

**FastAPI Server**:
```bash
uvicorn app.api.main:create_app --reload
# or via Python
python -c "from app.api.main import create_app; import uvicorn; uvicorn.run(create_app())"
```

**Streamlit UI**:
```bash
streamlit run streamlit_app.py
```

**Hyperparameter Search**:
```bash
python -m scripts.run_full_hyperparam_search --markets day_ahead --models random_forest --n-iter 50
```

**Data Updates** (manual):
```bash
python -m app.services.data_update_service  # CLI mode (if __name__ == "__main__")
```

---

## 8. Visualization and Evaluation

### Forecast Visualization

**Streamlit UI** (`streamlit_app.py`, `plot_forecast_results()` function):

**Regression Markets** (day_ahead, imbalance prices):
- **Chart Type**: Line graph with markers
- **Layers**: 
  - Training data (dotted line, light blue)
  - Historical data (solid line, gray)
  - Forecast predictions (solid line with markers, blue/green/red for model comparison)
  - Weather overlay (secondary y-axis): temperature, wind, solar radiation, clouds, precipitation
- **Metrics Displayed**: Mean/Min/Max price, RMSE (if overlap with actual data)

**Classification Market** (regulation_state):
- **Chart Type**: **Vertical bar chart** (categorical states)
- **Color Mapping**:
  - Red (`#E74C3C`): Deficit state (-1)
  - Gray (`#95A5A6`): Balanced state (0)
  - Blue (`#3498DB`): Light surplus (+1)
  - Green (`#2ECC71`): Strong surplus (+2)
- **Y-Axis**: Discrete categorical ticks with labels ("Deficit", "Balanced", etc.)
- **Metrics Displayed**: Most common state, Accuracy, F1-Macro, Confusion Matrix

### Evaluation Metrics

**Regression Evaluation** (`ForecastService.calculate_forecast_rmse()`):
- **Metric**: RMSE (Root Mean Squared Error)
- **Units**: EUR/MWh
- **Calculation**: Merges forecast with actual market data on timestamp, computes `sqrt(mean((y_pred - y_true)²))`
- **Use Cases**: Day-ahead prices, imbalance shortage/surplus prices

**Classification Evaluation** (`ForecastService.evaluate_classification_forecast()`):
- **Metrics**:
  - **Accuracy**: Percentage of correct predictions
  - **F1-Macro**: Macro-averaged F1 score (balanced across all regulation states)
  - **Confusion Matrix**: 4×4 matrix showing actual vs predicted state counts
- **Use Case**: Regulation state predictions
- **Libraries**: `sklearn.metrics.accuracy_score`, `f1_score`, `confusion_matrix`

**API Responses**:
- Forecast endpoints return timestamps and predictions
- Evaluation metrics computed on-demand in UI layer
- Future enhancement: Include metrics directly in API responses

---

## 9. Testing and Validation

### Test Organization

- **Total**: 432 tests (100% passing)
- **Coverage**: 75.54% overall

**Structure**:
```
tests/
├── unit/                      # 367 tests - isolated component testing
│   ├── test_models.py         # Model fit/predict/feature_importance
│   ├── test_feature_engineering.py  # Time features, lags, weather merge
│   ├── test_data_validation.py  # Timezone and date range checks
│   ├── test_hyperparameter_search.py  # Param grids, search execution
│   └── ...
├── services/                  # 40 tests - service integration
│   ├── test_data_update_service.py  # Initialization, updates, scheduler
│   ├── test_data_refresh_service.py  # Freshness checks, append-only updates
│   └── test_imbalance_data_loader.py  # CSV combining, deduplication
└── api/                       # 25 tests - endpoint testing
    ├── test_main.py           # App lifecycle, CORS, startup/shutdown
    └── test_forecast_api.py   # Forecast generation, validation, error handling, data refresh
```

### Coverage Highlights

- **Well-Covered** (>85%): `config.py`, `main.py`, `data_validation.py`, `imbalance_data_loader.py`
- **Moderate** (70-85%): Most services and models
- **Lower** (<70%): `entsoe_data_updater.py` (40%), `weather_data_manager.py` (46%), `xgboost_classifier.py` (55%)

### Data Validation Checks

All enforced in `app/services/data_validation.py`:

1. **Timezone Awareness**: No naive datetimes allowed; all UTC
2. **Date Range Alignment**: Market and weather data must overlap (configurable tolerance)
3. **Schema Validation**: Required columns checked before processing
4. **Duplicate Handling**: Removed during CSV save/append operations
5. **Chronological Ordering**: Data sorted by timestamp before feature engineering

---

## 9. How to Extend the System

### Recipe 1: Add a New Market Type

**Files to Modify**:
1. `app/core/config.py`: Add entry to `MARKET_TYPES` dict
   ```python
   "intraday": {
       "display_name": "Intraday Market",
       "entsoe_document_type": "A70",  # Check ENTSO-E docs
       "data_file": "entsoe_intraday_prices.csv",
       "target_column": "price_eur_per_mwh",
       "target_type": "regression",
   }
   ```

2. `app/services/entsoe_client.py`: Add fetch method (copy `fetch_day_ahead_prices_2025_nl` and modify document type)

3. `app/services/data_update_service.py`: Add to `initialize_historical_data_2025()` and `update_latest_data()`

4. Create data file: `data/entsoe_intraday_prices.csv` (or let initialization create it)

5. **Optional**: Add hyperparameter search configs in `hyperparameters/intraday/`

### Recipe 2: Add a New Model Type

**Files to Modify**:
1. `app/models/your_model.py`: Create class inheriting from `BaseModel`
   ```python
   from app.models.base import BaseModel
   
   class YourModel(BaseModel):
       def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
           # Training logic
       
       def predict(self, X: pd.DataFrame) -> np.ndarray:
           # Prediction logic
   ```

2. `app/core/config.py`: Add to `MODEL_TYPES` dict
   ```python
   MODEL_TYPES["your_model"] = {
       "display_name": "Your Model",
       "supports_regression": True,
       "supports_classification": False,
   }
   ```

3. `app/services/forecast_service.py`: Add case in `_get_model_instance()`
   ```python
   elif model_type == "your_model":
       return YourModel(**params if params else {})
   ```

4. `app/services/hyperparameter_service.py`: Add param grid in `get_param_grid()`
   ```python
   elif model_type == "your_model":
       return {
           "param1": randint(1, 100),
           "param2": uniform(0, 1),
       }
   ```

5. **Optional**: Add unit tests in `tests/unit/test_your_model.py`

### Recipe 3: Add a New External Data Source

**Files to Create**:
1. `app/services/your_data_client.py`:
   ```python
   class YourDataClient:
       def __init__(self, api_key: str = None):
           self.api_key = api_key or settings.your_api_key
       
       def fetch_data(self, start: datetime, end: datetime) -> pd.DataFrame:
           # API call logic
           # Return DataFrame with timestamp_utc column
   ```

2. Add to `app/core/config.py`:
   ```python
   class Settings(BaseSettings):
       your_api_key: str = ""
       your_base_url: str = "https://api.example.com"
   ```

3. Add to `.env.example`:
   ```
   YOUR_API_KEY=your_key_here
   ```

4. Integrate in `app/services/data_update_service.py`:
   - Add to `initialize_historical_data_2025()`
   - Add to `update_latest_data()`

5. Add tests in `tests/unit/test_your_data_client.py`

---

## 10. Open Questions and TODOs

### Known Technical Debt

1. **Model Persistence**: Models are cached in-memory only
   - *TODO*: Implement pickle/joblib serialization to disk
   - *TODO*: Add model versioning and metadata tracking

2. **CSV Concurrency**: No file locking for concurrent writes
   - *TODO*: Migrate to PostgreSQL/TimescaleDB for production
   - *TODO*: Add write locks or queue-based file access

3. **API Rate Limiting**: No built-in rate limiting for ENTSO-E/Meteosource
   - *TODO*: Implement exponential backoff and request throttling
   - *TODO*: Add request caching layer

4. **Error Recovery**: Partial failures in data updates may leave inconsistent state
   - *TODO*: Implement transactional updates with rollback
   - *TODO*: Add data integrity checks after updates

### Planned Improvements

1. **Advanced Models**: LSTM, Transformer architectures for time series
2. **Feature Store**: Centralized feature computation and caching
3. **Model Monitoring**: Track prediction drift, retrain triggers
4. **Automated Scheduling**: Production-grade job scheduler (replace APScheduler with Airflow/Prefect)
5. **API Authentication**: JWT tokens, API keys, rate limiting per user
6. **Containerization**: Docker setup for reproducible deployments
7. **CI/CD Pipeline**: GitHub Actions for test/deploy automation (partially implemented)

### Unclear Design Decisions

1. **Hyperparameter Storage Format**: JSON files vs database vs MLflow
   - *Current*: JSON files with timestamps
   - *Question*: Should we adopt MLflow for experiment tracking?

2. **Weather Forecast Age**: When to re-fetch Meteosource forecasts?
   - *Current*: On-demand for each prediction
   - *Question*: Should we cache forecasts for N minutes?

3. **Training Data Window**: Fixed vs rolling window
   - *Current*: All available historical data
   - *Question*: Should we limit to last N months for concept drift?

4. **Model Selection Strategy**: Manual vs automatic based on RMSE
   - *Current*: User selects model
   - *Question*: Auto-select best model per market type?

---

**Document Version**: 1.0  
**Last Updated**: November 20, 2025  
**Maintainer**: GitHub Copilot
