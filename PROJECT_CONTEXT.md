# Energy Market Forecasting Project - Context Document

## Project Overview
This is a Python-based energy market forecasting system for the Netherlands electricity market. It predicts day-ahead prices and imbalance prices using machine learning models trained on historical market data and weather features.

**Current Date**: November 20, 2025  
**Data Coverage**: October 1, 2024 through November 18, 2025 (~13 months)

## Core Functionality
The system:
1. Fetches historical market data (day-ahead prices, imbalance prices) from ENTSO-E Transparency Platform
2. Fetches historical weather data from KNMI (Royal Netherlands Meteorological Institute)
3. Processes and stores data in CSV files
4. Engineers features (time-based, lag features, rolling statistics, weather variables)
5. Trains ML models (Persistence, Linear Regression, Random Forest, XGBoost)
6. Generates price forecasts for specified horizons (up to 48 hours)
7. Provides REST API endpoints for forecasting and model management
8. Visualizes forecasts with interactive Plotly charts

## Technology Stack
- **Language**: Python 3.11.9
- **Web Framework**: Flask 3.0+
- **ML Libraries**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly
- **Testing**: pytest (262 tests, 100% passing)
- **Environment**: Virtual environment (.venv)
- **Code Quality**: black formatter, type hints

## Project Structure

```
Market_Forecasting_example/
├── app/
│   ├── api/                      # Flask REST API endpoints
│   │   ├── __init__.py
│   │   ├── forecast.py           # Main forecast endpoints
│   │   └── routes.py             # Additional routes
│   ├── models/                   # ML model implementations
│   │   ├── __init__.py
│   │   ├── base.py               # BaseModel abstract class
│   │   ├── persistence.py        # Persistence baseline model
│   │   ├── linear_regression.py # Linear regression model
│   │   ├── random_forest.py     # Random forest model
│   │   └── xgboost_classifier.py # XGBoost for regulation state
│   ├── services/                 # Business logic and data services
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration management
│   │   ├── data_store.py         # CSV file I/O operations
│   │   ├── data_update_service.py # Orchestrates data updates
│   │   ├── entsoe_client.py      # ENTSO-E API client
│   │   ├── feature_engineering.py # Feature creation
│   │   ├── forecast_service.py   # Model training & forecasting
│   │   ├── knmi_historical_client.py # KNMI historical weather
│   │   ├── meteosource_client.py # Meteosource weather forecasts
│   │   └── plotting.py           # Plotly visualization
│   └── main.py                   # Flask app initialization
├── data/                         # CSV data storage
│   ├── entsoe_day_ahead_prices_2025.csv  # 13,474 records
│   ├── imbalance_unified.csv              # 39,740 records
│   └── weather_data_2025.csv              # 9,937 records
├── imbalance_data/               # Monthly imbalance CSVs (source)
│   ├── imbalance_2024_10.csv
│   ├── imbalance_2024_11.csv
│   └── ... (14 monthly files)
├── notebooks/                    # Jupyter notebooks
│   ├── forecast_exploration.py   # Interactive forecasting experiments
│   └── imbalance_data_loader.py  # Script to rebuild unified CSV
├── scripts/                      # Utility scripts
│   ├── fetch_2024_q4_data.py     # Fetch Q4 2024 historical data
│   └── update_data.py            # Daily data update script
├── tests/                        # Test suite (262 tests)
│   ├── integration/              # Integration tests
│   ├── unit/                     # Unit tests
│   ├── conftest.py               # pytest fixtures
│   └── test_*.py                 # Various test modules
├── .env                          # Environment variables (API keys)
├── .gitignore
├── pyproject.toml                # Project dependencies & config
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

## Data Sources & Formats

### 1. ENTSO-E Day-Ahead Prices
- **Source**: ENTSO-E Transparency Platform API
- **Resolution**: 15-minute intervals (4 per hour)
- **Columns**: `timestamp_utc`, `price_eur_per_mwh`, `market_type`
- **Records**: 13,474 (Sept 30, 2024 - Nov 19, 2025)
- **API Method**: `EntsoeClient.fetch_day_ahead_prices_2025_nl()`
- **Default Start**: `datetime(2024, 10, 1, 0, 0, 0)`

### 2. ENTSO-E Imbalance Prices
- **Source**: Local CSV files in `/imbalance_data/` (14 monthly files)
- **Resolution**: 15-minute intervals
- **Columns**: `timestamp_utc`, `shortage_price`, `surplus_price`, `regulation_state`
- **Records**: 39,740 (Oct 1, 2024 - Nov 18, 2025)
- **Regulation States**: 
  - `-1`: DOWN (14,409 records)
  - `1`: UP (14,141 records)
  - `2`: UP_AND_DOWN (10,733 records)
  - `0`: BALANCED (457 records)
- **Note**: API access for historical imbalance data has issues; use local CSV files

### 3. KNMI Historical Weather
- **Source**: KNMI API (https://www.daggegevens.knmi.nl/klimatologie/uurgegevens)
- **Resolution**: Hourly
- **Columns**: `timestamp`, `temperature_deg_c`, `wind_speed_m_per_s`, `global_radiation_w_per_m2`, `cloud_cover_pct`, `precipitation_mm`, `data_type`, `fetch_timestamp`
- **Records**: 9,937 hourly records
- **Date Range**: Oct 1, 2024 - Nov 19, 2025
- **Station**: De Bilt (260), Netherlands
- **API Method**: `KNMIHistoricalClient.fetch_historical_for_2025()`
- **Default Start**: `datetime(2024, 10, 1, 1)` (hour 1, not hour 0)
- **Data Processing**:
  - Temperature: 0.1°C resolution → convert to °C
  - Wind speed: 0.1 m/s resolution → convert to m/s
  - Radiation: J/cm²/h → W/m² conversion
  - Cloud cover: oktas (0-9) → percentage (0-100%)
  - Precipitation: 0.1mm resolution, -1 = trace (< 0.05mm)
  - Hour 24 format converted to next day hour 0

### 4. Meteosource Weather Forecasts
- **Source**: Meteosource API (for future predictions)
- **Resolution**: Hourly
- **Location**: Amsterdam coordinates (52.3676, 4.9041)
- **Used for**: Future weather data when making forecasts
- **API Method**: `MeteosourceClient.fetch_forecast_for_2025()`

## Machine Learning Models

### Available Models
1. **Persistence Model** (`persistence`)
   - Baseline: predicts last known value or mean
   - Methods: `last` (default) or `mean`
   - No training required, fast inference
   - Useful for comparison benchmark

2. **Linear Regression** (`linear_regression`)
   - Scikit-learn LinearRegression
   - Fast training, interpretable coefficients
   - Good for linear relationships
   - Provides feature importance via coefficients

3. **Random Forest** (`random_forest`)
   - Scikit-learn RandomForestRegressor
   - Default: 100 trees, max_depth=10
   - Handles non-linear relationships
   - Provides feature importance via tree splits
   - More robust to outliers

4. **XGBoost Classifier** (`xgboost_classifier`)
   - For regulation state prediction (classification)
   - Predicts: DOWN (-1), UP (1), UP_AND_DOWN (2), BALANCED (0)
   - Returns probability distributions

### Feature Engineering

**Time Features** (automatically created):
- `hour` (0-23)
- `day_of_week` (0-6, Monday=0)
- `month` (1-12)
- `is_weekend` (0 or 1)
- `hour_sin`, `hour_cos` (cyclic encoding)
- `day_of_week_sin`, `day_of_week_cos` (cyclic encoding)

**Lag Features** (configurable):
- Default lags: [1, 2, 24, 168] (1h, 2h, 1day, 1week ago)
- `price_lag_1`, `price_lag_2`, etc.
- `price_rolling_mean_24h`, `price_rolling_std_24h`

**Weather Features** (optional, configurable):
- `temperature_deg_c`
- `wind_speed_m_per_s`
- `global_radiation_w_per_m2`
- `cloud_cover_pct`
- `precipitation_mm`

**Configuration**:
```python
training_config = {
    "use_temperature": True,
    "use_wind": True,
    "use_solar": True,
    "use_cloud": True,
    "use_precipitation": True,
    "lags": [1, 2, 24, 168],
    "historical_window_hours": 168  # 7 days
}
```

## API Endpoints

### Core Endpoints (Flask REST API)

**Base URL**: `http://localhost:5000`

#### 1. Generate Forecast
```
POST /forecast
Content-Type: application/json

{
  "market_type": "day_ahead",  // or "imbalance"
  "model_type": "random_forest",  // persistence, linear_regression, random_forest
  "horizon_hours": 24,  // 1-48
  "training_config": {  // optional
    "use_temperature": true,
    "use_wind": true,
    "use_solar": true,
    "lags": [1, 2, 24, 168]
  }
}

Response:
{
  "forecast": [
    {"timestamp": "2025-11-20T01:00:00Z", "predicted_price": 45.23},
    ...
  ],
  "market_type": "day_ahead",
  "model_type": "random_forest",
  "horizon_hours": 24,
  "rmse": 12.34,  // if actual data available
  "model_info": {
    "n_samples": 5000,
    "n_features": 15,
    "training_period": "2024-10-01 to 2025-11-19"
  }
}
```

#### 2. Compare Models
```
POST /forecast/compare
Content-Type: application/json

{
  "market_type": "day_ahead",
  "model_types": ["persistence", "linear_regression", "random_forest"],
  "horizon_hours": 24
}

Response:
{
  "comparisons": [
    {
      "model_type": "persistence",
      "rmse": 25.67,
      "forecast": [...]
    },
    ...
  ]
}
```

#### 3. Get Model Info
```
GET /forecast/model-info?market_type=day_ahead&model_type=random_forest

Response:
{
  "model_type": "random_forest",
  "market_type": "day_ahead",
  "is_trained": true,
  "n_samples": 5000,
  "n_features": 15,
  "training_period": "2024-10-01 to 2025-11-19",
  "feature_importance": {
    "hour": 0.25,
    "price_lag_1": 0.20,
    ...
  }
}
```

#### 4. Visualize Forecast
```
POST /forecast/visualize
Content-Type: application/json

{
  "market_type": "day_ahead",
  "model_type": "random_forest",
  "horizon_hours": 24
}

Response:
{
  "html": "<div>... Plotly chart HTML ...</div>"
}
```

#### 5. Update Data
```
POST /data/update

Response:
{
  "status": "success",
  "updated": {
    "day_ahead": 96,  // new records
    "imbalance": 96,
    "weather": 24
  }
}
```

## Configuration (app/services/config.py)

### Environment Variables (.env file)
```
ENTSOE_API_KEY=your_entsoe_api_key
METEOSOURCE_API_KEY=your_meteosource_key
KNMI_API_KEY=  # Not required, API is open
```

### Key Configuration Values
```python
# Market Types
MARKET_TYPES = ["day_ahead", "imbalance"]

# Model Types
MODEL_TYPES = ["persistence", "linear_regression", "random_forest", "xgboost_classifier"]

# Netherlands EIC Code
NETHERLANDS_EIC = "10YNL----------L"

# Time Zone
TIMEZONE = "Europe/Amsterdam"

# Data Paths
DATA_DIR = "data/"
IMBALANCE_DATA_DIR = "imbalance_data/"

# API Configuration
MAX_FORECAST_HORIZON_HOURS = 48
DEFAULT_HORIZON_HOURS = 24
```

## Key Design Patterns

### 1. Model Interface (BaseModel)
All models inherit from `BaseModel` abstract class:
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        return None
```

### 2. Service Layer Architecture
- **Data Store**: Handles CSV file I/O with deduplication and sorting
- **API Clients**: Encapsulate external API interactions (ENTSO-E, KNMI, Meteosource)
- **Feature Engineering**: Pure functions for feature creation
- **Forecast Service**: Orchestrates model training, caching, and prediction
- **Data Update Service**: Coordinates fetching and storing new data

### 3. Model Caching
Models are cached in-memory with keys based on:
- Market type
- Model type
- Training configuration (weather features, lags, etc.)

Cache invalidation on force retrain.

### 4. Error Handling
- API errors logged but don't crash the system
- Returns empty DataFrames on API failures
- Validation at API endpoints (market type, model type, horizon)
- Pytest tests cover error cases

## Data Flow

### Training Flow
1. Load historical market data from CSV (`data_store.load_market_data()`)
2. Load weather data from CSV (`data_store.load_weather_data()`)
3. Create time features (`create_time_features()`)
4. Merge weather data (`merge_weather_data()`)
5. Create lag features and rolling stats (`create_lag_features()`)
6. Build feature matrix X and target y (`build_features_and_target()`)
7. Train model (`model.fit(X, y)`)
8. Cache trained model

### Forecasting Flow
1. Load recent historical data (for lag features)
2. Fetch weather forecast for future horizon
3. Generate future timestamps
4. Create time features for future timestamps
5. Calculate lag features from recent historical data
6. Merge weather forecast
7. Build feature matrix X
8. Predict prices (`model.predict(X)`)
9. Calculate RMSE if actual data available
10. Return forecast DataFrame

### Data Update Flow
1. Determine last timestamp in stored data
2. Fetch new data from APIs (ENTSO-E, KNMI)
3. Deduplicate and sort combined data
4. Save back to CSV files
5. Return count of new records

## Testing Strategy

### Test Coverage
- **Total Tests**: 262 (all passing)
- **Unit Tests**: ~180 (isolated component testing)
- **Integration Tests**: ~80 (end-to-end workflows)
- **Coverage Areas**:
  - API endpoint validation
  - Model training and prediction
  - Feature engineering correctness
  - Data store I/O operations
  - API client error handling
  - Visualization generation
  - Configuration validation

### Key Test Fixtures (tests/conftest.py)
```python
@pytest.fixture
def sample_timestamps():
    """15-minute resolution timestamps from 2024-10-01"""
    start = datetime(2024, 10, 1, 0, 0, 0)
    end = datetime(2025, 11, 18, 23, 45, 0)
    return pd.date_range(start, end, freq="15min")

@pytest.fixture
def sample_entsoe_xml():
    """Mock ENTSO-E API response XML"""
    # Returns XML string with test data
```

## Recent Changes (Nov 2024)

### Data Policy Update
**Changed all data start dates from 2025-01-01 to 2024-10-01** to include Q4 2024 data:

**Modified Files**:
1. `app/services/data_update_service.py` - Updated `initialize_historical_data_2025()` start_date
2. `app/services/entsoe_client.py` - Updated both fetch methods default start dates
3. `app/services/knmi_historical_client.py` - Updated `fetch_historical_for_2025()` start date
4. `notebooks/forecast_exploration.py` - Updated `train_start = "2024-10-01"`
5. `tests/conftest.py` - Updated `sample_timestamps()` fixture start date
6. `tests/unit/test_knmi_client.py` - Updated test assertions for new date range

**Data Fetched**:
- Added 2,224 day-ahead price records (Q4 2024)
- Added 2,232 weather records (Q4 2024)
- Rebuilt imbalance unified dataset from 14 monthly CSVs (39,740 total records)

**Bug Fixes Applied**:
- Fixed KNMI numeric conversion with `pd.to_numeric(errors='coerce')`
- Fixed cloud cover and precipitation handling (non-numeric values → NaN)
- Fixed weather CSV duplicate column issue from incorrect `reset_index()` usage

## Common Operations

### Run the Flask App
```bash
python app/main.py
# Server runs on http://localhost:5000
```

### Update Data (Manual)
```bash
python scripts/update_data.py
```

### Fetch Historical Data
```bash
python scripts/fetch_2024_q4_data.py
```

### Rebuild Imbalance Unified CSV
```bash
python notebooks/imbalance_data_loader.py
```

### Run Tests
```bash
pytest tests/ -v              # All tests with verbose
pytest tests/unit/ -v         # Unit tests only
pytest tests/integration/ -v  # Integration tests only
pytest tests/ -k "forecast"   # Tests matching "forecast"
```

### Format Code
```bash
python -m black app/ tests/ notebooks/ scripts/
```

## Known Issues & Limitations

1. **ENTSO-E Imbalance API**: Historical imbalance data API returns 400 errors for Q4 2024. Workaround: Use local CSV files in `/imbalance_data/`.

2. **Weather Data Hour 0**: KNMI API starts hourly data at hour 1, not hour 0. First record is `2024-10-01 01:00:00`.

3. **Data Resolutions**: Market data is 15-min, weather is hourly. Weather features are forward-filled to match market timestamps.

4. **Model Persistence**: Models are cached in-memory only. They are retrained on app restart or force retrain.

5. **API Rate Limits**: ENTSO-E and Meteosource have rate limits. The code includes basic retry logic but may fail under heavy load.

6. **Date Range Validation**: Forecasts beyond available data return warnings. System doesn't prevent requests for far-future dates (no weather forecast available).

## Dependencies (Key Libraries)

```
flask>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
plotly>=5.17.0
requests>=2.31.0
python-dotenv>=1.0.0
pytest>=7.4.0
```

## Performance Considerations

- **Training Time**: 
  - Persistence: < 1 second
  - Linear Regression: 1-2 seconds (5,000 samples)
  - Random Forest: 5-10 seconds (100 trees)
  - XGBoost: 3-5 seconds

- **Inference Time**: All models < 100ms for 48-hour forecast

- **Data Loading**: CSV reads are fast (< 1 second for all files)

- **Memory Usage**: ~200MB with all data loaded and models cached

## Future Enhancement Areas

1. **Database Integration**: Replace CSV files with PostgreSQL/TimescaleDB
2. **Model Persistence**: Save trained models to disk (pickle/joblib)
3. **Automated Scheduling**: Cron jobs for daily data updates
4. **Advanced Models**: LSTM, GRU, Transformer architectures
5. **Hyperparameter Tuning**: Grid search, Bayesian optimization
6. **Real-time Forecasts**: WebSocket updates for live predictions
7. **Monitoring**: Model performance tracking, data drift detection
8. **API Authentication**: JWT tokens, rate limiting
9. **Containerization**: Docker setup for deployment
10. **Cloud Deployment**: AWS/Azure/GCP deployment scripts

---

**Document Version**: 1.0  
**Last Updated**: November 20, 2025  
**Project Maintainer**: Market Forecasting Team
