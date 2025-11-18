# Quick Start Guide

## Two Ways to Run

### 🧪 Demo Mode (Recommended for Quick Start)
Run immediately with sample data - **no API keys needed!**

### 🚀 Production Mode
Connect to real ENTSO-E and KNMI APIs for live data

---

## Demo Mode Setup (5 minutes)

### 1. Install Dependencies

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install project
pip install -e .
```

### 2. Enable Demo Mode

The `.env` file already has demo mode enabled:
```
DEMO_MODE=true
```

### 3. Run the App

```powershell
streamlit run streamlit_app.py
```

✅ **That's it!** Open http://localhost:8501 and start forecasting with sample data.

**Demo mode includes:**
- 168 hours (7 days) of sample market data for all three markets
- Pre-configured weather data
- All three forecasting models working
- No external API dependencies

---

## Production Mode Setup

### Prerequisites

1. **Python 3.11+** installed
2. **API Keys**:
   - ENTSO-E Transparency Platform: https://transparency.entsoe.eu/
   - KNMI Open Data Platform: https://developer.dataplatform.knmi.nl/

## Installation Steps

### 1. Setup Environment

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Add API Keys

Edit `.env` file:

```bash
# Disable demo mode
DEMO_MODE=false

# Add your API keys
ENTSOE_API_KEY=your_entsoe_api_key_here
KNMI_API_KEY=your_knmi_api_key_here
```

### 3. Initialize Historical Data

Download 2025 data from APIs:

```powershell
python -m app.services.data_update_service --init
```

This will take several minutes to fetch all market and weather data.

### 4. Run the Application

**Streamlit Frontend (Recommended)**

```powershell
streamlit run streamlit_app.py
```

**FastAPI Backend (Optional)**

```powershell
uvicorn app.api.main:app --reload --port 8000
```

Visit: http://localhost:8000/docs for API documentation

---

## Current Functionalities

### ✅ Data Integration
- **ENTSO-E Transparency Platform**: Day-ahead, intraday, and imbalance prices for Netherlands
- **KNMI Weather Data**: Temperature, wind speed, and solar radiation
- **Automated Updates**: Daily scheduler fetches latest data at midnight (Amsterdam time)
- **CSV Storage**: All data persisted locally for offline access

### ✅ Machine Learning Models
Three forecasting models with feature engineering:

1. **Persistence Model**: Simple baseline (future = recent past)
2. **Linear Regression**: Linear model with lag and weather features  
3. **Random Forest**: Ensemble tree-based model with tunable hyperparameters

**Features used:**
- Time features: Hour of day, day of week, month (with cyclic encoding)
- Lag features: 1, 2, 3, 6, 12, 24 hours (demo) or 1, 2, 3, 24, 48, 168 hours (production)
- Rolling statistics: 24-hour mean and standard deviation
- Weather features: Temperature, wind speed, solar radiation

### ✅ Interactive Web Interface (Streamlit)
- Market selection: Day-ahead, intraday, or imbalance
- Model comparison: Test all three models side-by-side
- Adjustable forecast horizon: 1-36 hours ahead
- Historical data overlay: View past trends alongside forecasts
- Feature importance visualization: See which features drive predictions
- Data summary dashboard: Monitor available data coverage

### ✅ REST API (FastAPI)
- `/forecast`: Generate forecasts programmatically
- `/health`: System health check
- `/docs`: Interactive API documentation (Swagger UI)
- Auto-generated client SDKs
- CORS support for web integration

### ✅ Comprehensive Testing
- **165 passing tests** covering all components
- Unit tests for models, clients, data store, feature engineering
- Mock external APIs for reliable testing
- Fast execution (~5 seconds for full suite)
- Coverage reports available with pytest-cov

### ✅ Demo Mode
- Works without API keys or external dependencies
- Uses 168 hours of sample data
- Reduced lag features optimized for small datasets
- Perfect for learning and experimentation

---

## Quick Usage Examples

### Check Data Summary

```powershell
python -m app.services.data_update_service --summary
```

### Manual Data Update

```powershell
python -m app.services.data_update_service --update
```

### Run All Tests

```powershell
pytest
```

### Run Tests with Coverage

```powershell
pytest --cov=app --cov-report=html
```

### Test Demo Mode Forecast

```powershell
python test_demo.py
```

---

## Troubleshooting

### Issue: Models fail to train in demo mode

**Solution:** Ensure `DEMO_MODE=true` in `.env` file. Demo mode uses reduced lag features that work with 7 days of data.

### Issue: "No data available" in production mode

**Solution:** Initialize data first:
```powershell
python -m app.services.data_update_service --init
```

### Issue: "API key not found"

**Solution:** Check that `.env` file exists with valid API keys:
```powershell
cat .env
```

### Issue: Import errors

**Solution:** Reinstall in editable mode:
```powershell
pip install -e .
```

### Issue: Timezone comparison errors

**Solution:** This is fixed in the latest version. Ensure you have the updated `streamlit_app.py`.

### Issue: KNMI data download fails

**Solution:** The KNMI dataset configuration may need adjustment. Check:
- Dataset name in `app/core/config.py`
- API key validity
- Network connectivity

---

## Project Architecture

```
app/
├── api/
│   ├── main.py              # FastAPI app with scheduler
│   └── forecast.py          # Forecast endpoints
├── core/
│   ├── config.py            # Settings and configuration
│   └── logging.py           # Logging setup
├── models/
│   ├── persistence_model.py # Baseline model
│   ├── linear_regression_model.py
│   └── random_forest_model.py
├── services/
│   ├── entsoe_client.py     # ENTSO-E API client
│   ├── knmi_client.py       # KNMI API client
│   ├── data_store.py        # CSV persistence
│   ├── data_update_service.py # Scheduled data updates
│   ├── feature_engineering.py # Feature creation
│   └── forecast_service.py  # Model training & prediction

tests/
├── unit/                    # 146 unit tests
│   ├── test_config.py
│   ├── test_entsoe_client.py
│   ├── test_knmi_client.py
│   ├── test_data_store.py
│   ├── test_feature_engineering.py
│   └── test_models.py
├── integration/             # Integration tests (TODO)
└── conftest.py             # Shared test fixtures

streamlit_app.py            # Interactive web UI
data/                       # CSV data files
.env                        # Configuration (not in git)
```

---

## Key Implementation Details

### Demo Mode vs Production Mode

| Feature | Demo Mode | Production Mode |
|---------|-----------|-----------------|
| API Keys Required | ❌ No | ✅ Yes |
| Data Source | Sample CSV files | Live ENTSO-E/KNMI APIs |
| Lag Features | [1,2,3,6,12,24] hours | [1,2,3,24,48,168] hours |
| Min Training Samples | 48 (2 days) | 720 (30 days) |
| Data Coverage | 168 hours (7 days) | Full 2025 year |
| Best For | Quick testing, learning | Production forecasts |

### Testing Strategy

- **Unit tests**: Test individual components in isolation with mocking
- **Fixtures**: Shared test data and mock environment setup
- **Module reloading**: Proper settings isolation between tests
- **Fast execution**: < 6 seconds for all 165 tests
- **High coverage**: All core components tested

### Model Training Process

1. Load market data from CSV
2. Create time features (hour, day, month with cyclic encoding)
3. Create lag features based on mode (demo/production)
4. Calculate rolling statistics (24h mean/std)
5. Merge weather data by timestamp
6. Remove rows with NaN values
7. Train model on features → predict price
8. Cache trained model for reuse

---

## Next Steps

### For Learning
- ✅ Run in demo mode and explore the UI
- ✅ Compare different models and markets
- ✅ Adjust forecast horizons and see how predictions change
- ✅ Check feature importance to understand model behavior

### For Development
- Add more sophisticated models (XGBoost, neural networks)
- Implement cross-validation and backtesting
- Add prediction intervals (uncertainty quantification)
- Create integration tests for end-to-end workflows
- Add model performance metrics tracking
- Implement automatic model retraining

### For Production
- Set up real API keys and initialize full dataset
- Configure automated daily updates
- Deploy to cloud (AWS, Azure, GCP)
- Add monitoring and alerting
- Implement model versioning and A/B testing

---

## Additional Resources

- **Main README**: Detailed project documentation
- **TEST_STATUS.md**: Complete testing documentation  
- **TESTING_COMPLETE.md**: Test implementation summary
- **API Docs**: http://localhost:8000/docs (when FastAPI is running)
- **ENTSO-E API Docs**: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
- **KNMI API Docs**: https://developer.dataplatform.knmi.nl/open-data-api
