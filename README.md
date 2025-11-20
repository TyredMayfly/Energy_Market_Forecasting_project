# Energy Market Forecasting - Power Price Forecasting for the Netherlands

A production-ready web application for forecasting power market prices in the Netherlands. This interactive Streamlit application provides real-time market forecasting by combining ENTSO-E market prices with KNMI weather data and machine learning models.

🌐 **Live Application**: [https://energymarketforecastingproject.streamlit.app/](https://energymarketforecastingproject.streamlit.app/)

## Features

### Core Forecasting Capabilities
- **Multiple Market Types**: Day-ahead, intraday, and imbalance market forecasting
- **Three ML Models**: Compare Persistence, Linear Regression, and Random Forest models
- **15-Minute Resolution**: High-granularity forecasts (96 intervals per 24-hour period)
- **Historical Window**: Adjustable historical data display (24-168 hours) for context and RMSE calculation
- **Model Performance**: Real-time RMSE scoring with data point counts

### Advanced Features
- **Granular Weather Controls**: Select specific weather features (temperature, wind speed, cloud cover, precipitation)
- **Training Data Selection**: Choose market data sources for model training
- **Interactive Visualizations**: Plotly-powered charts with historical data, forecasts, and actual prices
- **Weather Data Overlay**: Toggle weather metrics on forecast charts
- **Model Comparison**: Side-by-side comparison of all three models

### Data Integration
- **Real-time Market Data**: Live ENTSO-E day-ahead prices and imbalance data
- **Live Weather Data**: KNMI weather API extended 36 hours into the future
- **Automatic Updates**: Data refreshes automatically when stale (>12 hours for market, >60 minutes for weather)
- **15-Minute Resolution**: High-frequency data at quarter-hour intervals

## Target Audience

Data scientists, energy market analysts, students, and professionals interested in power market forecasting and machine learning applications in the energy sector.

## Tech Stack

- **Python 3.11**
- **Frontend**: Streamlit with Plotly visualizations
- **ML Framework**: scikit-learn (LinearRegression, RandomForestRegressor)
- **Data Processing**: pandas, numpy
- **Data Sources**: ENTSO-E Transparency Platform, KNMI Open Data API
- **Testing**: pytest (363 tests, 100% passing)
- **Deployment**: Streamlit Cloud

## Quick Start

### Live Application
Visit the deployed app: [https://energymarketforecastingproject.streamlit.app/](https://energymarketforecastingproject.streamlit.app/)

### Local Development

1. **Clone the repository:**
```bash
git clone https://github.com/TyredMayfly/Energy_Market_Forecasting_project.git
cd Energy_Market_Forecasting_project
```

2. **Create and activate a virtual environment:**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app:**
```bash
streamlit run streamlit_app.py
```

The app will be available at http://localhost:8501

## Usage Guide

### Generating Forecasts

1. **Select Market Type**: Choose from day-ahead, intraday, or imbalance markets
2. **Choose Model**: Pick Persistence (baseline), Linear Regression, or Random Forest
3. **Set Forecast Horizon**: 6, 12, or 24 hours ahead
4. **Adjust Historical Window**: View 24-168 hours of historical context
5. **Configure Weather Features**: Toggle individual weather parameters for training
6. **Compare Models**: Use "Compare All Models" to see performance differences

### Understanding Results

- **Price Statistics**: Mean, median, min/max prices with units in EUR/MWh
- **RMSE Score**: Model accuracy measured on overlapping historical data
- **Interactive Charts**: Zoom, pan, and hover for detailed price information
- **Weather Overlay**: View correlation between weather and prices

## Project Structure

```
/app
  /core             # Configuration, logging (Pydantic v2)
  /models           # ML model implementations (Persistence, LinearRegression, RandomForest, XGBoost)
  /services         # Data loading, feature engineering, forecast generation, auto-updates
/artifacts          # Hyperparameter search results (model tuning)
  /hyperparameter_search/
    ├── summary.csv              # Global search results
    ├── hyperparams_defaults.json  # Recommended defaults
    ├── plots/                   # Analysis visualizations
    └── <market>/<model>_*.json/csv  # Per-combination results
/data               # Market and weather data files (auto-updated)
  ├── entsoe_day_ahead_prices_2025.csv
  ├── imbalance_unified.csv
  └── weather_data_2025.csv
/tests              # 312 pytest tests (unit + integration)
  /unit             # Component-level tests
  /integration      # End-to-end feature tests
/scripts            # Utility scripts (hyperparameter search, data updates)
  ├── run_full_hyperparam_search.py  # Full hyperparameter search pipeline
  ├── analyze_hyperparams.py         # Analysis and recommendations
  └── search_hyperparameters.py      # Individual model search
/examples           # Example usage scripts
/docs               # Additional documentation
  ├── FULL_HYPERPARAM_PIPELINE.md  # Complete hyperparameter search & analysis guide
  └── HYPERPARAMETER_SEARCH.md     # Individual model hyperparameter tuning
/.streamlit         # Streamlit Cloud configuration
streamlit_app.py    # Main application entry point
requirements.txt    # Production dependencies
pyproject.toml      # Full project configuration
```

## Data Sources & Format

### ENTSO-E Transparency Platform
- **Day-Ahead Prices**: Market clearing prices (EUR/MWh) at 15-minute intervals
- **Imbalance Data**: System imbalance volumes and prices at 15-minute intervals  
- **Bidding Zone**: Netherlands (10YNL----------L)
- **Auto-Update**: Fetches latest data when >12 hours old

### KNMI Weather Data
- **Resolution**: 15-minute meteorological observations
- **Variables**: 
  - Temperature (°C)
  - Wind speed (m/s)
  - Cloud cover (oktas)
  - Precipitation (mm)
  - Global radiation (W/m²) - derived from cloud cover
- **Coverage**: Netherlands weather stations
- **Auto-Update**: Fetches forecast when >60 minutes old, extends 36 hours ahead

## Forecasting Models

### 1. Persistence Model
Baseline model that assumes future prices equal the most recent observed value. Useful for establishing performance benchmarks.

### 2. Linear Regression
Uses historical prices, time-based features (hour, day of week, weekend), and selected weather variables. Trained with sklearn's `LinearRegression`.

**Features:**
- Lag features: 1, 2, 3, 24, 48, 168 intervals (1h, 2h, 3h, 1 day, 2 days, 1 week)
- Rolling statistics: 24-interval mean and standard deviation
- Cyclic time encoding: hour and day of week
- Optional weather features based on user selection

### 3. Random Forest
Ensemble model using sklearn's `RandomForestRegressor` with 50 trees. Captures non-linear relationships between features and prices.

### 4. Histogram Gradient Boosting
Efficient gradient boosting regressor using sklearn's `HistGradientBoostingRegressor`. Uses histogram-based algorithm for faster training on large datasets while maintaining high accuracy. Native support for missing values and excellent performance on non-linear patterns.

### 5. XGBoost Classifier
Gradient boosting classifier for predicting regulation states (UP, DOWN, BALANCED, UP_AND_DOWN) in imbalance markets. Uses advanced ensemble learning.

## Hyperparameter Tuning

The project includes a comprehensive hyperparameter search pipeline for optimizing all model × market combinations:

```bash
# Search all combinations
python scripts/search_hyperparameters.py

# Search specific market
python scripts/search_hyperparameters.py --market day_ahead

# Quick search (reduced iterations)
python scripts/search_hyperparameters.py --n-iter 20 --cv-splits 3

# List available tuned models
python scripts/search_hyperparameters.py --list
```

**Features:**
- Time-series cross-validation (respects temporal ordering)
- RandomizedSearchCV with customizable iterations
- Automatic parameter grid definition per model type
- Results saved to `artifacts/hyperparameter_search/`
- Automatic loading of tuned parameters during training

For full documentation, see: [docs/HYPERPARAMETER_SEARCH.md](docs/HYPERPARAMETER_SEARCH.md)

**Parameters:**
- n_estimators: 50
- max_depth: None (fully developed trees)
- min_samples_split: 2
- random_state: 42 (reproducible results)

## Model Performance

All models include:
- **RMSE Calculation**: Automatically computed on overlapping historical data
- **Data Point Count**: Shows how many actual vs. forecast comparisons were made
- **Historical Context**: Adjustable 24-168 hour window for visual comparison
- **Cross-Model Comparison**: Side-by-side evaluation of all three models

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test suite
pytest tests/unit/
pytest tests/integration/
```

### Test Coverage

- **205 tests** covering all functionality
- **Unit tests**: Model training, feature engineering, data loading, configuration
- **Integration tests**: End-to-end forecasting, weather feature selection, model comparison
- **Test features**: 15-minute resolution, historical windows, RMSE calculation, weather toggles

## Configuration

### Environment Variables

Create a `.env` file (optional, for API integration):
```bash
ENTSOE_API_KEY=your_entsoe_key_here
KNMI_API_KEY=your_knmi_key_here
```

### Streamlit Configuration

The app uses `.streamlit/config.toml` for:
- Theme customization (blue primary color)
- Server settings (headless mode for cloud deployment)
- Browser preferences

## Deployment

The application is deployed on **Streamlit Cloud** with automatic updates:

1. **Push to GitHub**: Changes pushed to the `main` branch trigger automatic redeployment
2. **Build Process**: Streamlit Cloud installs dependencies from `requirements.txt`
3. **API Keys**: Configure `ENTSOE_API_KEY` and `METEOSOURCE_API_KEY` in Streamlit Cloud secrets
4. **Data Loading**: Market and weather data auto-update from live APIs
5. **Live URL**: [https://energymarketforecastingproject.streamlit.app/](https://energymarketforecastingproject.streamlit.app/)

### Manual Deployment

To deploy your own instance:
1. Fork this repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select repository, branch (`main`), and main file (`streamlit_app.py`)
5. Configure secrets: Add your ENTSO-E and Meteosource API keys
5. Deploy!

## Contributing

Contributions are welcome! Areas of interest:
- Additional forecasting models (ARIMA, LSTM, Prophet)
- Enhanced feature engineering
- Model hyperparameter optimization (see `docs/FULL_HYPERPARAM_PIPELINE.md`)
- Extended test coverage
- Documentation improvements

Please open an issue or pull request on GitHub.

## Documentation

- **[FULL_HYPERPARAM_PIPELINE.md](docs/FULL_HYPERPARAM_PIPELINE.md)** - Comprehensive hyperparameter search and analysis pipeline
- **[HYPERPARAMETER_SEARCH.md](docs/HYPERPARAMETER_SEARCH.md)** - Individual model hyperparameter tuning guide
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide for local development
- **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** - Complete project context and architecture

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **ENTSO-E** for transparent power market data access
- **KNMI** for open meteorological data
- **Streamlit** for the excellent deployment platform
- The **Python data science community** for powerful open-source tools

---

**Repository**: [https://github.com/TyredMayfly/Energy_Market_Forecasting_project](https://github.com/TyredMayfly/Energy_Market_Forecasting_project)  
**Live Demo**: [https://energymarketforecastingproject.streamlit.app/](https://energymarketforecastingproject.streamlit.app/)  
**Python Version**: 3.11  
**Last Updated**: November 2025
