# Energy Market Forecasting - Power Price Forecasting for the Netherlands

A production-ready web application for forecasting power market prices in the Netherlands. This interactive Streamlit application demonstrates real-time market forecasting using demo data, combining ENTSO-E market prices with KNMI weather data and machine learning models.

🌐 **Live Demo**: [https://energymarketforecastingproject.streamlit.app/](https://energymarketforecastingproject.streamlit.app/)

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
- **Demo Data**: Pre-loaded with 30,912+ market records and 31,056+ weather records (Jan 1 - Nov 18, 2025)
- **Real-time Weather**: Extended 36 hours into the future for forecasting
- **15-Minute Intervals**: Both market and weather data at quarter-hour resolution

## Target Audience

Data scientists, energy market analysts, students, and professionals interested in power market forecasting and machine learning applications in the energy sector.

## Tech Stack

- **Python 3.11**
- **Frontend**: Streamlit with Plotly visualizations
- **ML Framework**: scikit-learn (LinearRegression, RandomForestRegressor)
- **Data Processing**: pandas, numpy
- **Data Sources**: ENTSO-E Transparency Platform, KNMI Open Data API
- **Testing**: pytest (205 tests, 100% passing)
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
  /models           # ML model implementations (Persistence, LinearRegression, RandomForest)
  /services         # Data loading, feature engineering, forecast generation
/data               # Demo CSV files (30K+ records)
  ├── entsoe_day_ahead_prices_2025.csv
  ├── entsoe_intraday_prices_2025.csv
  ├── entsoe_imbalance_data_2025.csv
  └── knmi_weather_2025.csv
/tests              # 205 pytest tests (unit + integration)
  /unit             # Component-level tests
  /integration      # End-to-end feature tests
/.streamlit         # Streamlit Cloud configuration
streamlit_app.py    # Main application entry point (602 lines)
requirements.txt    # Production dependencies
pyproject.toml      # Full project configuration
```

## Data Sources & Format

### ENTSO-E Transparency Platform
- **Day-Ahead Prices**: Market clearing prices (EUR/MWh) at 15-minute intervals
- **Intraday Prices**: Continuous intraday market prices at 15-minute intervals  
- **Imbalance Data**: System imbalance volumes and prices at 15-minute intervals
- **Bidding Zone**: Netherlands (10YNL----------L)
- **Demo Period**: January 1 - November 18, 2025

### KNMI Weather Data
- **Resolution**: 15-minute meteorological observations
- **Variables**: 
  - Temperature (°C)
  - Wind speed (m/s)
  - Cloud cover (oktas)
  - Precipitation (mm)
  - Global radiation (W/m²) - derived from cloud cover
- **Coverage**: Netherlands weather stations
- **Future Extension**: Weather forecasts extended 36 hours ahead

## Forecasting Models

### 1. Persistence Model
Baseline model that assumes future prices equal the most recent observed value. Useful for establishing performance benchmarks.

### 2. Linear Regression
Uses historical prices, time-based features (hour, day of week, weekend), and selected weather variables. Trained with sklearn's `LinearRegression`.

**Features:**
- Lag features: 1, 2, 3, 6, 12, 24 intervals (demo mode)
- Rolling statistics: 24-interval mean and standard deviation
- Cyclic time encoding: hour and day of week
- Optional weather features based on user selection

### 3. Random Forest
Ensemble model using sklearn's `RandomForestRegressor` with 50 trees. Captures non-linear relationships between features and prices.

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
3. **Data Loading**: Demo CSV files are included in the repository for immediate availability
4. **Live URL**: [https://energymarketforecastingproject.streamlit.app/](https://energymarketforecastingproject.streamlit.app/)

### Manual Deployment

To deploy your own instance:
1. Fork this repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select repository, branch (`main`), and main file (`streamlit_app.py`)
5. Deploy!

## Contributing

Contributions are welcome! Areas of interest:
- Additional forecasting models (ARIMA, LSTM, XGBoost)
- Enhanced feature engineering
- Model hyperparameter optimization
- Extended test coverage
- Documentation improvements

Please open an issue or pull request on GitHub.

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
