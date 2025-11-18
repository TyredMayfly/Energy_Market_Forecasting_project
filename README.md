# Market Forecasting - Educational Power Market Forecasting for the Netherlands

An educational web application for learning about power market forecasting in the Netherlands. This project demonstrates how to fetch real market data from ENTSO-E, combine it with weather data from KNMI, and build simple forecasting models using scikit-learn.

## Features

- **Real Market Data**: Fetches day-ahead, intraday, and imbalance market data from ENTSO-E Transparency Platform
- **Weather Integration**: Incorporates KNMI weather data (temperature, wind speed, solar radiation)
- **Multiple Models**: Compare persistence, linear regression, and random forest forecasting models
- **Interactive UI**: Streamlit-based frontend for easy exploration
- **FastAPI Backend**: RESTful API for forecasting services
- **Automated Updates**: Daily data refresh at midnight (Europe/Amsterdam timezone)

## Target Audience

Students and learners interested in understanding power market forecasting. The code prioritizes clarity and educational value over advanced ML techniques.

## Tech Stack

- **Python 3.11**
- **Backend**: FastAPI, scikit-learn, pandas, numpy
- **Frontend**: Streamlit with Plotly visualizations
- **Data Sources**: ENTSO-E Transparency Platform, KNMI Open Data API
- **Testing**: pytest
- **Code Quality**: black formatter

## Setup

### Prerequisites

- Python 3.11 or higher
- ENTSO-E API key (register at https://transparency.entsoe.eu/)
- KNMI API key (register at https://developer.dataplatform.knmi.nl/)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Market_Forecasting_example
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Create a `.env` file with your API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. Initialize historical data (downloads 2025 data up to today):
```bash
python -m app.services.data_update_service --init
```

## Usage

### Running the FastAPI Backend

```bash
uvicorn app.api.main:app --reload --port 8000
```

API will be available at http://localhost:8000
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Running the Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

UI will be available at http://localhost:8501

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
```

## Project Structure

```
/app
  /api              # FastAPI routers and endpoints
  /core             # Configuration, logging, utilities
  /models           # Forecasting model implementations
  /services         # Data fetchers, ETL, business logic
/data               # Local data storage (CSV files)
/tests              # pytest test suite
/infra              # Docker, CI/CD configurations
streamlit_app.py    # Streamlit UI entrypoint
pyproject.toml      # Project dependencies and configuration
```

## Data Sources

### ENTSO-E Transparency Platform
- **Day-Ahead Prices**: Document type A44
- **Intraday Prices**: Document type A45
- **Imbalance Data**: Document type A53
- **Bidding Zone**: Netherlands (10YNL----------L)

### KNMI Open Data Platform
- **Dataset**: 10-minute or hourly meteorological observations
- **Variables**: Temperature, wind speed, global radiation
- **Coverage**: Netherlands weather stations

## Forecasting Models

### 1. Persistence Model
Simple baseline that assumes future prices equal recent observed values.

### 2. Linear Regression
Uses lagged prices, time features, and weather variables with sklearn's LinearRegression.

### 3. Random Forest
Ensemble model using sklearn's RandomForestRegressor with conservative parameters.

## API Endpoints

### POST /forecast
Generate a price forecast for a specific market, model, and horizon.

**Request Body:**
```json
{
  "market_type": "day_ahead",
  "model_type": "linear_regression",
  "horizon_hours": 24
}
```

**Response:**
```json
{
  "timestamps": ["2025-01-01T00:00:00Z", ...],
  "forecasts": [45.2, 46.1, ...],
  "market_type": "day_ahead",
  "model_type": "linear_regression"
}
```

## Automated Data Updates

The system automatically updates data daily at 00:00 Europe/Amsterdam time using APScheduler. The update fetches the previous day's data from both ENTSO-E and KNMI and appends it to existing datasets.

## Docker Support

Build and run with Docker:

```bash
docker build -t market-forecasting .
docker run -p 8000:8000 --env-file .env market-forecasting
```

Or use docker-compose:

```bash
docker-compose up
```

## Contributing

This is an educational project. Contributions that improve clarity, add documentation, or implement additional simple forecasting techniques are welcome.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- ENTSO-E for providing transparent power market data
- KNMI for open meteorological data
- The open-source Python data science community
