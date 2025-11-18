"""
Example usage scripts for the Market Forecasting application.

This module demonstrates how to use the various components programmatically.
"""

from datetime import datetime

from app.services.data_store import get_data_summary, load_market_data
from app.services.entsoe_client import EntsoeClient
from app.services.forecast_service import get_forecast_service
from app.services.knmi_client import KnmiClient


def example_1_check_data_availability():
    """Example: Check what data is available."""
    print("=" * 80)
    print("Example 1: Checking Data Availability")
    print("=" * 80)

    summary = get_data_summary()

    print("\nMarket Data:")
    for market_type, info in summary["markets"].items():
        if info["records"] > 0:
            print(
                f"  {market_type:15} : {info['records']:6} records "
                f"({info['start_date'][:10]} to {info['end_date'][:10]})"
            )
        else:
            print(f"  {market_type:15} : No data")

    print("\nWeather Data:")
    if summary["weather"]["records"] > 0:
        info = summary["weather"]
        print(
            f"  Weather: {info['records']} records "
            f"({info['start_date'][:10]} to {info['end_date'][:10]})"
        )
    else:
        print("  No weather data")


def example_2_load_and_explore_market_data():
    """Example: Load and explore market data."""
    print("\n" + "=" * 80)
    print("Example 2: Loading and Exploring Market Data")
    print("=" * 80)

    df = load_market_data("day_ahead")

    if df is not None and not df.empty:
        print(f"\nLoaded {len(df)} day-ahead price records")
        print(f"\nFirst few records:")
        print(df.head())

        print(f"\nPrice statistics:")
        print(df["price_eur_per_mwh"].describe())
    else:
        print("\nNo day-ahead data available. Run data initialization first.")


def example_3_generate_single_forecast():
    """Example: Generate a forecast with a single model."""
    print("\n" + "=" * 80)
    print("Example 3: Generating a Single Forecast")
    print("=" * 80)

    service = get_forecast_service()

    # Generate 24-hour forecast for day-ahead market using linear regression
    df_forecast = service.generate_forecast(
        market_type="day_ahead",
        model_type="linear_regression",
        horizon_hours=24,
    )

    if df_forecast is not None and not df_forecast.empty:
        print(f"\nGenerated {len(df_forecast)} forecast values")
        print(f"\nForecast preview:")
        print(df_forecast.head(10))

        print(f"\nForecast statistics:")
        print(df_forecast["forecast_price_eur_per_mwh"].describe())
    else:
        print("\nFailed to generate forecast. Check data availability.")


def example_4_compare_multiple_models():
    """Example: Compare forecasts from different models."""
    print("\n" + "=" * 80)
    print("Example 4: Comparing Multiple Models")
    print("=" * 80)

    service = get_forecast_service()

    # Compare all models
    df_comparison = service.compare_models(
        market_type="day_ahead",
        model_types=["persistence", "linear_regression", "random_forest"],
        horizon_hours=12,
    )

    if df_comparison is not None and not df_comparison.empty:
        print(f"\nGenerated forecasts from {df_comparison['model_type'].nunique()} models")

        # Show mean forecast by model
        print("\nMean forecast price by model:")
        for model in df_comparison["model_type"].unique():
            df_model = df_comparison[df_comparison["model_type"] == model]
            mean_price = df_model["forecast_price_eur_per_mwh"].mean()
            print(f"  {model:20} : {mean_price:.2f} EUR/MWh")
    else:
        print("\nFailed to generate comparison.")


def example_5_fetch_fresh_data():
    """Example: Fetch fresh data from ENTSO-E (requires API key)."""
    print("\n" + "=" * 80)
    print("Example 5: Fetching Fresh Data from ENTSO-E")
    print("=" * 80)

    try:
        client = EntsoeClient()

        # Fetch last 2 days of day-ahead prices
        end_date = datetime.utcnow()
        start_date = datetime(end_date.year, end_date.month, end_date.day - 2)

        print(f"\nFetching data from {start_date.date()} to {end_date.date()}")

        df = client.fetch_day_ahead_prices_2025_nl(start_date, end_date)

        if not df.empty:
            print(f"Fetched {len(df)} records")
            print("\nRecent prices:")
            print(df.tail())
        else:
            print("No data fetched")

    except ValueError as e:
        print(f"\nError: {e}")
        print("Make sure ENTSOE_API_KEY is set in .env file")
    except Exception as e:
        print(f"\nError fetching data: {e}")


def example_6_train_specific_model():
    """Example: Train a specific model and get information."""
    print("\n" + "=" * 80)
    print("Example 6: Training a Specific Model")
    print("=" * 80)

    service = get_forecast_service()

    # Train random forest model for intraday market
    success = service.train_model(
        market_type="day_ahead",
        model_type="random_forest",
        force_retrain=True,
    )

    if success:
        print("\nModel trained successfully!")

        # Get model info
        info = service.get_model_info("day_ahead", "random_forest")

        if info:
            print(f"\nModel Information:")
            print(f"  Trained at: {info['trained_at']}")
            print(f"  Training samples: {info['n_samples']}")
            print(f"  Number of features: {len(info['feature_columns'])}")
            print(f"\nFeature columns:")
            for feature in info["feature_columns"][:10]:  # Show first 10
                print(f"    - {feature}")
            if len(info["feature_columns"]) > 10:
                print(f"    ... and {len(info['feature_columns']) - 10} more")
    else:
        print("\nModel training failed. Check data availability.")


def main():
    """Run all examples."""
    example_1_check_data_availability()
    example_2_load_and_explore_market_data()
    example_3_generate_single_forecast()
    example_4_compare_multiple_models()
    # example_5_fetch_fresh_data()  # Uncomment to test (requires API key)
    example_6_train_specific_model()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
