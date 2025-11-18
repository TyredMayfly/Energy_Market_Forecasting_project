"""
Forecasting API endpoints.

Provides REST API for generating price forecasts and managing models.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from app.core.config import MARKET_TYPES, MODEL_TYPES
from app.core.logging import get_logger
from app.services.data_store import get_data_summary
from app.services.forecast_service import get_forecast_service

logger = get_logger(__name__)

router = APIRouter()


# Request/Response models
class ForecastRequest(BaseModel):
    """Request model for forecast generation."""

    market_type: str = Field(
        ...,
        description="Type of market",
        examples=["day_ahead", "intraday", "imbalance"],
    )
    model_type: str = Field(
        ...,
        description="Type of forecasting model",
        examples=["persistence", "linear_regression", "random_forest"],
    )
    horizon_hours: int = Field(
        ...,
        description="Forecast horizon in hours",
        ge=1,
        le=36,
    )
    forecast_start: Optional[str] = Field(
        None,
        description="Starting timestamp for forecast (ISO format, UTC). If not provided, uses current time.",
    )


class ForecastResponse(BaseModel):
    """Response model for forecast generation."""

    timestamps: List[str] = Field(..., description="Forecast timestamps (ISO format, UTC)")
    forecasts: List[float] = Field(..., description="Forecasted prices in EUR/MWh")
    market_type: str = Field(..., description="Market type")
    model_type: str = Field(..., description="Model type")
    horizon_hours: int = Field(..., description="Forecast horizon")


class DataSummaryResponse(BaseModel):
    """Response model for data summary."""

    markets: dict = Field(..., description="Summary of market data availability")
    weather: dict = Field(..., description="Summary of weather data availability")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    trained_at: str = Field(..., description="Timestamp when model was trained")
    n_samples: int = Field(..., description="Number of training samples")
    feature_columns: List[str] = Field(..., description="List of feature columns")


# Endpoints
@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Status information
    """
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get("/data/summary", response_model=DataSummaryResponse)
async def get_data_summary_endpoint():
    """
    Get summary of available data.

    Returns:
        Summary of market and weather data
    """
    try:
        summary = get_data_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/markets")
async def list_markets():
    """
    List available market types.

    Returns:
        Dictionary of available markets
    """
    return {
        market_type: {
            "display_name": config["display_name"],
        }
        for market_type, config in MARKET_TYPES.items()
    }


@router.get("/models")
async def list_models():
    """
    List available model types.

    Returns:
        Dictionary of available models
    """
    return {
        model_type: {
            "display_name": config["display_name"],
            "description": config["description"],
        }
        for model_type, config in MODEL_TYPES.items()
    }


@router.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """
    Generate a price forecast.

    Args:
        request: Forecast request parameters

    Returns:
        Forecast results
    """
    # Validate market type
    if request.market_type not in MARKET_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid market type. Must be one of: {list(MARKET_TYPES.keys())}",
        )

    # Validate model type
    if request.model_type not in MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Must be one of: {list(MODEL_TYPES.keys())}",
        )

    # Parse forecast start time
    forecast_start = None
    if request.forecast_start:
        try:
            forecast_start = datetime.fromisoformat(request.forecast_start.replace("Z", "+00:00"))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {e}")

    # Get forecast service
    service = get_forecast_service()

    try:
        # Generate forecast
        df_forecast = service.generate_forecast(
            market_type=request.market_type,
            model_type=request.model_type,
            horizon_hours=request.horizon_hours,
            forecast_start=forecast_start,
        )

        if df_forecast is None or df_forecast.empty:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate forecast. Check if data is available.",
            )

        # Convert to response format
        response = ForecastResponse(
            timestamps=[ts.isoformat() for ts in df_forecast["timestamp_utc"]],
            forecasts=df_forecast["forecast_price_eur_per_mwh"].tolist(),
            market_type=request.market_type,
            model_type=request.model_type,
            horizon_hours=request.horizon_hours,
        )

        return response

    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast/compare")
async def compare_models(
    market_type: str = Query(..., description="Type of market"),
    model_types: List[str] = Query(..., description="List of model types to compare"),
    horizon_hours: int = Query(..., ge=1, le=36, description="Forecast horizon in hours"),
):
    """
    Compare forecasts from multiple models.

    Args:
        market_type: Type of market
        model_types: List of model types
        horizon_hours: Forecast horizon

    Returns:
        Combined forecasts from all models
    """
    # Validate inputs
    if market_type not in MARKET_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid market type: {market_type}")

    for model_type in model_types:
        if model_type not in MODEL_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")

    # Get forecast service
    service = get_forecast_service()

    try:
        df_combined = service.compare_models(
            market_type=market_type,
            model_types=model_types,
            horizon_hours=horizon_hours,
        )

        if df_combined is None or df_combined.empty:
            raise HTTPException(status_code=500, detail="Failed to generate forecasts")

        # Convert to dictionary format
        result = df_combined.to_dict(orient="records")

        # Convert timestamps to ISO format
        for record in result:
            record["timestamp_utc"] = record["timestamp_utc"].isoformat()

        return {"forecasts": result, "n_models": len(model_types)}

    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/train")
async def train_model(
    market_type: str = Query(..., description="Type of market"),
    model_type: str = Query(..., description="Type of model"),
    force_retrain: bool = Query(False, description="Force retraining even if cached"),
):
    """
    Train a specific model.

    Args:
        market_type: Type of market
        model_type: Type of model
        force_retrain: Force retraining

    Returns:
        Training status
    """
    # Validate inputs
    if market_type not in MARKET_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid market type: {market_type}")

    if model_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")

    # Get forecast service
    service = get_forecast_service()

    try:
        success = service.train_model(
            market_type=market_type,
            model_type=model_type,
            force_retrain=force_retrain,
        )

        if success:
            model_info = service.get_model_info(market_type, model_type)
            return {
                "status": "success",
                "market_type": market_type,
                "model_type": model_type,
                "info": model_info,
            }
        else:
            raise HTTPException(status_code=500, detail="Model training failed")

    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(
    market_type: str = Query(..., description="Type of market"),
    model_type: str = Query(..., description="Type of model"),
):
    """
    Get information about a trained model.

    Args:
        market_type: Type of market
        model_type: Type of model

    Returns:
        Model information
    """
    # Validate inputs
    if market_type not in MARKET_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid market type: {market_type}")

    if model_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")

    # Get forecast service
    service = get_forecast_service()

    model_info = service.get_model_info(market_type, model_type)

    if model_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model not trained yet: {market_type}/{model_type}",
        )

    # Convert datetime to ISO format
    model_info["trained_at"] = model_info["trained_at"].isoformat()

    return model_info
