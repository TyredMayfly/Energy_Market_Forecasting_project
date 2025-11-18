"""
FastAPI application factory and main entry point.

Creates and configures the FastAPI application with routers, middleware,
and startup/shutdown event handlers.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.forecast import router as forecast_router
from app.core.logging import get_logger
from app.services.data_update_service import start_scheduler, stop_scheduler

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting Market Forecasting API")

    # Start the data update scheduler
    try:
        start_scheduler()
        logger.info("Data update scheduler started")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Market Forecasting API")

    # Stop the scheduler
    try:
        stop_scheduler()
        logger.info("Data update scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Market Forecasting API",
        description="Educational power market forecasting for the Netherlands",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(forecast_router, prefix="/api", tags=["forecasting"])

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "Market Forecasting API",
            "version": "0.1.0",
            "status": "running",
        }

    return app


# Create app instance
app = create_app()
