# Multi-stage Dockerfile for Market Forecasting application

# Stage 1: Base Python image with dependencies
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libhdf5-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Stage 2: API service
FROM base as api

COPY app/ ./app/
COPY data/ ./data/

# Expose port for FastAPI
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Streamlit service
FROM base as streamlit

COPY app/ ./app/
COPY streamlit_app.py ./
COPY data/ ./data/

# Expose port for Streamlit
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
