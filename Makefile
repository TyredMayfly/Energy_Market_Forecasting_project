# Makefile for Market Forecasting project
# Note: This is a PowerShell-compatible makefile (use with nmake or similar)

.PHONY: help install test lint format clean run-api run-ui docker-build docker-up init-data update-data

help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  init-data    - Initialize historical data"
	@echo "  update-data  - Update with latest data"
	@echo "  test         - Run tests"
	@echo "  lint         - Check code quality"
	@echo "  format       - Format code with black"
	@echo "  run-api      - Run FastAPI backend"
	@echo "  run-ui       - Run Streamlit UI"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up    - Start services with docker-compose"
	@echo "  clean        - Remove generated files"

install:
	pip install -e .

init-data:
	python -m app.services.data_update_service --init

update-data:
	python -m app.services.data_update_service --update

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=app --cov-report=html --cov-report=term

lint:
	black --check .
	ruff check .

format:
	black .

run-api:
	uvicorn app.api.main:app --reload --port 8000

run-ui:
	streamlit run streamlit_app.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up

docker-down:
	docker-compose down

clean:
	Remove-Item -Recurse -Force __pycache__, .pytest_cache, .coverage, htmlcov, *.egg-info -ErrorAction SilentlyContinue
	Get-ChildItem -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
