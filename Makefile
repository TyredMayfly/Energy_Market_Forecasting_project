.PHONY: help test test-cov test-cov-html test-cov-xml clean lint format install

help:
	@echo "Available targets:"
	@echo "  help           - Show this help message"
	@echo "  install        - Install the package in development mode"
	@echo "  test           - Run all tests with coverage"
	@echo "  test-cov       - Run tests with coverage report in terminal"
	@echo "  test-cov-html  - Run tests and generate HTML coverage report"
	@echo "  test-cov-xml   - Run tests and generate XML coverage report (for CI)"
	@echo "  lint           - Run code linters (ruff)"
	@echo "  format         - Format code with black"
	@echo "  clean          - Remove cache and coverage files"

install:
	pip install -e .[dev]

test:
	pytest

test-cov:
	pytest --cov=app --cov-report=term-missing

test-cov-html:
	pytest --cov=app --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

test-cov-xml:
	pytest --cov=app --cov-report=xml

test-cov-all:
	pytest --cov=app --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "Coverage reports generated:"
	@echo "  - Terminal output above"
	@echo "  - HTML: htmlcov/index.html"
	@echo "  - XML: coverage.xml"

lint:
	ruff check app tests scripts

format:
	black app tests scripts streamlit_app.py

clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
