# Testing and Coverage Guide

This document describes the testing infrastructure and how to use coverage tooling in this project.

## Overview

The project uses **pytest** as the test runner with **pytest-cov** for coverage reporting. All tests are located in the `tests/` directory with separate `unit/` and `integration/` subdirectories.

## Quick Start

### Running Tests

```bash
# Run all tests with coverage (default configuration)
pytest

# Run tests without coverage
pytest --no-cov

# Run specific test files
pytest tests/unit/test_forecast_service.py
pytest tests/integration/

# Run tests matching a pattern
pytest -k "test_linear_regression"
```

### Coverage Reports

The project is configured to generate coverage reports automatically when running `pytest`:

```bash
# Terminal report with missing lines (default)
pytest

# HTML report (interactive, detailed)
pytest --cov-report=html
# Open htmlcov/index.html in your browser

# XML report (for CI tools)
pytest --cov-report=xml

# All formats at once
pytest --cov-report=term-missing --cov-report=html --cov-report=xml
```

## Makefile Commands

For convenience, use these make targets:

```bash
make test              # Run tests with coverage
make test-cov          # Terminal coverage report
make test-cov-html     # HTML coverage report
make test-cov-xml      # XML coverage report (for CI)
make test-cov-all      # Generate all report formats
make clean             # Remove cache and coverage files
```

## Configuration Files

### `.coveragerc`

Coverage configuration file that specifies:
- **Source**: `app/` package is measured
- **Omit**: Excludes `tests/`, `scripts/`, `__init__.py` files
- **Branch coverage**: Enabled (measures if/else branches)
- **Show missing**: Displays line numbers without coverage

### `pyproject.toml` - [tool.pytest.ini_options]

Pytest configuration with coverage enabled by default:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "-v",                          # Verbose output
    "--strict-markers",            # Enforce marker registration
    "--tb=short",                  # Short traceback format
    "--cov=app",                   # Measure coverage of app/ package
    "--cov-report=term-missing",   # Show missing lines in terminal
    "--cov-report=html",           # Generate HTML report
]
```

## Coverage Goals

- **Minimum coverage**: 70% (enforced in CI)
- **Target coverage**: 80%+
- **Focus areas**:
  - Core business logic in `app/services/`
  - Model implementations in `app/models/`
  - Data loading and processing
  - Feature engineering

## Understanding Coverage Reports

### Terminal Report

```
---------- coverage: platform windows, python 3.11.0 -----------
Name                                      Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------
app\__init__.py                               0      0   100%
app\core\config.py                           45      2    96%   78, 92
app\models\linear_regression.py              67      5    93%   45-49
app\services\forecast_service.py            156     12    92%   89, 145-156
-----------------------------------------------------------------------
TOTAL                                       823     47    94%
```

**Columns:**
- **Stmts**: Total statements
- **Miss**: Statements not executed
- **Cover**: Coverage percentage
- **Missing**: Line numbers without coverage

### HTML Report

The HTML report (`htmlcov/index.html`) provides:
- **Interactive interface** with file tree
- **Line-by-line highlighting**:
  - ✅ Green: Covered
  - 🔴 Red: Not covered
  - ⚠️ Yellow: Partially covered (branches)
- **Branch coverage details**
- **Sortable tables** by coverage percentage

## CI/CD Integration

### GitHub Actions

The `.github/workflows/tests.yml` workflow:
1. Runs on push to `main` and `develop` branches
2. Tests with Python 3.11 and 3.12
3. Generates coverage reports
4. **Fails if coverage < 70%**
5. Uploads coverage to Codecov (optional)

```yaml
pytest --cov=app --cov-report=term-missing --cov-report=xml --cov-fail-under=70
```

### Coverage Badge

To add a coverage badge to your README:

1. Sign up at [Codecov.io](https://codecov.io)
2. Connect your GitHub repository
3. Add badge markdown to README.md:
   ```markdown
   ![Coverage](https://codecov.io/gh/TyredMayfly/Energy_Market_Forecasting_project/branch/main/graph/badge.svg)
   ```

## Best Practices

### Writing Testable Code

1. **Separate concerns**: Keep business logic separate from I/O
2. **Dependency injection**: Pass dependencies as parameters
3. **Mock external services**: Use pytest fixtures and mocks
4. **Small functions**: Easier to test and achieve full coverage

### Improving Coverage

1. **Identify gaps**:
   ```bash
   pytest --cov=app --cov-report=term-missing
   ```
   Look at the "Missing" column for uncovered lines.

2. **Add tests for edge cases**:
   - Error conditions
   - Boundary values
   - Invalid inputs
   - Empty data sets

3. **Test branches**:
   - If/else statements
   - Try/except blocks
   - Loop variations
   - Early returns

4. **View detailed HTML report**:
   ```bash
   pytest --cov-report=html
   open htmlcov/index.html
   ```
   Click on files with low coverage to see exactly which lines need tests.

### Coverage != Quality

Remember:
- ✅ High coverage indicates tested code
- ❌ High coverage doesn't guarantee correct behavior
- 🎯 Focus on **meaningful tests** that verify business logic
- 🔍 Use coverage to find **untested code**, not as the only metric

## Troubleshooting

### Coverage not working

```bash
# Ensure pytest-cov is installed
pip install pytest-cov

# Verify installation
pytest --version
# Should show: plugins: cov-4.1.0
```

### HTML report not generated

```bash
# Explicitly specify HTML output
pytest --cov=app --cov-report=html

# Check if htmlcov/ directory was created
ls htmlcov/
```

### Coverage percentage seems wrong

```bash
# Clean old coverage data
rm -rf .coverage htmlcov/

# Re-run tests
pytest
```

### Excluding files from coverage

Edit `.coveragerc`:
```ini
[report]
omit =
    tests/*
    scripts/*
    your_file_to_exclude.py
```

## Advanced Usage

### Parallel coverage

For running tests in parallel:
```bash
pip install pytest-xdist
pytest -n auto --cov=app
```

### Coverage for specific modules

```bash
# Only measure coverage for specific modules
pytest --cov=app.services --cov=app.models

# Exclude specific modules
pytest --cov=app --cov-config=.coveragerc-custom
```

### Incremental coverage

```bash
# Only show coverage for changed files
pytest --cov=app --cov-report=term:skip-covered
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [Codecov documentation](https://docs.codecov.com/)
