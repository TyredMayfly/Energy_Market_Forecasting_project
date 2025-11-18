# Testing Guide for Market Forecasting Application

## Running Tests

### Run all tests
```powershell
pytest
```

### Run with coverage
```powershell
pytest --cov=app --cov-report=html --cov-report=term
```

### Run specific test modules
```powershell
pytest tests/unit/test_entsoe_client.py -v
pytest tests/integration/test_api_forecast_endpoint.py -v
```

### Run with verbose output
```powershell
pytest -v -s
```

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures for all tests
├── unit/                    # Unit tests for individual modules
│   ├── conftest.py          # Unit-specific fixtures
│   ├── test_config.py       # Configuration tests
│   ├── test_entsoe_client.py
│   ├── test_knmi_client.py
│   ├── test_data_store.py
│   ├── test_data_update_service.py
│   ├── test_feature_engineering.py
│   ├── test_models_persistence.py
│   ├── test_models_linear_regression.py
│   ├── test_models_random_forest.py
│   └── test_forecast_service.py
└── integration/             # Integration tests
    ├── conftest.py          # Integration-specific fixtures
    └── test_api_forecast_endpoint.py
```

## Mocking Strategy

### External APIs
All external API calls are mocked to ensure:
- No real network requests during tests
- No API keys required for testing
- Fast, deterministic test execution

**Mocked Services:**
- ENTSO-E Transparency Platform API (`requests.get`)
- KNMI Open Data Platform API (`requests.get`)
- File system operations use `tmp_path` fixtures

### Environment Variables
Environment variables are mocked using `monkeypatch`:
- `ENTSOE_API_KEY` → `"test_entsoe_key"`
- `KNMI_API_KEY` → `"test_knmi_key"`
- `DATA_DIR` → temporary directory

## Key Fixtures

### `test_data_dir` (conftest.py)
Provides a temporary directory for data files that is cleaned up after tests.

### `mock_env_vars` (conftest.py)
Sets up mock environment variables for configuration.

### `sample_entsoe_xml` (unit/conftest.py)
Provides valid ENTSO-E XML responses for testing parsing.

### `sample_market_data` (unit/conftest.py)
Provides sample pandas DataFrames with market data.

### `sample_weather_data` (unit/conftest.py)
Provides sample pandas DataFrames with weather data.

### `mock_requests` (unit/conftest.py)
Provides pre-configured mock objects for HTTP requests.

## Writing New Tests

### Test Naming Convention
- Use descriptive names: `test_<function>_<scenario>`
- Examples:
  - `test_parse_xml_with_valid_data()`
  - `test_parse_xml_with_malformed_input()`
  - `test_forecast_with_invalid_market_type()`

### Test Structure
```python
def test_feature_does_something():
    """Test that feature behaves correctly under specific conditions."""
    # Arrange - set up test data
    input_data = create_test_data()
    
    # Act - execute the code being tested
    result = function_under_test(input_data)
    
    # Assert - verify expected behavior
    assert result.shape == (10, 5)
    assert result["column_name"].dtype == "float64"
```

### Using Fixtures
```python
def test_with_temp_data(test_data_dir, sample_market_data):
    """Test using provided fixtures."""
    # test_data_dir is a Path to temp directory
    # sample_market_data is a DataFrame
    save_path = test_data_dir / "market_data.csv"
    sample_market_data.to_csv(save_path)
    
    # Test loading
    loaded = pd.read_csv(save_path)
    assert len(loaded) == len(sample_market_data)
```

## Common Test Patterns

### Testing Error Handling
```python
def test_raises_error_on_invalid_input():
    """Test that appropriate errors are raised."""
    with pytest.raises(ValueError, match="Invalid market type"):
        function_that_should_fail("invalid_input")
```

### Testing HTTP Mocking
```python
def test_api_call(mock_requests):
    """Test with mocked HTTP response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "<xml>response</xml>"
    mock_requests.get.return_value = mock_response
    
    result = make_api_call()
    assert result is not None
```

### Testing DataFrames
```python
def test_dataframe_structure(sample_market_data):
    """Test DataFrame structure."""
    assert "timestamp_utc" in sample_market_data.columns
    assert sample_market_data["timestamp_utc"].dtype == "datetime64[ns]"
    assert sample_market_data["price_eur_per_mwh"].min() >= 0
```

## CI/CD Integration

Tests run automatically in GitHub Actions on:
- Push to `main` or `develop` branches
- Pull requests

The CI pipeline:
1. Sets up Python 3.11
2. Installs dependencies
3. Runs `black --check` for formatting
4. Runs `pytest` with coverage reporting
5. Builds Docker images

## Troubleshooting

### Import Errors
If you see import errors, ensure you're running from the project root:
```powershell
cd c:\Users\thdeb\Projects\Market_Forecasting_example
pytest
```

### Fixture Not Found
Ensure `conftest.py` files are present in parent directories of your tests.

### Mock Not Working
Check that you're patching at the right import location:
```python
# Patch where it's used, not where it's defined
@patch('app.services.entsoe_client.requests.get')
```

### Data Directory Issues
Tests use temporary directories. If you need to inspect test data:
```python
def test_with_debug(test_data_dir):
    print(f"Test data dir: {test_data_dir}")
    # Data will be cleaned up after test
```

## Coverage Goals

Target coverage by module:
- Core (config, logging): 90%+
- Clients (ENTSO-E, KNMI): 85%+
- Data services: 80%+
- Feature engineering: 80%+
- Models: 75%+
- Forecast service: 85%+
- API endpoints: 90%+

## Adding New Tests

When adding new functionality:
1. Write tests first (TDD approach recommended)
2. Test both happy path and edge cases
3. Mock external dependencies
4. Ensure tests are fast (<1s per test ideally)
5. Add docstrings explaining what is tested

## Questions or Issues?

If tests fail unexpectedly:
1. Read the error message carefully
2. Check that mocks are configured correctly
3. Verify test data matches expected formats
4. Run with `-v -s` for verbose output
5. Check `conftest.py` fixtures are being used correctly
