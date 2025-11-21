# Coverage Quick Reference

## Essential Commands

```bash
# Run tests with coverage (default)
pytest

# Run without coverage
pytest --no-cov

# Terminal report with missing lines
pytest --cov-report=term-missing

# HTML report (interactive)
pytest --cov-report=html
# Then open: htmlcov/index.html

# Set minimum coverage threshold
pytest --cov-fail-under=80

# Coverage for specific module
pytest --cov=app.services
```

## Makefile Shortcuts

```bash
make test              # Run tests with coverage
make test-cov          # Terminal coverage report  
make test-cov-html     # HTML coverage report
make test-cov-all      # All formats (term + HTML + XML)
make clean             # Remove coverage files
```

## Coverage Files

- `.coveragerc` - Coverage configuration
- `.coverage` - Coverage data (generated, ignored by git)
- `htmlcov/` - HTML report directory (ignored by git)
- `coverage.xml` - XML report for CI (ignored by git)

## Configuration Locations

### pyproject.toml
```toml
[tool.pytest.ini_options]
addopts = [
    "--cov=app",
    "--cov-report=term-missing",
]
```

### .coveragerc
```ini
[run]
source = app
branch = True

[report]
omit = tests/*, scripts/*
show_missing = True
```

## Reading Coverage Output

```
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
app/services/api.py       156     12    92%   89, 145-156
-----------------------------------------------------
TOTAL                     823     47    94%
```

- **Stmts**: Total statements
- **Miss**: Uncovered statements  
- **Cover**: Coverage percentage
- **Missing**: Line numbers without coverage

## CI Coverage Gate

Minimum coverage: **70%** (enforced in GitHub Actions)

```bash
pytest --cov-fail-under=70
```

## Tips

✅ **DO**:
- Run `pytest` regularly during development
- Check HTML report for detailed line coverage
- Aim for 80%+ coverage on new code
- Focus tests on business logic

❌ **DON'T**:
- Obsess over 100% coverage
- Skip testing edge cases
- Test implementation details
- Sacrifice test quality for coverage

## Troubleshooting

```bash
# Clean old coverage data
make clean

# Verify pytest-cov is installed
pytest --version  # Should show: plugins: cov-X.X.X

# Re-install dev dependencies
pip install -e .[dev]
```

## More Information

See `docs/TESTING_AND_COVERAGE.md` for comprehensive guide.
