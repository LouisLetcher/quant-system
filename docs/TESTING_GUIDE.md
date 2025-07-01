# Testing Guide

This document provides comprehensive information about testing the Quant Trading System.

## Table of Contents

- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Docker Testing](#docker-testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Test Coverage](#test-coverage)
- [Writing Tests](#writing-tests)

## Test Structure

The testing infrastructure is organized as follows:

```
tests/
├── core/                     # Unit tests for core components
│   ├── test_data_manager.py        # Data management tests
│   ├── test_cache_manager.py       # Cache management tests
│   ├── test_backtest_engine.py     # Backtesting engine tests
│   ├── test_portfolio_manager.py   # Portfolio management tests
│   └── test_result_analyzer.py     # Result analysis tests
├── integration/              # Integration tests
│   ├── test_full_workflow.py       # End-to-end workflow tests
│   ├── test_api_integration.py     # API integration tests
│   └── test_data_pipeline.py       # Data pipeline tests
├── cli/                     # CLI tests
│   └── test_unified_cli.py         # CLI command tests
├── conftest.py              # Shared test fixtures
└── pytest.ini              # Pytest configuration
```

## Running Tests

### Quick Test Run

```bash
# Run all tests (excluding slow and API-dependent tests)
poetry run pytest

# Run with coverage report
poetry run pytest --cov=src --cov-report=html
```

### Using the Test Script

The repository includes a comprehensive test script:

```bash
# Run full test suite
./scripts/run-tests.sh

# Include slow tests
./scripts/run-tests.sh --slow

# Include API-dependent tests (requires API keys)
./scripts/run-tests.sh --api
```

### Specific Test Categories

```bash
# Unit tests only
poetry run pytest tests/core/ -v

# Integration tests only
poetry run pytest tests/integration/ -v

# Tests by markers
poetry run pytest -m "not slow and not requires_api"
poetry run pytest -m "slow"
poetry run pytest -m "requires_api"
```

## Test Categories

### Markers

Tests are categorized using pytest markers:

- `unit`: Unit tests for individual components
- `integration`: Integration tests for component interaction
- `slow`: Tests that take longer to run (>30 seconds)
- `requires_api`: Tests that need external API access

### Test Types

#### Unit Tests

Test individual components in isolation:

```python
# Example unit test
def test_cache_manager_init(cache_manager):
    assert cache_manager.max_size_bytes > 0
    assert cache_manager.cache_dir.exists()
```

#### Integration Tests

Test component interactions and workflows:

```python
# Example integration test
@pytest.mark.integration
def test_full_backtest_workflow(data_manager, backtest_engine):
    # Test complete workflow from data fetch to result analysis
    pass
```

#### Performance Tests

Test system performance and benchmarks:

```python
# Example performance test
@pytest.mark.slow
def test_large_portfolio_performance(portfolio_manager):
    # Test with 100+ symbols
    pass
```

## Docker Testing

### Building Test Container

```bash
# Build test image
docker build -t quant-system:test --target testing .

# Run tests in Docker
docker run --rm quant-system:test
```

### Using Docker Compose

```bash
# Run tests with docker-compose
./scripts/run-docker.sh test

# Run with coverage
docker-compose --profile test run --rm quant-test \
    poetry run pytest --cov=src --cov-report=xml
```

### Testing Different Environments

```bash
# Test production image
./scripts/run-docker.sh build
docker run --rm quant-system:latest python -m src.cli.unified_cli --help

# Test development environment
./scripts/run-docker.sh dev
docker-compose exec quant-dev pytest tests/
```

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline runs multiple test stages:

1. **Linting and Code Quality**
   - Ruff linting
   - Black formatting
   - isort import sorting
   - mypy type checking

2. **Unit and Integration Tests**
   - Core component tests
   - Integration workflow tests
   - Coverage reporting

3. **Slow Tests** (main branch only)
   - Performance benchmarks
   - Large dataset tests

4. **Security Scanning**
   - Safety dependency check
   - Bandit security linting

5. **Docker Build and Test**
   - Multi-stage Docker builds
   - Container functionality tests

### Pipeline Configuration

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
```

### Running Pipeline Locally

```bash
# Install act (GitHub Actions locally)
# https://github.com/nektos/act

# Run CI pipeline locally
act -j test
```

## Test Coverage

### Coverage Requirements

- Minimum coverage: 80%
- Core components: 90%+
- Integration tests: 70%+

### Generating Coverage Reports

```bash
# HTML report
poetry run pytest --cov=src --cov-report=html
open htmlcov/index.html

# Terminal report
poetry run pytest --cov=src --cov-report=term-missing

# XML report (for CI)
poetry run pytest --cov=src --cov-report=xml
```

### Coverage Configuration

```ini
# pytest.ini
[tool:pytest]
addopts = 
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80
```

## Writing Tests

### Test Structure Guidelines

1. **Arrange-Act-Assert Pattern**

```python
def test_cache_data_storage(cache_manager, sample_data):
    # Arrange
    symbol = 'AAPL'
    
    # Act
    cache_key = cache_manager.cache_data(symbol, sample_data)
    
    # Assert
    assert cache_key is not None
    assert cache_manager.get_data(symbol) is not None
```

2. **Use Fixtures for Setup**

```python
@pytest.fixture
def sample_portfolio():
    return {
        'name': 'Test Portfolio',
        'symbols': ['AAPL', 'MSFT'],
        'strategies': ['rsi'],
        'risk_profile': 'moderate'
    }
```

3. **Mock External Dependencies**

```python
@patch('src.core.data_manager.yf.download')
def test_yahoo_finance_fetch(mock_download, data_manager, sample_data):
    mock_download.return_value = sample_data
    result = data_manager.fetch_data('AAPL', '2023-01-01', '2023-12-31')
    assert isinstance(result, pd.DataFrame)
```

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<specific_behavior>`

### Common Patterns

#### Testing Async Code

```python
@pytest.mark.asyncio
async def test_async_data_fetch(async_data_manager):
    result = await async_data_manager.fetch_data_async('AAPL')
    assert result is not None
```

#### Testing with Temporary Files

```python
def test_file_operations(tmp_path):
    test_file = tmp_path / "test_data.csv"
    # Use test_file for operations
```

#### Parametrized Tests

```python
@pytest.mark.parametrize("symbol,expected_type", [
    ('AAPL', 'stocks'),
    ('BTCUSDT', 'crypto'),
    ('EURUSD=X', 'forex')
])
def test_symbol_classification(symbol, expected_type):
    result = classify_symbol(symbol)
    assert result == expected_type
```

## Test Environment Setup

### Environment Variables

```bash
# Required for API tests
export BYBIT_API_KEY="your_api_key"
export BYBIT_SECRET_KEY="your_secret_key"
export ALPHA_VANTAGE_API_KEY="your_api_key"

# Test database
export TEST_DATABASE_URL="postgresql://test:test@localhost:5432/test_db"
```

### Test Data

Test data is generated using fixtures:

```python
@pytest.fixture
def market_data():
    """Generate realistic market data for testing."""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    prices = generate_realistic_prices(initial=100, periods=252)
    return pd.DataFrame({
        'Open': prices[:-1],
        'Close': prices[1:],
        'High': prices[1:] * 1.02,
        'Low': prices[1:] * 0.98,
        'Volume': np.random.randint(1000000, 10000000, 252)
    }, index=dates)
```

## Debugging Tests

### Running Single Tests

```bash
# Run specific test
poetry run pytest tests/core/test_cache_manager.py::test_cache_data -v

# Run with debugging
poetry run pytest tests/core/test_cache_manager.py::test_cache_data -v -s
```

### Using pytest-pdb

```bash
# Drop into debugger on failure
poetry run pytest --pdb

# Drop into debugger on first failure
poetry run pytest --pdb -x
```

### Verbose Output

```bash
# Maximum verbosity
poetry run pytest -vvv

# Show local variables on failure
poetry run pytest --tb=long
```

## Performance Testing

### Benchmarking

```python
import time

def test_cache_performance(cache_manager, large_dataset):
    """Test cache performance with large datasets."""
    start_time = time.time()
    
    cache_manager.cache_data('LARGE_DATASET', large_dataset)
    cache_time = time.time() - start_time
    
    start_time = time.time()
    retrieved = cache_manager.get_data('LARGE_DATASET')
    retrieve_time = time.time() - start_time
    
    assert cache_time < 5.0  # Should cache in under 5 seconds
    assert retrieve_time < 1.0  # Should retrieve in under 1 second
    assert len(retrieved) == len(large_dataset)
```

### Memory Testing

```python
import psutil
import os

def test_memory_usage():
    """Test memory usage during operations."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operation
    large_data = generate_large_dataset()
    
    peak_memory = process.memory_info().rss
    memory_increase = peak_memory - initial_memory
    
    # Memory increase should be reasonable
    assert memory_increase < 500 * 1024 * 1024  # Less than 500MB
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure proper Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Database Connection Issues**
   ```bash
   # Check database is running
   docker-compose ps postgres
   ```

3. **API Rate Limits**
   ```bash
   # Use test markers to skip API tests
   poetry run pytest -m "not requires_api"
   ```

### Test Isolation

Ensure tests are isolated:

```python
@pytest.fixture(autouse=True)
def isolate_filesystem(tmp_path, monkeypatch):
    """Isolate each test to its own temporary directory."""
    monkeypatch.chdir(tmp_path)
```

### Cleanup

```python
@pytest.fixture
def cleanup_cache():
    """Clean up cache after tests."""
    cache_manager = UnifiedCacheManager()
    yield cache_manager
    cache_manager.clear_all_cache()
```

This comprehensive testing guide ensures robust test coverage and reliable CI/CD processes for the Quant Trading System.
