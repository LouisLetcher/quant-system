# Agent Development Guide

This file contains essential commands, conventions, and development practices for the Quant Trading System.

## Project Structure

```
quant-system/
├── src/                           # Main source code
│   ├── backtesting_engine/       # Strategies submodule (quant-strategies repo)
│   │   └── algorithms/python/    # Python strategy implementations (40+ strategies)
│   ├── core/                     # Core trading logic
│   ├── cli/                      # Command-line interface
│   ├── portfolio/                # Portfolio management
│   ├── reporting/                # Report generation
│   └── utils/                    # Utility functions
├── config/                       # Configuration files
│   └── portfolios/               # Portfolio configurations
├── tests/                        # Test suite
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
├── cache/                        # Data cache
├── exports/                      # Export outputs
└── reports_output/               # Generated reports
```

## Essential Commands

### Development Setup
```bash
# Install dependencies
poetry install --with dev

# Activate virtual environment
poetry shell

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run only unit tests
pytest -m "not integration"

# Run only integration tests
pytest -m "integration"

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_data_manager.py

# Run tests in parallel
pytest -n auto
```

### Code Quality
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
ruff check .

# Essential code quality (aligned with simplified CI)
poetry run black .
poetry run isort .
poetry run ruff check .

# Pre-commit hooks (GitHub provides security scanning)
pre-commit run --all-files
```

### System Commands
```bash
# List all portfolios
python -m src.cli.unified_cli portfolio list

# Test a specific portfolio
python -m src.cli.unified_cli portfolio test <portfolio_name>

# Test all portfolios
python -m src.cli.unified_cli portfolio test-all

# Generate reports
python -m src.cli.unified_cli reports generate <portfolio_name>

# Run with Docker
docker-compose up --build
```

### Build and Deployment
```bash
# Build package
poetry build

# Build Docker image
docker build -t quant-system .

# Run production Docker stack
docker-compose -f docker-compose.yml up -d
```

## Code Conventions

### Python Style
- **Line length**: 88 characters (Black default)
- **Imports**: Use isort with Black profile
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style docstrings for all modules, classes, and functions

### Naming Conventions
- **Classes**: PascalCase (e.g., `UnifiedDataManager`)
- **Functions/Methods**: snake_case (e.g., `fetch_data`)
- **Variables**: snake_case (e.g., `portfolio_value`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_COMMISSION`)
- **Files**: snake_case (e.g., `data_manager.py`)

### Project-Specific Patterns
- **Data sources**: Always use fallback mechanisms
- **Symbol transformation**: Handle different data source formats
- **Error handling**: Use try-catch with appropriate logging
- **Configuration**: Store in JSON files under `config/`
- **Caching**: Use file-based caching for market data

### Testing Guidelines
- **Unit tests**: Mock external dependencies
- **Integration tests**: Use `@pytest.mark.integration`
- **Coverage**: Maintain 80%+ code coverage
- **Fixtures**: Use pytest fixtures for common test data
- **Mocking**: Use `unittest.mock` for external services

## Data Sources and APIs

### Supported Data Sources
1. **Yahoo Finance** (primary)
2. **Alpha Vantage** (API key required)
3. **Twelve Data** (API key required)
4. **Polygon.io** (API key required)
5. **Tiingo** (API key required)
6. **Finnhub** (API key required)
7. **Bybit** (crypto data)
8. **Pandas DataReader** (FRED, etc.)

### Environment Variables
```bash
# API Keys (store in .env)
ALPHA_VANTAGE_API_KEY=your_key
TWELVE_DATA_API_KEY=your_key
POLYGON_API_KEY=your_key
TIINGO_API_KEY=your_key
FINNHUB_API_KEY=your_key

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/quant_db

# System settings
CACHE_DIR=./cache
LOG_LEVEL=INFO
```

## Portfolio Configuration

### Required Fields
```json
{
  "name": "Portfolio Name",
  "symbols": ["AAPL", "MSFT"],
  "initial_capital": 100000,
  "commission": 0.001,
  "strategy": {
    "name": "BuyAndHold",
    "parameters": {}
  }
}
```

### Optional Fields
```json
{
  "data_source": {
    "primary_source": "yahoo",
    "fallback_sources": ["alpha_vantage"]
  },
  "risk_management": {
    "max_position_size": 0.1,
    "stop_loss": 0.05,
    "take_profit": 0.15
  },
  "benchmark": "^GSPC"
}
```

## Troubleshooting

### Common Issues
1. **Data fetch failures**: Check internet connection and API keys
2. **Symbol not found**: Verify symbol format for data source
3. **Import errors**: Ensure virtual environment is activated
4. **Permission errors**: Check file permissions for cache/exports directories

### Debug Commands
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Clear cache
rm -rf cache/*

# Reset environment
poetry env remove python
poetry install --with dev
```

## CI/CD Pipeline (Simplified for Showcase)

### GitHub Actions Workflows
- **CI**: Essential checks (format, lint, test, build) on every push/PR
- **Release**: Build and create GitHub releases on git tags
- **Built-in features**: CodeQL security scanning, Dependabot updates

### Pre-commit Hooks
- Black formatting
- isort import sorting
- Ruff linting
- **Note**: GitHub provides security scanning automatically

## Performance Considerations

### Data Caching
- Market data cached for 1 hour (configurable)
- Cache invalidation based on data age
- Compressed storage using Parquet format

### Memory Management
- Stream large datasets when possible
- Use pandas chunking for large files
- Monitor memory usage in long-running processes

### Optimization Tips
- Use vectorized operations (pandas/numpy)
- Parallel processing for independent portfolios
- Database connections pooling for production

## Security Best Practices

### API Keys
- Store in environment variables or .env files
- Never commit keys to version control
- Use different keys for development/production

### Data Validation
- Validate all external data inputs
- Sanitize user inputs in CLI/API
- Use type hints and runtime validation

### Dependency Management
- GitHub Dependabot provides automated security updates
- Keep dependencies updated via automated PRs
- Use lock files for reproducible builds
