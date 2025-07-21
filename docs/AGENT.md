# Agent Development Guide

Essential commands and conventions for the Quant Trading System (Docker-only setup).

## Project Structure

```
quant-system/
├── src/                     # Core source code
│   ├── core/               # Trading logic & backtesting
│   ├── cli/                # Command-line interface
│   └── utils/              # Utilities & data management
├── config/portfolios/      # Portfolio configurations (220+ crypto symbols)
├── exports/               # Generated reports (Docker mounted)
├── cache/                 # Data cache (Docker mounted)
└── logs/                  # System logs (Docker mounted)
```

## Essential Commands

### Docker Commands (Primary)
```bash
# Build and run
docker-compose up --build

# Interactive shell
docker-compose run --rm quant bash

# Run backtest
docker-compose run --rm quant python -m src.cli.unified_cli portfolio backtest \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --strategy BuyAndHold \
  --start-date 2023-01-01 \
  --end-date 2024-12-31

# Test all portfolios (Sortino ratio default)
docker-compose run --rm quant python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/crypto.json \
  --metric sortino_ratio \
  --period max

# Cache management
docker-compose run --rm quant python -m src.cli.unified_cli cache stats
docker-compose run --rm quant python -m src.cli.unified_cli cache clear
```

### Testing
```bash
# Run tests in Docker
docker-compose run --rm quant pytest
```

### Code Quality (Development)
```bash
# Format code (if developing locally)
black .
isort .
ruff check .

# Lint markdown
markdownlint **/*.md
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
- **Performance Metrics**: Sortino ratio is the default (superior to Sharpe)
- **Data sources**: Always use fallback mechanisms
- **Symbol transformation**: Handle different data source formats
- **Error handling**: Use try-catch with appropriate logging
- **Configuration**: Store in JSON files under `config/`
- **Caching**: Use file-based caching for market data

### Performance Metrics Hierarchy
1. **Sortino Ratio** (primary) - Downside risk-adjusted returns
2. **Calmar Ratio** (secondary) - Annual return / Max drawdown
3. **Sharpe Ratio** (tertiary) - Traditional risk-adjusted returns
4. **Profit Factor** (supplementary) - Gross profit/loss ratio

**Why Sortino over Sharpe:**
- **Sortino** only penalizes **downside volatility** (what investors actually care about)
- **Sharpe** penalizes all volatility, including upside moves (which aren't really "risk")
- **Hedge funds prefer Sortino** because upside volatility is desirable

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
  "benchmark": "^GSPC",
  "optimization": {
    "metric": "sortino_ratio",
    "secondary_metrics": ["calmar_ratio", "sharpe_ratio", "profit_factor"]
  }
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
- Markdownlint
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
