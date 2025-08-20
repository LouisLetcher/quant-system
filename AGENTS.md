# Agent Guidelines for Quant System

## Build & Test Commands

### Code Quality
```bash
# Linting and formatting (ruff-only)
ruff check .       # Lint code
ruff format .      # Format code

# Type checking
mypy src/

# Security checks
bandit -r src/
```

### Testing
```bash
# All tests with coverage
docker-compose run --rm quant pytest

# Unit tests only
docker-compose run --rm quant pytest -m "not integration"

# Integration tests
docker-compose run --rm quant pytest -m "integration"

# With coverage reporting
docker-compose run --rm quant pytest --cov=src --cov-report=term-missing
```

### Docker Commands
```bash
# Build and run
docker-compose up --build

# Run CLI commands
docker-compose run --rm quant python -m src.cli.unified_cli portfolio test-all \
  --symbols TLT IEF SHY \
  --start-date 2023-01-01 \
  --end-date 2024-12-31

# Interactive shell
docker-compose run --rm quant bash
```

## Architecture Overview

### Core Components
- **`src/core/direct_backtest.py`**: Direct backtesting library integration (replaces old UnifiedBacktestEngine)
- **`src/core/data_manager.py`**: Data fetching with UTC timezone consistency
- **`src/database/models.py`**: SQLAlchemy models for BestStrategy, Trade, BacktestResult
- **`src/reporting/detailed_portfolio_report.py`**: Real data HTML report generation
- **`src/ai/`**: AI recommendation system using real backtest data

### Key Principles
1. **All outputs use real data from backtesting library** - no fake/mock data
2. **UTC timezone consistency** throughout all datetime operations
3. **Sortino ratio as primary metric** (better than Sharpe for downside risk)
4. **Database-first approach** for data persistence and retrieval

### Code Style Preferences
- **Linting**: ruff-only (replaces black + isort + flake8)
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style for modules, classes, and functions
- **Line length**: 88 characters (ruff default)
- **Import sorting**: Handled by ruff

### Data Flow
1. **Data Sources** → Yahoo Finance (primary), Alpha Vantage, Bybit (crypto)
2. **Backtesting** → Direct `backtesting` library integration
3. **Storage** → PostgreSQL database (BestStrategy, Trade, BacktestResult tables)
4. **Outputs** → HTML reports, CSV exports, TradingView alerts, AI recommendations

### Environment Variables (.env)
```bash
DATABASE_URL=postgresql://quantuser:quantpass@localhost:5432/quant_system
ALPHA_VANTAGE_API_KEY=your_key
TWELVE_DATA_API_KEY=your_key
POLYGON_API_KEY=your_key
TIINGO_API_KEY=your_key
```

### Project Structure
```
src/
├── ai/                   # AI recommendation system
├── cli/                  # Command-line interface
├── core/                 # Core system components
│   ├── data_manager.py   # Data fetching and management
│   ├── direct_backtest.py # Direct backtesting library integration
│   └── portfolio_manager.py # Portfolio management
├── database/             # Database models and operations
├── reporting/            # Report generation
└── utils/                # Utility functions
```

### Common Debugging Tips
1. **Database issues**: Check PostgreSQL container is running with `docker-compose ps`
2. **Import errors**: Ensure Docker rebuild with `docker-compose build quant`
3. **API failures**: Verify API keys in `.env` file
4. **Timezone issues**: All datetimes should use `pd.to_datetime(..., utc=True)`
5. **Test failures**: Clear pytest cache with `rm -rf .pytest_cache`

### Performance Metrics Hierarchy
1. **Sortino Ratio** (primary) - Downside risk-adjusted returns
2. **Calmar Ratio** (secondary) - Annual return / Max drawdown
3. **Sharpe Ratio** (tertiary) - Traditional risk-adjusted returns
4. **Profit Factor** (supplementary) - Gross profit/loss ratio
