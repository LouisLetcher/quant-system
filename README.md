# 🚀 Quantitative Analysis System

A comprehensive, production-ready quantitative analysis system with multi-asset support, advanced portfolio optimization, and extensive backtesting capabilities.

## ✨ Features

### 📊 **Multi-Asset Trading Support**
- **Stocks**: Individual stocks, ETFs, indices (5 specialized TraderFox portfolios)
- **Forex**: 72+ major, minor, and exotic currency pairs
- **Crypto**: 220+ Bybit perpetual futures with real-time data
- **Commodities**: 46+ CFD/rolling futures contracts
- **Bonds**: 30+ government and corporate bond ETFs
- **Indices**: 114+ global country and sector ETFs

### 🌐 **Multiple Data Sources**
- **Yahoo Finance** (Free, no API key required)
- **Alpha Vantage** (Stock/Forex/Crypto data)
- **Twelve Data** (Multi-asset financial data)
- **Polygon.io** (Real-time market data)
- **Tiingo** (Stock and ETF data)
- **Finnhub** (Market data)
- **Bybit** (Crypto derivatives)
- **Pandas DataReader** (Economic data)

### 🧠 **Advanced Portfolio Management**
- **5 Specialized TraderFox Portfolios**:
  - German DAX/MDAX stocks (130 symbols)
  - US Technology sector (275 symbols)
  - US Healthcare/Biotech (450 symbols)
  - US Financials (185 symbols)
  - European blue chips (95 symbols)

### ⚡ **Unified CLI System**
All functionality accessible through a single command interface:

```bash
# Portfolio Testing
poetry run python -m src.cli.unified_cli portfolio test-all --portfolio config/portfolios/crypto.json --metric sharpe_ratio --period max --test-timeframes --open-browser

# Data Management
poetry run python -m src.cli.unified_cli data download --symbols BTC-USD,ETH-USD --start-date 2023-01-01 --source bybit

# Cache Management
poetry run python -m src.cli.unified_cli cache stats
poetry run python -m src.cli.unified_cli cache clear

# Report Generation
poetry run python -m src.cli.unified_cli reports organize
```

### 🔄 **Smart Symbol Transformation**
Automatic symbol format conversion between data sources:
- Yahoo Finance: `EURUSD=X`
- Twelve Data: `EUR/USD`
- Bybit: `BTCUSDT`
- Polygon: `BTC-USD`

### 📈 **Interactive Reporting**
- **HTML Portfolio Reports** with Plotly charts
- **Performance Analytics** with risk metrics
- **Comparison Analysis** across strategies and timeframes
- **Auto-opening browser** for immediate visualization

### 🐳 **Docker Support**
Complete containerization with docker-compose:
- Production deployment
- Development environment
- Testing environment
- Jupyter Lab for analysis
- API service
- Database (PostgreSQL)
- Caching (Redis)
- Monitoring (Prometheus + Grafana)

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Poetry
- Docker (optional)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LouisLetcher/quant-system.git
   cd quant-system
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Test the system**:
   ```bash
   poetry run python -m src.cli.unified_cli cache stats
   ```

### Example Usage

**Test a cryptocurrency portfolio:**
```bash
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/crypto.json \
  --metric sharpe_ratio \
  --period max \
  --test-timeframes \
  --open-browser
```

**Download forex data:**
```bash
poetry run python -m src.cli.unified_cli data download \
  --symbols EURUSD=X,GBPUSD=X \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --source twelve_data
```

**Analyze German DAX stocks:**
```bash
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/stocks_traderfox_dax.json \
  --metric sortino_ratio \
  --period 1y \
  --test-timeframes
```

## 📂 Project Structure

```
quant-system/
├── src/
│   ├── core/                 # Unified system core
│   │   ├── data_manager.py   # Multi-source data management
│   │   ├── backtest_engine.py # Unified backtesting
│   │   └── cache_manager.py  # Intelligent caching
│   ├── cli/                  # Command-line interface
│   │   └── unified_cli.py    # Main CLI entry point
│   ├── reporting/            # Report generation
│   ├── portfolio/            # Portfolio optimization
│   └── utils/                # Utilities
├── config/
│   ├── portfolios/           # Portfolio configurations
│   │   ├── crypto.json       # Crypto futures
│   │   ├── forex.json        # Currency pairs
│   │   ├── bonds.json        # Fixed income
│   │   ├── commodities.json  # Commodity CFDs
│   │   ├── indices.json      # Global indices
│   │   └── stocks_traderfox_*.json # TraderFox stocks
│   └── optimization_config.json
├── docs/                     # Documentation
├── tests/                    # Test suite
├── docker-compose.yml        # Container orchestration
├── Dockerfile               # Container definition
└── pyproject.toml           # Dependencies
```

## 🔧 Configuration

### Portfolio Configuration
Each portfolio is defined in JSON format with:
- **symbols**: List of trading instruments
- **data_sources**: Primary and fallback data sources
- **intervals**: Supported timeframes
- **risk_parameters**: Position sizing and risk management
- **optimization**: Strategy and metric preferences

Example:
```json
{
  "crypto": {
    "name": "Crypto Portfolio",
    "symbols": ["BTCUSDT", "ETHUSDT", ...],
    "data_sources": {
      "primary": ["bybit", "polygon", "twelve_data"],
      "fallback": ["alpha_vantage", "yahoo_finance"]
    },
    "intervals": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
    "risk_profile": "high",
    "leverage": 10
  }
}
```

### Environment Variables
Required API keys and configuration:
```bash
# Data Sources
ALPHA_VANTAGE_API_KEY=your_key
TWELVE_DATA_API_KEY=your_key
POLYGON_API_KEY=your_key
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret

# System Configuration
CACHE_ENABLED=true
CACHE_DURATION_HOURS=24
```

## 🐳 Docker Deployment

### Quick Start with Docker
```bash
# Run production system
docker-compose up quant-system

# Run with full stack
docker-compose --profile database --profile api --profile monitoring up
```

### Available Profiles
- `dev`: Development environment
- `test`: Testing environment
- `api`: Web API service
- `database`: PostgreSQL database
- `cache`: Redis caching
- `monitoring`: Prometheus + Grafana
- `jupyter`: Jupyter Lab analysis

## 📊 Portfolio Portfolios

### Crypto (220+ symbols)
Bybit perpetual futures covering:
- Major cryptocurrencies (BTC, ETH, etc.)
- DeFi tokens
- Layer 1/2 protocols
- Meme coins
- Emerging altcoins

### Forex (72+ pairs)
Complete currency coverage:
- Major pairs (EUR/USD, GBP/USD, etc.)
- Minor pairs (cross currencies)
- Exotic pairs (emerging markets)

### TraderFox Stocks (1000+ symbols)
Research-based stock selection:
- **German DAX**: SAP, Siemens, BMW, etc.
- **US Tech**: FAANG, semiconductors, software
- **US Healthcare**: Pharma, biotech, devices
- **US Financials**: Banks, fintech, insurance
- **European**: ASML, Nestlé, LVMH, etc.

### Bonds (30+ ETFs)
Fixed income diversification:
- Government bonds (US, international)
- Corporate bonds
- TIPS (inflation-protected)
- Municipal bonds

### Commodities (46+ CFDs)
Direct commodity exposure:
- Precious metals (Gold, Silver, Platinum)
- Energy (Oil, Natural Gas, Coal)
- Agriculture (Wheat, Corn, Coffee)
- Industrial metals (Copper, Aluminum)

### Indices (114+ ETFs)
Global market coverage:
- Country-specific ETFs
- Sector ETFs
- Factor-based ETFs
- Regional groupings

## 📚 Documentation

- [Complete CLI Guide](docs/COMPLETE_CLI_GUIDE.md)
- [Data Sources Guide](docs/DATA_SOURCES_GUIDE.md)
- [Docker Guide](docs/DOCKER_GUIDE.md)
- [Symbol Transformation Guide](docs/SYMBOL_TRANSFORMATION_GUIDE.md)
- [System Summary](docs/FINAL_SYSTEM_SUMMARY.md)

## 🧪 Testing & Development

### Testing Commands
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
poetry run black .

# Sort imports
poetry run isort .

# Lint code
poetry run ruff check .

# Pre-commit checks (aligned with CI)
pre-commit run --all-files
```

### Development Setup
```bash
# Install dependencies with dev tools
poetry install --with dev

# Activate virtual environment
poetry shell

# Install pre-commit hooks
pre-commit install

# Build package
poetry build
```

### CI/CD Pipeline (Simplified for Showcase)
- **Essential checks**: Format, lint, test, build on every push/PR
- **GitHub native features**: CodeQL security scanning, Dependabot dependency updates
- **Automated releases**: GitHub releases with artifacts on tags
- **KISS principle**: Minimal, focused workflows leveraging GitHub's built-in capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `docs/` directory
- **Issues**: Open a GitHub issue
- **Discord**: Join our trading community

## 🔗 Links

- **Repository**: https://github.com/LouisLetcher/quant-system
- **Documentation**: https://LouisLetcher.github.io/quant-system
- **Docker Hub**: https://hub.docker.com/r/LouisLetcher/quant-system

---

**⚡ Built for speed, designed for scale, optimized for profit.**
