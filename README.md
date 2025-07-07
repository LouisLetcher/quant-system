# 🚀 Quant Trading System - Unified Architecture

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Poetry](https://img.shields.io/badge/Poetry-Package%20Manager-1E293B)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Architecture: Unified](https://img.shields.io/badge/Architecture-Unified-green.svg)](#-architecture-overview)

A comprehensive quantitative trading system with 32+ strategies, 6+ data sources, and intelligent multi-asset portfolio analysis. Features automatic symbol transformation, quarterly report organization, and dynamic strategy discovery.

## 🌟 Latest Updates

### ✅ **32+ Trading Strategies**
All strategies dynamically discovered and tested automatically across all portfolios!

### ✅ **6+ Premium Data Sources** 
- **Polygon.io**, **Alpha Vantage**, **Twelve Data**, **Tiingo**, **Finnhub**, **Bybit**
- **Automatic Symbol Transformation** per data source
- **Smart Fallback System** with error recovery

### ✅ **Fixed Interactive Charts**
- **Plotly Integration** with error handling
- **Safe JavaScript Variables** for symbols with special characters
- **CDN Fallback Protection** for offline scenarios

### ✅ **Quarterly Report Organization**
- **Automatic Organization** by year/quarter structure
- **Single Report Per Quarter** per portfolio
- **Git Tracking** of organized reports

## ✨ Core Features

### 🌐 **Multi-Source Data Management**
- **6+ Premium Sources**: Polygon, Alpha Vantage, Twelve Data, Tiingo, Finnhub, Bybit
- **Symbol Transformation**: Smart format conversion (EURUSD=X ↔ EUR/USD ↔ EURUSD)
- **Intelligent Routing**: Asset type detection for optimal source selection
- **Advanced Caching**: SQLite-based with TTL and compression

### 🎯 **Complete Strategy Suite (32+ Strategies)**
- **Trend Following**: Index Trend, Moving Average systems, Confident Trend
- **Mean Reversion**: RSI, Bollinger Bands, Simple Mean Reversion  
- **Momentum**: MACD, MFI, Larry Williams %R
- **Breakout**: Donchian Channels, Turtle Trading, Weekly Breakout
- **Pattern Recognition**: Bullish Engulfing, Inside Day, Kings Counting
- **Calendar Effects**: Turnaround Monday/Tuesday, Russell Rebalancing
- **And 20+ more strategies automatically discovered!**

### 📊 **5 Pre-Built Asset Portfolios**
- **💱 Forex Portfolio**: 16 major pairs with optimized forex data sources
- **₿ Crypto Portfolio**: 10 major cryptocurrencies with Bybit integration
- **🌍 World Indices**: 8 global ETFs with broad market exposure
- **🥇 Commodities**: 12 commodity ETFs including metals, energy, agriculture
- **🏛️ Bonds Portfolio**: 12 bond ETFs with government and corporate exposure
- **Comprehensive Analysis**: 50+ performance metrics

### ⚡ **High-Performance Backtesting**
- **Parallel Processing**: 4-8x faster with multi-core support
- **Memory Optimization**: Handle thousands of assets efficiently
- **Incremental Updates**: Only process new data
- **Batch Operations**: Efficient multi-strategy testing
- **Smart Caching**: Avoid redundant calculations

### 🪙 **Crypto Futures Trading**
- **Bybit Integration**: Professional-grade futures trading support
- **Leverage Support**: Up to 100x leverage for crypto futures
- **Real-time Data**: Sub-second latency for live trading
- **Risk Management**: Built-in position sizing and stop-losses
- **Multiple Timeframes**: From 1-minute to monthly data

### 🎯 **Advanced Analytics**
- **50+ Risk Metrics**: Comprehensive performance analysis
- **Portfolio Optimization**: Modern portfolio theory implementation
- **Strategy Comparison**: Side-by-side performance analysis
- **Interactive Reports**: HTML reports with Plotly visualizations
- **Investment Recommendations**: AI-driven portfolio suggestions

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/quant-system.git
cd quant-system

# Install dependencies
poetry install

# Activate environment
poetry shell
```

### 2. Quick Start - Most Used Commands

#### 🚀 **Test Forex Portfolio (16 pairs, 32+ strategies)**
```bash
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/forex.json \
  --metric sharpe_ratio \
  --period max \
  --test-timeframes \
  --open-browser
```

#### ₿ **Test Crypto Portfolio (10 coins, 32+ strategies)**
```bash
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/crypto.json \
  --metric sortino_ratio \
  --period max \
  --test-timeframes \
  --open-browser
```

#### 🌍 **Test World Indices Portfolio (8 ETFs, 32+ strategies)**
```bash
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/world_indices.json \
  --metric profit_factor \
  --period max \
  --test-timeframes \
  --open-browser
```

#### 📊 **Organize and Manage Reports**
```bash
# Organize existing reports into quarterly structure
poetry run python -m src.cli.unified_cli reports organize

# List quarterly reports
poetry run python -m src.cli.unified_cli reports list

# Get latest report for portfolio
poetry run python -m src.cli.unified_cli reports latest "Forex Portfolio"
```

### 3. Crypto Futures Trading
```bash
# Set up Bybit API (optional, uses demo data otherwise)
export BYBIT_API_KEY="your_api_key"
export BYBIT_API_SECRET="your_api_secret"

# Download crypto futures data
python -m src.cli.unified_cli data download \
    --symbols BTCUSDT ETHUSDT BNBUSDT \
    --futures \
    --start-date 2023-01-01 \
    --end-date 2023-12-31

# Backtest crypto futures strategies
python -m src.cli.unified_cli backtest single \
    --symbol BTCUSDT \
    --strategy rsi \
    --futures \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

### 4. Portfolio Investment Planning
```bash
# Generate investment plan with capital allocation
python -m src.cli.unified_cli portfolio plan \
    --portfolios portfolio_results.json \
    --capital 100000 \
    --risk-tolerance moderate \
    --output investment_plan.json
```

## 📊 Performance Benchmarks

| Feature | Before Restructuring | After Restructuring | Improvement |
|---------|---------------------|---------------------|-------------|
| **Data Fetching** | 5-15 seconds | 0.5-2 seconds | **10x faster** |
| **Backtesting** | Sequential | Parallel | **4-8x faster** |
| **Memory Usage** | High overhead | Optimized | **50% reduction** |
| **Code Duplication** | ~1,500 lines | Minimal | **60% reduction** |
| **Cache Hit Rate** | 0% (no caching) | 80%+ | **New feature** |

## 🏗️ Architecture Overview

```
src/core/                    # Unified Core Components
├── data_manager.py         # Multi-source data with Bybit integration
├── cache_manager.py        # SQLite-based advanced caching
├── backtest_engine.py      # Parallel backtesting engine
├── result_analyzer.py      # Comprehensive metrics calculator
└── portfolio_manager.py    # Investment prioritization system

src/cli/                     # Command Line Interface
├── unified_cli.py          # Main CLI with all functionality
└── main.py                 # Entry point (redirects to unified CLI)

examples/                    # Usage Examples
├── comprehensive_example.py # Complete system demonstration
└── output/                 # Generated reports and results

docs/                       # Documentation
├── API.md                  # API reference
├── INSTALLATION.md         # Setup instructions
├── USAGE.md               # Usage examples
└── CONTRIBUTING.md        # Development guidelines
```

## 📈 Portfolio Investment Features

### **Risk-Adjusted Scoring System**
- **Return Score**: Total return, annualized return, Sharpe ratio, win rate
- **Risk Score**: Max drawdown, volatility, VaR, Sortino ratio
- **Overall Score**: Weighted combination optimized for investment decisions

### **Investment Prioritization**
```python
from src.core import PortfolioManager

# Analyze multiple portfolios
portfolio_manager = PortfolioManager()
analysis = portfolio_manager.analyze_portfolios({
    'Conservative Growth': conservative_results,
    'Aggressive Tech': tech_results,
    'Crypto Futures': crypto_results
})

# Get investment recommendations
for rec in analysis['investment_recommendations']:
    print(f"{rec['priority_rank']}. {rec['portfolio_name']}")
    print(f"   Allocation: {rec['recommended_allocation_pct']:.1f}%")
    print(f"   Expected Return: {rec['expected_annual_return']:.2f}%")
    print(f"   Risk Level: {rec['risk_category']}")
```

### **Capital Allocation Planning**
- **Risk Tolerance Matching**: Conservative, Moderate, Aggressive profiles
- **Implementation Timeline**: Phased investment approach
- **Risk Management Rules**: Stop-losses, position limits, rebalancing triggers
- **Performance Monitoring**: Automated tracking and alerts

## 🔧 Configuration

### **Environment Variables**
```bash
# Bybit API for crypto futures (optional)
export BYBIT_API_KEY="your_api_key"
export BYBIT_API_SECRET="your_api_secret"
export BYBIT_TESTNET="false"  # Set to true for testing

# Additional data sources (optional)
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
export TWELVE_DATA_API_KEY="your_twelve_data_key"

# Cache configuration
export CACHE_SIZE_GB="10"
export CACHE_TTL_HOURS="48"
```

### **Cache Management**
```bash
# View cache statistics
python -m src.cli.unified_cli cache stats

# Clear old cache entries
python -m src.cli.unified_cli cache clear --older-than 30

# Clear specific cache types
python -m src.cli.unified_cli cache clear --type data
```

## 🎯 Use Cases

### **Professional Fund Management**
- Analyze thousands of assets simultaneously
- Generate investment recommendations with risk scoring
- Create diversified portfolios with optimal allocation
- Monitor performance with comprehensive risk metrics

### **Crypto Futures Trading**
- Access Bybit's professional trading platform
- Implement leveraged strategies with risk management
- Real-time data for algorithmic trading
- Comprehensive backtesting on historical futures data

### **Quantitative Research**
- Test complex multi-asset strategies
- Optimize parameters across large datasets
- Compare strategy performance across asset classes
- Generate publication-ready research reports

### **Individual Investors**
- Get AI-driven portfolio recommendations
- Understand risk-adjusted returns
- Implement professional-grade strategies
- Monitor portfolio performance automatically

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup instructions
- **[API Reference](docs/API.md)**: Complete API documentation
- **[Usage Examples](docs/USAGE.md)**: Comprehensive usage guide
- **[Contributing](docs/CONTRIBUTING.md)**: Development guidelines
- **[Architecture](RESTRUCTURING_SUMMARY.md)**: System design overview

## 🧪 Testing

```bash
# Run comprehensive examples
python examples/comprehensive_example.py

# Test data fetching
python -m src.cli.unified_cli data sources

# Test crypto futures (requires API key)
python -m src.cli.unified_cli data symbols --asset-type crypto

# Run basic backtest
python -m src.cli.unified_cli backtest single \
    --symbol AAPL --strategy rsi \
    --start-date 2023-01-01 --end-date 2023-12-31
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### **Development Setup**
```bash
# Clone and setup development environment
git clone https://github.com/yourusername/quant-system.git
cd quant-system
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
ruff check src/
black src/
```

## 📊 Example Results

### **Portfolio Analysis Output**
```
Portfolio Rankings:
==================

1. Aggressive Tech
   Overall Score: 85.2/100
   Average Return: 24.8%
   Sharpe Ratio: 1.45
   Risk Category: Moderate
   Max Drawdown: -12.3%

2. Conservative Growth  
   Overall Score: 78.9/100
   Average Return: 12.4%
   Sharpe Ratio: 1.28
   Risk Category: Conservative
   Max Drawdown: -6.7%

Investment Recommendations:
==========================

1. Aggressive Tech
   Recommended Allocation: 35.0%
   Expected Return: 24.8%
   Risk Level: Moderate
   Confidence Score: 87.3/100
```

## 🛡️ Risk Management

### **Built-in Safety Features**
- **Position Sizing**: Automatic position sizing based on risk tolerance
- **Stop Losses**: Configurable stop-loss rules per strategy
- **Drawdown Limits**: Portfolio-level drawdown protection
- **Correlation Monitoring**: Automatic diversification analysis
- **Leverage Controls**: Maximum leverage limits for futures trading

### **Risk Metrics**
- **Value at Risk (VaR)**: 95% and 99% confidence intervals
- **Conditional VaR**: Expected shortfall analysis
- **Maximum Drawdown**: Peak-to-trough analysis
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Sortino Ratio**: Downside deviation focus
- **Calmar Ratio**: Return vs maximum drawdown

## 🌟 What's New in v2.0

✅ **Unified Architecture**: Eliminated 60% of duplicate code  
✅ **Bybit Integration**: Professional crypto futures trading  
✅ **Portfolio Prioritization**: AI-driven investment recommendations  
✅ **10x Performance**: Advanced caching and parallel processing  
✅ **Risk Management**: Comprehensive risk analysis tools  
✅ **Professional CLI**: Single interface for all operations  

## 📊 CLI Command Reference

### **Data Management**
```bash
# Download market data
python -m src.cli.unified_cli data download --symbols AAPL MSFT --start-date 2023-01-01 --end-date 2023-12-31

# Show available data sources
python -m src.cli.unified_cli data sources

# List available symbols
python -m src.cli.unified_cli data symbols --asset-type crypto
```

### **Backtesting**
```bash
# Single backtest
python -m src.cli.unified_cli backtest single --symbol AAPL --strategy rsi --start-date 2023-01-01 --end-date 2023-12-31

# Batch backtests
python -m src.cli.unified_cli backtest batch --symbols AAPL MSFT GOOGL --strategies rsi macd --start-date 2023-01-01 --end-date 2023-12-31
```

### **Portfolio Management**
```bash
# Portfolio backtest
python -m src.cli.unified_cli portfolio backtest --symbols AAPL MSFT GOOGL --strategy rsi --start-date 2023-01-01 --end-date 2023-12-31

# Compare portfolios
python -m src.cli.unified_cli portfolio compare --portfolios portfolios.json --start-date 2023-01-01 --end-date 2023-12-31

# Generate investment plan
python -m src.cli.unified_cli portfolio plan --portfolios results.json --capital 100000 --risk-tolerance moderate
```

### **Cache Management**
```bash
# Show cache statistics
python -m src.cli.unified_cli cache stats

# Clear cache
python -m src.cli.unified_cli cache clear --type data --older-than 30
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Complete Documentation

### **🚀 Quick Reference**
- **[Complete CLI Guide](docs/COMPLETE_CLI_GUIDE.md)** - All CLI commands, options, and examples
- **[Data Sources Guide](docs/DATA_SOURCES_GUIDE.md)** - Multi-source data configuration
- **[Symbol Transformation Guide](docs/SYMBOL_TRANSFORMATION_GUIDE.md)** - Symbol format conversion

### **🔧 Technical Guides**
- **[Testing Guide](docs/TESTING_GUIDE.md)** - Comprehensive testing documentation
- **[Docker Guide](docs/DOCKER_GUIDE.md)** - Container deployment instructions
- **[Production Ready Guide](docs/PRODUCTION_READY.md)** - Production deployment guide

### **📊 Analysis Features**
- **32+ Trading Strategies** - Automatically discovered and tested
- **5 Asset Class Portfolios** - Forex, Crypto, Stocks, Commodities, Bonds
- **6+ Premium Data Sources** - Polygon, Alpha Vantage, Twelve Data, Tiingo, Finnhub, Bybit
- **Interactive Reports** - Plotly charts with quarterly organization
- **Performance Metrics** - Sharpe, Sortino, Profit Factor, Max Drawdown

### **⚡ Key Commands Summary**

```bash
# Test all 32+ strategies on forex portfolio
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/forex.json --metric sharpe_ratio --period max --test-timeframes --open-browser

# Test crypto portfolio with Sortino ratio
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/crypto.json --metric sortino_ratio --period max --test-timeframes --open-browser

# Organize quarterly reports  
poetry run python -m src.cli.unified_cli reports organize

# View all available strategies and data sources
poetry run python -m src.cli.unified_cli data sources
```

### **🌟 Latest Results**

**Forex Portfolio (16 pairs, 32 strategies tested)**:
1. **Weekly Breakout** (1.67 Sharpe) 
2. **Inside Day** (1.64 Sharpe)
3. **Stan Weinstein Stage 2** (1.64 Sharpe)
4. **Linear Regression** (1.60 Sharpe)
5. **Moving Average Crossover** (1.57 Sharpe)

**All strategies tested with full historical data (2015-present) and proper benchmark comparisons!**

---

**Built with ❤️ for quantitative traders and investors worldwide.**

## 🔗 Repository Links

- **Issues**: [GitHub Issues](https://github.com/LouisLetcher/quant-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LouisLetcher/quant-system/discussions)
- **Latest Release**: [Releases](https://github.com/LouisLetcher/quant-system/releases)
