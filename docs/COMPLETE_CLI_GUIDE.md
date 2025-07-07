# Complete CLI Guide - Quant Trading System

## Overview
Comprehensive guide to all CLI commands, features, and usage patterns for the unified quant trading system.

---

## 🚀 Quick Start Commands

### Most Used Commands

#### 1. **Test Forex Portfolio with All Strategies**
```bash
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/forex.json \
  --metric sharpe_ratio \
  --period max \
  --test-timeframes \
  --open-browser
```

#### 2. **Test Crypto Portfolio**
```bash
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/crypto.json \
  --metric sortino_ratio \
  --period max \
  --test-timeframes \
  --open-browser
```

#### 3. **Test World Indices Portfolio**
```bash
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/world_indices.json \
  --metric profit_factor \
  --period max \
  --test-timeframes \
  --open-browser
```

---

## 📋 Complete Command Reference

### **Data Commands**

#### Download Market Data
```bash
# Download specific symbols
poetry run python -m src.cli.unified_cli data download \
  --symbols AAPL MSFT GOOGL \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --interval 1d \
  --source yahoo_finance

# Download forex data
poetry run python -m src.cli.unified_cli data download \
  --symbols EURUSD=X GBPUSD=X \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --interval 1h \
  --source alpha_vantage

# Download crypto data
poetry run python -m src.cli.unified_cli data download \
  --symbols BTC-USD ETH-USD \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --interval 1d \
  --source bybit
```

#### List Available Data Sources
```bash
poetry run python -m src.cli.unified_cli data sources
```

#### List Available Symbols
```bash
poetry run python -m src.cli.unified_cli data symbols --source yahoo_finance
poetry run python -m src.cli.unified_cli data symbols --source bybit --category spot
```

### **Portfolio Commands**

#### Test All Strategies on Portfolio
```bash
# Basic portfolio test
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/world_indices.json \
  --metric sharpe_ratio \
  --period 2y

# Advanced portfolio test with all options
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/forex.json \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --metric sortino_ratio \
  --timeframes 1d 4h 1h \
  --test-timeframes \
  --open-browser
```

#### Single Strategy Backtest
```bash
poetry run python -m src.cli.unified_cli portfolio backtest \
  --portfolio config/portfolios/crypto.json \
  --strategy rsi \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --interval 1d \
  --capital 50000
```

#### Compare Multiple Portfolios
```bash
poetry run python -m src.cli.unified_cli portfolio compare \
  --portfolios config/portfolios/forex.json config/portfolios/crypto.json \
  --metric profit_factor \
  --period 1y \
  --output reports_output/portfolio_comparison.html
```

#### Generate Investment Plan
```bash
poetry run python -m src.cli.unified_cli portfolio plan \
  --portfolio config/portfolios/world_indices.json \
  --budget 100000 \
  --risk-level moderate \
  --output reports_output/investment_plan.json
```

### **Backtest Commands**

#### Single Asset Backtest
```bash
# Test single strategy on single asset
poetry run python -m src.cli.unified_cli backtest single \
  --symbol AAPL \
  --strategy rsi \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --interval 1d \
  --capital 10000

# Test multiple strategies on single asset
poetry run python -m src.cli.unified_cli backtest multi \
  --symbol EURUSD=X \
  --strategies rsi macd bollinger_bands \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --interval 4h \
  --capital 25000
```

#### Batch Backtesting
```bash
# Test multiple assets with multiple strategies
poetry run python -m src.cli.unified_cli backtest batch \
  --symbols AAPL MSFT GOOGL AMZN \
  --strategies rsi macd bollinger_bands sma_crossover \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --interval 1d \
  --capital 10000 \
  --max-workers 4 \
  --save-trades \
  --save-equity
```

### **Optimization Commands**

#### Parameter Optimization
```bash
poetry run python -m src.cli.unified_cli optimize params \
  --symbol AAPL \
  --strategy rsi \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --interval 1d \
  --metric sharpe_ratio \
  --iterations 100

poetry run python -m src.cli.unified_cli optimize genetic \
  --symbol BTC-USD \
  --strategy macd \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --interval 1h \
  --metric profit_factor \
  --population 50 \
  --generations 100
```

#### Walk-Forward Analysis
```bash
poetry run python -m src.cli.unified_cli optimize walkforward \
  --symbol EURUSD=X \
  --strategy bollinger_bands \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --interval 1d \
  --train-months 12 \
  --test-months 3 \
  --step-months 1
```

### **Analysis Commands**

#### Performance Analysis
```bash
poetry run python -m src.cli.unified_cli analyze performance \
  --results results/backtest_AAPL_rsi.json \
  --benchmark SPY \
  --output reports_output/performance_analysis.html

poetry run python -m src.cli.unified_cli analyze compare \
  --results results/backtest_*.json \
  --metric sharpe_ratio \
  --output reports_output/strategy_comparison.html
```

#### Risk Analysis
```bash
poetry run python -m src.cli.unified_cli analyze risk \
  --portfolio config/portfolios/world_indices.json \
  --confidence-level 0.95 \
  --horizon 252 \
  --output reports_output/risk_analysis.html
```

#### Market Analysis
```bash
poetry run python -m src.cli.unified_cli analyze market \
  --symbols AAPL MSFT GOOGL \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --analysis correlation volatility returns \
  --output reports_output/market_analysis.html
```

### **Cache Commands**

#### View Cache Statistics
```bash
poetry run python -m src.cli.unified_cli cache stats
```

#### Clear Cache
```bash
# Clear all cache
poetry run python -m src.cli.unified_cli cache clear --all

# Clear specific cache type
poetry run python -m src.cli.unified_cli cache clear --type data
poetry run python -m src.cli.unified_cli cache clear --type backtest
poetry run python -m src.cli.unified_cli cache clear --type optimization

# Clear cache for specific symbol/source
poetry run python -m src.cli.unified_cli cache clear --symbol AAPL
poetry run python -m src.cli.unified_cli cache clear --source yahoo_finance

# Clear old cache (older than N days)
poetry run python -m src.cli.unified_cli cache clear --older-than 30
```

### **Reports Commands**

#### Organize Reports
```bash
# Organize existing reports into quarterly structure
poetry run python -m src.cli.unified_cli reports organize

# List quarterly reports
poetry run python -m src.cli.unified_cli reports list
poetry run python -m src.cli.unified_cli reports list --year 2024

# Get latest report for portfolio
poetry run python -m src.cli.unified_cli reports latest "Forex Portfolio"

# Cleanup old reports (keep last 8 quarters)
poetry run python -m src.cli.unified_cli reports cleanup
poetry run python -m src.cli.unified_cli reports cleanup --keep-quarters 12
```

---

## 📊 Available Metrics

### **Performance Metrics**
- `sharpe_ratio` - Risk-adjusted returns (default)
- `sortino_ratio` - Downside risk-adjusted returns
- `profit_factor` - Gross profit / gross loss
- `total_return` - Total percentage return
- `max_drawdown` - Maximum drawdown
- `calmar_ratio` - Annual return / max drawdown
- `omega_ratio` - Probability-weighted gains vs losses

### **Risk Metrics**
- `volatility` - Price volatility
- `var_95` - Value at Risk (95% confidence)
- `beta` - Market beta
- `alpha` - Market alpha

---

## ⏰ Available Timeframes

### **High Frequency**
- `1min` - 1 minute
- `5min` - 5 minutes  
- `15min` - 15 minutes
- `30min` - 30 minutes

### **Standard Frequency**
- `1h` - 1 hour
- `4h` - 4 hours
- `1d` - 1 day (default)

### **Low Frequency**
- `1wk` - 1 week
- `1M` - 1 month

---

## 📅 Time Periods

### **Preset Periods**
- `max` - All available data (2015-present)
- `10y` - Last 10 years
- `5y` - Last 5 years
- `2y` - Last 2 years
- `1y` - Last 1 year

### **Custom Periods**
Use `--start-date` and `--end-date` with format `YYYY-MM-DD`

---

## 🎯 Available Strategies (32+ Total)

### **Trend Following**
- `index_trend` - SMA crossover based
- `moving_average_trend` - 200-day MA based
- `moving_average_crossover` - Classic crossover system
- `confident_trend` - High-confidence trend following
- `face_the_train` - Strong trend following
- `lazy_trend_follower` - Minimal effort trend following
- `trend_risk_protection` - Trend following with protection

### **Mean Reversion**
- `rsi` - Relative Strength Index
- `simple_mean_reversion` - Statistical mean reversion
- `bollinger_bands` - Volatility-based mean reversion

### **Momentum**
- `macd` - Moving Average Convergence Divergence
- `mfi` - Money Flow Index
- `larry_williams_r` - Williams %R oscillator
- `ride_the_aggression` - Momentum-based strategy

### **Breakout**
- `donchian_channels` - Channel breakout system
- `turtle_trading` - Famous breakout system
- `weekly_breakout` - Time-based breakout system
- `narrow_range7` - Volatility compression breakout

### **Pattern Recognition**
- `bullish_engulfing` - Candlestick pattern strategy
- `inside_day` - Pattern-based with RSI exit
- `lower_highs_lower_lows` - Reversal after downtrend
- `kings_counting` - DeMark 9-13 sequence

### **Calendar Effects**
- `turnaround_monday` - Calendar-based reversal
- `turnaround_tuesday` - Calendar-based reversal
- `russell_rebalancing` - Index rebalancing based

### **Asset-Specific**
- `bitcoin` - Cryptocurrency-specific strategy
- `crude_oil` - Commodity-specific strategy

### **Statistical**
- `linear_regression` - Statistical approach
- `stan_weinstein_stage2` - Market stage analysis

### **Risk Management**
- `pullback_trading` - Buying dips in trends
- `counter_punch` - Reversal trading strategy

### **Technical Indicators**
- `adx` - Average Directional Index
- `sma_crossover` - Simple Moving Average crossover

---

## 📁 Portfolio Configuration

### **Available Portfolios**

#### 💱 **Forex Portfolio**
```json
{
  "forex": {
    "name": "Forex Portfolio",
    "symbols": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", ...],
    "benchmark": "EURUSD=X",
    "data_sources": {
      "primary": ["polygon", "alpha_vantage", "twelve_data"],
      "fallback": ["finnhub", "yahoo_finance"]
    }
  }
}
```

#### ₿ **Crypto Portfolio**
```json
{
  "crypto": {
    "name": "Crypto Portfolio", 
    "symbols": ["BTC-USD", "ETH-USD", "ADA-USD", ...],
    "benchmark": "BTC-USD",
    "data_sources": {
      "primary": ["bybit", "polygon", "twelve_data"],
      "fallback": ["alpha_vantage", "tiingo", "yahoo_finance"]
    }
  }
}
```

#### 🌍 **World Indices Portfolio**
```json
{
  "world_indices": {
    "name": "World Indices Portfolio",
    "symbols": ["SPY", "VTI", "QQQ", "IWM", "EFA", "VEA", "EEM", "VWO"],
    "benchmark": "SPY",
    "data_sources": {
      "primary": ["polygon", "twelve_data", "alpha_vantage"],
      "fallback": ["yahoo_finance", "tiingo", "pandas_datareader"]
    }
  }
}
```

#### 🥇 **Commodities Portfolio**
```json
{
  "commodities": {
    "name": "Commodities Portfolio",
    "symbols": ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", ...],
    "benchmark": "DBC",
    "data_sources": {
      "primary": ["polygon", "alpha_vantage", "twelve_data"],
      "fallback": ["yahoo_finance", "tiingo", "pandas_datareader"]
    }
  }
}
```

#### 🏛️ **Bonds Portfolio**
```json
{
  "bonds": {
    "name": "Bonds Portfolio",
    "symbols": ["TLT", "IEF", "SHY", "LQD", "HYG", "EMB", ...],
    "benchmark": "AGG",
    "data_sources": {
      "primary": ["polygon", "alpha_vantage", "twelve_data"],
      "fallback": ["yahoo_finance", "tiingo", "pandas_datareader"]
    }
  }
}
```

---

## 🔧 Environment Configuration

### **Required API Keys**
```bash
# Premium data sources (optional but recommended)
export POLYGON_API_KEY="your_polygon_key"
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key" 
export TWELVE_DATA_API_KEY="your_twelve_data_key"
export TIINGO_API_KEY="your_tiingo_key"
export FINNHUB_API_KEY="your_finnhub_key"

# For crypto futures trading
export BYBIT_API_KEY="your_bybit_key"
export BYBIT_API_SECRET="your_bybit_secret"
export BYBIT_TESTNET="false"
```

### **System Configuration**
```bash
# Logging level
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR

# Cache settings
export CACHE_ENABLED="true"
export CACHE_TTL_HOURS="24"

# Performance settings
export MAX_WORKERS="4"
export MEMORY_LIMIT_GB="8"
```

---

## 📈 Output Formats

### **Report Types**
- **HTML Reports** - Interactive reports with charts (default)
- **JSON Results** - Raw backtest results
- **CSV Data** - Tabular data export
- **PDF Reports** - Static printable reports

### **Report Locations**
- **Quarterly Structure**: `reports_output/{YEAR}/Q{QUARTER}/`
- **Naming Convention**: `{portfolio_name}_Q{quarter}_{year}.html`
- **Compressed Versions**: `.html.gz` files for storage efficiency

### **Chart Features**
- **Interactive Plotly Charts** - Zoomable, hoverable equity curves
- **Benchmark Comparison** - Strategy vs benchmark performance
- **Multiple Timeframes** - Switch between different timeframe results
- **KPI Dashboard** - Key performance indicators
- **Order History** - Detailed trade analysis

---

## 🚀 Advanced Usage

### **Parallel Processing**
```bash
# Enable parallel backtesting
poetry run python -m src.cli.unified_cli backtest batch \
  --symbols AAPL MSFT GOOGL AMZN TSLA \
  --strategies rsi macd bollinger_bands \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --max-workers 8 \
  --memory-limit 16
```

### **Custom Strategy Parameters**
```bash
# Override default strategy parameters
poetry run python -m src.cli.unified_cli backtest single \
  --symbol AAPL \
  --strategy rsi \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --params '{"rsi_period": 21, "overbought": 75, "oversold": 25}'
```

### **Multi-Asset Class Analysis**
```bash
# Analyze across multiple asset classes
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/multi_asset.json \
  --metric sharpe_ratio \
  --period max \
  --test-timeframes \
  --asset-types stocks forex crypto commodities \
  --correlation-analysis \
  --risk-parity \
  --open-browser
```

---

## 🛠️ Development Commands

### **Testing**
```bash
# Run all tests
poetry run python -m pytest tests/

# Run specific test categories
poetry run python -m pytest tests/core/
poetry run python -m pytest tests/integration/
poetry run python -m pytest tests/cli/

# Run with coverage
poetry run python -m pytest tests/ --cov=src --cov-report=html
```

### **Code Quality**
```bash
# Run linting
poetry run ruff check src/
poetry run ruff format src/

# Run type checking
poetry run mypy src/

# Run pre-commit hooks
poetry run pre-commit run --all-files
```

### **Docker Commands**
```bash
# Build and run with Docker
docker-compose up --build

# Run specific service
docker-compose run app poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/forex.json \
  --metric sharpe_ratio \
  --period max
```

---

## 📚 Additional Resources

- **[Data Sources Guide](DATA_SOURCES_GUIDE.md)** - Complete data source documentation
- **[Symbol Transformation Guide](SYMBOL_TRANSFORMATION_GUIDE.md)** - Symbol format conversion
- **[Testing Guide](TESTING_GUIDE.md)** - Comprehensive testing documentation
- **[Docker Guide](DOCKER_GUIDE.md)** - Docker deployment instructions
- **[Production Ready Guide](PRODUCTION_READY.md)** - Production deployment guide

---

## 🎯 Most Common Workflows

### **1. Daily Portfolio Analysis**
```bash
# Quick daily analysis of all portfolios
for portfolio in forex crypto world_indices commodities bonds; do
  poetry run python -m src.cli.unified_cli portfolio test-all \
    --portfolio config/portfolios/${portfolio}.json \
    --metric sharpe_ratio \
    --period 1y \
    --open-browser
done
```

### **2. Strategy Development**
```bash
# Test new strategy across multiple assets
poetry run python -m src.cli.unified_cli backtest batch \
  --symbols AAPL EURUSD=X BTC-USD GLD \
  --strategies new_strategy \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --interval 1d \
  --save-trades \
  --save-equity
```

### **3. Risk Assessment**
```bash
# Comprehensive risk analysis
poetry run python -m src.cli.unified_cli analyze risk \
  --portfolio config/portfolios/world_indices.json \
  --confidence-level 0.95 \
  --horizon 252 \
  --monte-carlo-sims 10000 \
  --output reports_output/risk_assessment.html
```

This guide covers all available features and commands in the quant trading system! 🚀
