# Data Sources Guide

## Overview
The quant system now supports multiple data sources with intelligent routing based on asset type, data quality, and coverage. Each portfolio is configured with optimal data sources for its specific asset class.

## Available Data Sources

### 📊 **Stock & ETF Sources**
| Source | Coverage | Quality | Rate Limit | Best For |
|--------|----------|---------|------------|----------|
| **Polygon.io** | 1970-present | Excellent | 5 req/sec | US stocks, real-time |
| **Twelve Data** | 1990-present | Excellent | 8 req/min | Global stocks, ETFs |
| **Alpha Vantage** | 1999-present | Very Good | 5 req/min | US/international stocks |
| **Tiingo** | 1990-present | Very Good | 1000 req/hr | US stocks, ETFs |
| **Yahoo Finance** | 1970-present | Good | No limit | Fallback, global coverage |
| **Pandas DataReader** | Varies | Good | No limit | Backup, FRED data |

### 💱 **Forex Sources**
| Source | Coverage | Quality | Rate Limit | Best For |
|--------|----------|---------|------------|----------|
| **Polygon.io** | 2004-present | Excellent | 5 req/sec | Major pairs, high frequency |
| **Alpha Vantage** | 1999-present | Excellent | 5 req/min | All major pairs |
| **Twelve Data** | 2000-present | Very Good | 8 req/min | Global forex pairs |
| **Finnhub** | 2010-present | Good | 60 req/min | OANDA forex data |
| **Yahoo Finance** | 2003-present | Good | No limit | Fallback |

### ₿ **Crypto Sources**
| Source | Coverage | Quality | Rate Limit | Best For |
|--------|----------|---------|------------|----------|
| **Bybit** | 2018-present | Excellent | 10 req/sec | Futures, spot trading |
| **Polygon.io** | 2017-present | Excellent | 5 req/sec | Major cryptos |
| **Twelve Data** | 2017-present | Very Good | 8 req/min | Wide crypto coverage |
| **Alpha Vantage** | 2017-present | Good | 5 req/min | Popular cryptos |
| **Tiingo** | 2017-present | Good | 1000 req/hr | Major cryptos |

### 🏛️ **Bonds & Economic Data**
| Source | Coverage | Quality | Rate Limit | Best For |
|--------|----------|---------|------------|----------|
| **Pandas DataReader** | 1954-present | Excellent | No limit | FRED economic data |
| **Polygon.io** | 2003-present | Excellent | 5 req/sec | Bond ETFs |
| **Alpha Vantage** | 2003-present | Very Good | 5 req/min | Treasury data |
| **Twelve Data** | 2003-present | Good | 8 req/min | Bond indices |

## Portfolio Configurations

### 🌍 **World Indices Portfolio**
```json
"data_sources": {
  "primary": ["polygon", "twelve_data", "alpha_vantage"],
  "fallback": ["yahoo_finance", "tiingo", "pandas_datareader"]
}
```
- **Best for**: US and international equity indices
- **Coverage**: 1990-present
- **Quality**: Excellent

### 💱 **Forex Portfolio** 
```json
"data_sources": {
  "primary": ["polygon", "alpha_vantage", "twelve_data"],
  "fallback": ["finnhub", "yahoo_finance"],
  "excluded": ["bybit"]
}
```
- **Best for**: Major forex pairs
- **Coverage**: 2000-present  
- **Quality**: Excellent
- **Note**: Bybit excluded (not suitable for forex)

### ₿ **Crypto Portfolio**
```json
"data_sources": {
  "primary": ["bybit", "polygon", "twelve_data"],
  "fallback": ["alpha_vantage", "tiingo", "yahoo_finance"],
  "excluded": ["finnhub"]
}
```
- **Best for**: Cryptocurrency trading
- **Coverage**: 2017-present
- **Quality**: Excellent
- **Note**: Bybit prioritized for crypto futures

### 🥇 **Commodities Portfolio**
```json
"data_sources": {
  "primary": ["polygon", "alpha_vantage", "twelve_data"],
  "fallback": ["yahoo_finance", "tiingo", "pandas_datareader"],
  "excluded": ["bybit"]
}
```
- **Best for**: Commodity ETFs and futures
- **Coverage**: 2006-present
- **Quality**: Good to Excellent

### 🏛️ **Bonds Portfolio**
```json
"data_sources": {
  "primary": ["polygon", "alpha_vantage", "twelve_data"],
  "fallback": ["yahoo_finance", "tiingo", "pandas_datareader"],
  "economic_data": ["pandas_datareader"],
  "excluded": ["bybit", "finnhub"]
}
```
- **Best for**: Government and corporate bonds
- **Coverage**: 2003-present
- **Quality**: Excellent

## Environment Variables

Set these API keys for premium data access:

```bash
# Required for enhanced functionality
export POLYGON_API_KEY="your_polygon_key"
export ALPHA_VANTAGE_API_KEY="your_av_key"
export TWELVE_DATA_API_KEY="your_twelve_data_key"

# Optional but recommended
export TIINGO_API_KEY="your_tiingo_key"
export FINNHUB_API_KEY="your_finnhub_key"

# For crypto futures trading
export BYBIT_API_KEY="your_bybit_key"
export BYBIT_API_SECRET="your_bybit_secret"
export BYBIT_TESTNET="false"
```

## Data Source Priority

The system automatically selects data sources based on:

1. **Asset Type Compatibility**: Forex sources for forex, crypto sources for crypto
2. **Data Quality**: Higher quality sources prioritized
3. **Coverage Period**: Sources with longer history preferred
4. **Rate Limits**: Balanced to avoid hitting limits
5. **API Key Availability**: Falls back to free sources if keys not available

## Free vs Premium Sources

### Free Sources (No API Key Required)
- **Yahoo Finance**: Good fallback for all asset types
- **Pandas DataReader**: FRED economic data, backup for stocks

### Premium Sources (API Key Required)
- **Polygon.io**: Best overall quality, requires paid plan for production
- **Alpha Vantage**: Free tier available (5 calls/min), premium for more
- **Twelve Data**: Free tier available (8 calls/min), premium for real-time
- **Tiingo**: Free tier available (1000 calls/day), premium for more
- **Finnhub**: Free tier available (60 calls/min), premium for more
- **Bybit**: Free for crypto trading, requires account

## Testing Commands

### Test Forex Portfolio (All 32+ Strategies)
```bash
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/forex.json \
  --metric sharpe_ratio \
  --period max \
  --test-timeframes
```

### Test Crypto Portfolio (All 32+ Strategies)
```bash
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/crypto.json \
  --metric sortino_ratio \
  --period max \
  --test-timeframes
```

### Test Commodities Portfolio (All 32+ Strategies)
```bash
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/commodities.json \
  --metric profit_factor \
  --period max \
  --test-timeframes
```

### Test Bonds Portfolio (All 32+ Strategies)
```bash
poetry run python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/bonds.json \
  --metric sharpe_ratio \
  --period max \
  --test-timeframes
```

## Available Strategies (32+ Total)

The system automatically discovers and tests all available strategies:

**Trend Following**: ADX, Confident Trend, Face the Train, Index Trend, Lazy Trend Follower, Moving Average Crossover, Moving Average Trend, Trend Risk Protection

**Mean Reversion**: Simple Mean Reversion, RSI, Bollinger Bands

**Momentum**: MACD, MFI, Larry Williams %R, Ride the Aggression

**Breakout**: Donchian Channels, Turtle Trading, Weekly Breakout, Narrow Range 7

**Pattern Recognition**: Bullish Engulfing, Inside Day, Lower Highs Lower Lows, Kings Counting

**Calendar Effects**: Turnaround Monday, Turnaround Tuesday, Russell Rebalancing

**Asset-Specific**: Bitcoin Strategy, Crude Oil Strategy

**Statistical**: Linear Regression, Stan Weinstein Stage 2

**Risk Management**: Pullback Trading, Counter Punch

And more! All strategies are automatically loaded and tested.

## Data Quality Matrix

| Asset Type | Primary Sources | Historical Depth | Intraday Support | Real-time |
|------------|----------------|-------------------|------------------|-----------|
| **Stocks** | Polygon, Twelve Data, Alpha Vantage | 1970+ | ✅ | ✅ |
| **Forex** | Polygon, Alpha Vantage, Twelve Data | 1999+ | ✅ | ✅ |
| **Crypto** | Bybit, Polygon, Twelve Data | 2017+ | ✅ | ✅ |
| **Commodities** | Polygon, Alpha Vantage, Twelve Data | 2006+ | ✅ | ✅ |
| **Bonds** | Polygon, Alpha Vantage, FRED | 2003+ | ✅ | ✅ |

The system now provides enterprise-grade data coverage with intelligent fallbacks and optimal source selection for each asset class! 🚀
