# Quant System

A comprehensive quantitative backtesting system built for institutional-grade collection analysis. Docker-based setup with production-grade features for analyzing stocks, bonds, crypto, forex, and commodities across global markets.

## 🚀 Quick Start

### Docker Setup (Only Way)

```bash
# Clone repository
git clone <repository-url>
cd quant-system

# Start PostgreSQL database and services
docker-compose up -d postgres pgadmin

# Build and run main system
docker-compose build quant
docker-compose run --rm quant python -m src.cli.unified_cli --help

# Run comprehensive collection backtesting
docker-compose run --rm quant python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/collections/bonds.json \
  --metric sortino_ratio \
  --period max

# Interactive shell
docker-compose run --rm quant bash
```

## 📊 Features

### Core Capabilities
- **Multi-Asset Support**: Stocks, bonds, crypto, forex, commodities via multiple data sources
- **AI Investment Recommendations**: Performance-based portfolio optimization with confidence scoring
- **Backtesting Library Integration**: Direct integration with `backtesting` library for institutional-grade performance analysis
- **Portfolio Analysis**: Risk-adjusted returns, correlation analysis, drawdown attribution
- **Data Integration**: PostgreSQL storage with Yahoo Finance, Bybit, Alpha Vantage APIs
- **Report Generation**: Automated quarterly HTML reports, CSV exports, TradingView alerts

### Data Sources by Asset Class
- **Stocks/Bonds**: Yahoo Finance (primary), Alpha Vantage (fallback)
- **Crypto**: Bybit (primary), Yahoo Finance (fallback)
- **Forex**: Alpha Vantage, Twelve Data, Polygon.io
- **Commodities**: Yahoo Finance, Tiingo

## 🏗️ Architecture

```
quant-system/
├── src/                     # Core source code
│   ├── core/               # Trading logic & backtesting
│   ├── cli/                # Command-line interface
│   └── utils/              # Utilities & data management
├── config/collections/     # Asset collections (stocks, bonds, crypto, forex)
├── exports/               # Organized exports (reports/alerts by quarter)
├── cache/                 # Data cache (Docker mounted)
└── logs/                  # System logs (Docker mounted)
```

## 📈 Usage

### Portfolio Management
```bash
# Comprehensive collection testing (generates HTML reports + database data)
docker-compose run --rm quant python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/collections/bonds.json \
  --metric sortino_ratio \
  --period max

# Single symbol backtest
docker-compose run --rm quant python -m src.cli.unified_cli portfolio backtest \
  --symbols TLT IEF SHY \
  --strategy BuyAndHold \
  --start-date 2023-01-01 \
  --end-date 2024-12-31

# Compare multiple portfolios
docker-compose run --rm quant python -m src.cli.unified_cli portfolio compare \
  config/collections/stocks_traderfox_us_tech.json config/collections/bonds.json

# Generate investment plan based on backtest results
docker-compose run --rm quant python -m src.cli.unified_cli portfolio plan \
  --portfolio config/collections/bonds.json
```

### AI Investment Recommendations
```bash
# Generate AI portfolio recommendations (creates markdown + HTML reports)
docker-compose run --rm quant python -m src.cli.unified_cli ai portfolio_recommend \
  --portfolio config/collections/bonds.json \
  --risk-tolerance moderate

# Get specific recommendations by quarter
docker-compose run --rm quant python -m src.cli.unified_cli ai recommend \
  --quarter Q3_2025 --risk-tolerance aggressive

# Explain asset recommendations
docker-compose run --rm quant python -m src.cli.unified_cli ai explain \
  --symbol TLT --timeframe 1d
```

### Data Management
```bash
# Download market data for collections
docker-compose run --rm quant python -m src.cli.unified_cli data download \
  --symbols TLT IEF SHY --asset-type bonds

# Show available data sources
docker-compose run --rm quant python -m src.cli.unified_cli data sources

# List symbols by asset type
docker-compose run --rm quant python -m src.cli.unified_cli data symbols --asset-type bonds

# Cache management
docker-compose run --rm quant python -m src.cli.unified_cli cache stats
docker-compose run --rm quant python -m src.cli.unified_cli cache clear --older-than-days 30
```

### Strategy Development
```bash
# List available strategies
docker-compose run --rm quant python -m src.cli.unified_cli strategy list

# Get strategy details
docker-compose run --rm quant python -m src.cli.unified_cli strategy info --strategy BuyAndHold

# Test custom strategy
docker-compose run --rm quant python -m src.cli.unified_cli strategy test \
  --strategy-file external_strategies/my_strategy.py
```

### Optimization
```bash
# Optimize single strategy parameters
docker-compose run --rm quant python -m src.cli.unified_cli optimize single \
  --symbol TLT --strategy RSI --method genetic --iterations 100

# Batch optimization across multiple symbols
docker-compose run --rm quant python -m src.cli.unified_cli optimize batch \
  --symbols TLT IEF SHY --strategies RSI BollingerBands --workers 4
```

### Analysis & Reporting
```bash
# Generate comprehensive analysis reports
docker-compose run --rm quant python -m src.cli.unified_cli analyze report \
  --portfolio config/collections/bonds.json

# Compare strategy performance
docker-compose run --rm quant python -m src.cli.unified_cli analyze compare \
  --symbols TLT IEF SHY --strategies BuyAndHold RSI
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# PostgreSQL Database (primary storage)
DATABASE_URL=postgresql://quantuser:quantpass@localhost:5432/quant_system

# Optional API keys for enhanced data access
ALPHA_VANTAGE_API_KEY=your_key
TWELVE_DATA_API_KEY=your_key
POLYGON_API_KEY=your_key
TIINGO_API_KEY=your_key
FINNHUB_API_KEY=your_key
```

### Collection Examples (config/collections/)

#### Stocks Collection
```json
{
  "name": "US Large Cap Stocks",
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
  "data_sources": {
    "primary": ["yahoo_finance"],
    "fallback": ["alpha_vantage"]
  }
}
```

#### Bonds Collection
```json
{
  "name": "US Treasury Bonds",
  "symbols": ["TLT", "IEF", "SHY", "TIPS"],
  "data_sources": {
    "primary": ["yahoo_finance"],
    "fallback": ["alpha_vantage"]
  }
}
```

#### Crypto Collection
```json
{
  "name": "Crypto Portfolio",
  "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
  "data_sources": {
    "primary": ["bybit", "yahoo_finance"],
    "fallback": ["alpha_vantage"]
  }
}
```

## 📊 Performance Metrics

**Primary Metric: Sortino Ratio** (default)

**Why Sortino over Sharpe:**
- **Sortino** only penalizes **downside volatility** (what investors actually care about)
- **Sharpe** penalizes all volatility, including upside moves (which aren't really "risk")
- **Hedge funds prefer Sortino** because upside volatility is desirable

**Metric Hierarchy for Quantitative Analysis:**
1. **Sortino Ratio** (primary) - Downside risk-adjusted returns
2. **Calmar Ratio** (secondary) - Annual return / Max drawdown
3. **Sharpe Ratio** (tertiary) - Traditional risk-adjusted returns
4. **Profit Factor** (supplementary) - Gross profit/loss ratio

**Additional Analysis:**
- **Drawdown Analysis**: Maximum drawdown, recovery periods
- **Volatility**: Standard deviation, downside deviation
- **Efficiency**: Win rate, risk-reward ratios

## 🧪 Testing

```bash
# Run tests in Docker
docker-compose run --rm quant pytest
```

## 📊 Export & Reporting

### Export & Reporting
```bash
# Export collection-specific CSV data from database (quarterly summary)
docker-compose run --rm quant python -m src.cli.unified_cli reports export-csv \
  --portfolio config/collections/bonds.json --format quarterly \
  --quarter Q3 --year 2025

# Export best strategies by quarter for all collections
docker-compose run --rm quant python -m src.cli.unified_cli reports export-csv \
  --format best-strategies --quarter Q3 --year 2025

# Export TradingView alerts with proper naming convention
docker-compose run --rm quant python -m src.utils.tradingview_alert_exporter \
  --output bonds_collection_tradingview_alerts_Q3_2025

# Generate AI investment recommendations (markdown format)
docker-compose run --rm quant python -m src.cli.unified_cli ai portfolio_recommend \
  --portfolio config/collections/bonds.json \
  --risk-tolerance moderate

# Organize reports by quarter/year
docker-compose run --rm quant python -m src.cli.unified_cli reports organize

# List available reports
docker-compose run --rm quant python -m src.cli.unified_cli reports list

# Get latest report for portfolio
docker-compose run --rm quant python -m src.cli.unified_cli reports latest \
  --portfolio config/collections/bonds.json
```

### Validation & Testing
```bash
# Validate strategy metrics against backtesting library
docker-compose run --rm quant python -m src.cli.unified_cli validate strategy \
  --symbol TLT --strategy BuyAndHold

# Batch validate multiple strategies
docker-compose run --rm quant python -m src.cli.unified_cli validate batch \
  --symbols TLT IEF SHY --strategies BuyAndHold RSI
```

**TradingView Alert Format**: Includes strategy, timeframe, Sortino ratio, profit metrics, and placeholders like `{{close}}`, `{{timenow}}`, `{{strategy.order.action}}`.

## 📁 Output & Storage

**PostgreSQL Database (Primary Storage):**
- Market data with optimized indexes for Sortino analysis
- Backtest results with comprehensive performance metrics
- Portfolio configurations with Sortino-first optimization

**Local Files (Organized by Quarter/Year):**
- `exports/reports/YYYY/QX/` - HTML portfolio reports by collection
- `exports/csv/YYYY/QX/` - CSV data exports by collection
- `exports/tradingview_alerts/YYYY/QX/` - TradingView alert exports
- `exports/recommendations/YYYY/QX/` - AI recommendation JSON exports
- `cache/` - Temporary files and quick access data
- `logs/` - System logs

## 🔒 Security

- Environment variable-based API key management
- Docker containerization for isolation
- No external database dependencies for basic usage

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

**⚠️ Disclaimer**: Educational purposes only. Not financial advice. Trade responsibly.
