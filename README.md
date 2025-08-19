# Quant System

A comprehensive quantitative backtesting system built for institutional-grade collection analysis. Docker-based setup with production-grade features for analyzing stocks, bonds, crypto, forex, and commodities across global markets.

## 🚀 Quick Start

### Docker Setup (Only Way)

```bash
# Clone repository
git clone <repository-url>
cd quant-system

# Build and run
docker-compose up --build

# Run portfolio backtest (stocks example)
docker-compose run --rm quant python -m src.cli.unified_cli portfolio backtest \
  --symbols AAPL MSFT TSLA \
  --strategy BuyAndHold \
  --start-date 2023-01-01 \
  --end-date 2024-12-31

# Run bond portfolio analysis
docker-compose run --rm quant python -m src.cli.unified_cli portfolio backtest \
  --symbols TLT IEF SHY \
  --strategy MeanReversion \
  --start-date 2023-01-01 \
  --end-date 2024-12-31

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

### Portfolio Commands
```bash
# Portfolio backtests with real data
docker-compose run --rm quant python -m src.cli.unified_cli portfolio backtest \
  --symbols AAPL MSFT TSLA \
  --strategy BuyAndHold \
  --start-date 2023-01-01 \
  --end-date 2024-12-31

# Test all strategies and timeframes
docker-compose run --rm quant python -m src.cli.unified_cli portfolio test-all \
  --symbols TLT IEF SHY \
  --start-date 2023-01-01 \
  --end-date 2024-12-31

# Get best performing strategies
docker-compose run --rm quant python -m src.cli.unified_cli portfolio best \
  --limit 10

# AI-powered portfolio recommendations
docker-compose run --rm quant python -m src.cli.unified_cli ai portfolio_recommend \
  --portfolio config/collections/bonds.json \
  --risk-tolerance moderate
```

### Data Management
```bash
# Cache statistics
docker-compose run --rm quant python -m src.cli.unified_cli cache stats

# Clear cache
docker-compose run --rm quant python -m src.cli.unified_cli cache clear
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

### AI Recommendations
```bash
# Generate AI portfolio recommendations
docker-compose run --rm quant python -m src.cli.unified_cli ai portfolio_recommend \
  --portfolio config/collections/stocks.json \
  --risk-tolerance moderate

# Export recommendations to JSON
docker-compose run --rm quant python -m src.cli.unified_cli ai export \
  --quarter Q3 --year 2025
```



### CSV Data Export
```bash
# Export best strategies by quarter (organized by year/quarter/collection)
docker-compose run --rm quant python -m src.cli.unified_cli reports export-csv \
  --format best-strategies --quarter Q3 --year 2025

# Export full quarterly data
docker-compose run --rm quant python -m src.cli.unified_cli reports export-csv \
  --format quarterly --quarter Q3 --year 2025
```

### TradingView Alert Export
```bash
# Auto-organized by quarter/year (recommended)
docker-compose run --rm quant python -m src.cli.unified_cli reports export-tradingview \
  --quarter Q3 --year 2025

# Export for specific collection
docker-compose run --rm quant python -m src.cli.unified_cli reports export-tradingview \
  --collection stocks --quarter Q3 --year 2025
```

**Alert Format**: Includes strategy, timeframe, Sortino ratio, profit metrics, and TradingView placeholders like `{{close}}`, `{{timenow}}`, `{{strategy.order.action}}`.

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
