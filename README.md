# Quant Trading System

A lightweight quantitative trading and backtesting system built for local report generation. Docker-based setup with production-grade features for analyzing crypto, stocks, and other financial instruments.

## 🚀 Quick Start

### Docker Setup (Only Way)

```bash
# Clone repository
git clone <repository-url>
cd quant-system

# Build and run
docker-compose up --build

# Run crypto backtest
docker-compose run --rm quant python -m src.cli.unified_cli portfolio backtest \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --strategy BuyAndHold \
  --start-date 2023-01-01 \
  --end-date 2024-12-31

# Interactive shell
docker-compose run --rm quant bash
```

## 📊 Features

### Core Capabilities
- **Multi-Asset Support**: Crypto, stocks, forex via multiple data sources
- **Backtesting Engine**: Performance analysis with comprehensive metrics
- **Portfolio Analysis**: Risk metrics, drawdown analysis, return attribution
- **Data Integration**: Yahoo Finance, Bybit, Alpha Vantage with fallback support
- **Report Generation**: Automated export to local `/exports` folder

### Data Sources
- **Primary**: Yahoo Finance, Bybit (crypto)
- **Fallback**: Alpha Vantage, Twelve Data, Polygon.io, Tiingo

## 🏗️ Architecture

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

## 📈 Usage

### Backtest Commands
```bash
# Crypto portfolio backtest
docker-compose run --rm quant python -m src.cli.unified_cli portfolio backtest \
  --symbols BTCUSDT ETHUSDT \
  --strategy BuyAndHold \
  --start-date 2023-01-01 \
  --end-date 2024-12-31

# Pre-configured crypto portfolio (220+ symbols)
docker-compose run --rm quant python -m src.cli.unified_cli portfolio test-all \
  --portfolio config/portfolios/crypto.json \
  --metric sortino_ratio \
  --period max
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

### Portfolio Example (config/portfolios/)
```json
{
  "name": "Crypto Portfolio",
  "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
  "initial_capital": 10000,
  "commission": 0.001,
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

## 📁 Output & Storage

**PostgreSQL Database (Primary Storage):**
- Market data with optimized indexes for Sortino analysis
- Backtest results with comprehensive performance metrics
- Portfolio configurations with Sortino-first optimization

**Local Files:**
- `/exports` - Generated reports and analysis
- `/cache` - Temporary files and quick access data  
- `/logs` - System logs

## 🔒 Security

- Environment variable-based API key management
- Docker containerization for isolation
- No external database dependencies for basic usage

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

**⚠️ Disclaimer**: Educational purposes only. Not financial advice. Trade responsibly.
