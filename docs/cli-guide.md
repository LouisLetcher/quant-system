# CLI Reference

Complete command-line interface reference for the Quant Trading System.

## Quick Start

```bash
# Use Docker (recommended)
docker-compose run --rm quant bash

# List available collections
docker-compose run --rm quant python -m src.cli.unified_cli backtest list-collections

# Run backtest on collection
docker-compose run --rm quant python -m src.cli.unified_cli backtest batch \
  --collection config/collections/stocks.json --metric sortino_ratio
```

## Command Structure

```
docker-compose run --rm quant python -m src.cli.unified_cli <category> <command> [options]
```

## Backtest Commands

### Single Asset Backtest
```bash
docker-compose run --rm quant python -m src.cli.unified_cli backtest single \
  --symbol AAPL --strategy BuyAndHold --start-date 2023-01-01

Options:
  --metric METRIC        Performance metric (sortino_ratio, sharpe_ratio, calmar_ratio)
  --timeframe PERIOD     Data resolution (1d, 1h, 15m, 5m, 1m)
  --end-date DATE        End date for backtest
  --initial-capital NUM  Starting capital (default: 10000)
```

### Collection Batch Backtest
```bash
docker-compose run --rm quant python -m src.cli.unified_cli backtest batch \
  --collection config/collections/stocks.json --metric sortino_ratio

Options:
  --strategy STRATEGY    Trading strategy to use
  --timeframes LIST      Multiple timeframes to test
  --save-trades         Save individual trades to database
```

## AI Recommendation Commands

### Generate Portfolio Recommendations
```bash
docker-compose run --rm quant python -m src.cli.unified_cli ai portfolio_recommend \
  --portfolio config/collections/bonds.json --risk-tolerance moderate

Options:
  --risk-tolerance LEVEL Risk level (conservative, moderate, aggressive)
  --max-assets NUM       Maximum assets to recommend
  --quarter QUARTER      Target quarter (Q1, Q2, Q3, Q4)
  --year YEAR           Target year
```

## Data Commands

### Download Data
```bash
python -m src.cli.unified_cli data download --symbols AAPL,GOOGL [options]

Options:
  --symbols SYMBOLS      Comma-separated symbols
  --start-date DATE      Start date (YYYY-MM-DD)
  --end-date DATE        End date (YYYY-MM-DD)
  --source SOURCE        Data source (yahoo, alpha_vantage, etc.)
```

## Cache Commands

### Cache Statistics
```bash
python -m src.cli.unified_cli cache stats
```

### Clear Cache
```bash
python -m src.cli.unified_cli cache clear [--all] [--symbol SYMBOL]
```

## Report Commands

### Generate Reports
```bash
python -m src.cli.unified_cli reports generate <portfolio> [options]

Options:
  --format FORMAT        Output format (html, pdf, json)
  --period PERIOD        Analysis period
  --output-dir DIR       Output directory
```

### Organize Reports
```bash
python -m src.cli.unified_cli reports organize
```

### Export to CSV
```bash
python -m src.cli.unified_cli reports export-csv [options]

Options:
  --portfolio FILE       Portfolio config file
  --output FILE          Output CSV filename
  --format FORMAT        Export format (full, best-strategies, quarterly)
  --quarter QUARTER      Quarter (Q1, Q2, Q3, Q4)
  --year YEAR           Year (YYYY)
```

## AI Recommendation Commands

### Generate AI Investment Recommendations
```bash
python -m src.cli.unified_cli ai recommend [options]

Options:
  --risk-tolerance LEVEL    Risk level (conservative, moderate, aggressive)
  --max-assets N           Maximum assets to recommend (default: 10)
  --min-confidence SCORE   Minimum confidence (0-1, default: 0.7)
  --quarter QUARTER        Specific quarter (e.g., Q3_2025)
  --output FILE            Save to file
  --format FORMAT          Output format (table, json, summary)
```

### Compare Assets
```bash
python -m src.cli.unified_cli ai compare SYMBOL1 SYMBOL2 [SYMBOL3...] [options]

Options:
  --strategy STRATEGY      Filter by specific strategy
```

### Explain Recommendation
```bash
python -m src.cli.unified_cli ai explain SYMBOL STRATEGY
```

## Examples

### Test Crypto Portfolio
```bash
# Using Sortino ratio (default - superior to Sharpe)
python -m src.cli.unified_cli portfolio test crypto \
  --metric sortino_ratio \
  --period 1y \
  --test-timeframes \
  --open-browser

# Traditional Sharpe ratio (for comparison)
python -m src.cli.unified_cli portfolio test crypto \
  --metric sharpe_ratio \
  --period 1y
```

### Download Forex Data
```bash
python -m src.cli.unified_cli data download \
  --symbols EURUSD=X,GBPUSD=X \
  --start-date 2023-01-01 \
  --source twelve_data
```

### Get AI Investment Recommendations
```bash
# Conservative portfolio recommendations
python -m src.cli.unified_cli ai recommend --risk-tolerance conservative --max-assets 5

# Compare specific crypto assets
python -m src.cli.unified_cli ai compare BTCUSDT ETHUSDT ADAUSDT --strategy rsi

# Export recommendations to file
python -m src.cli.unified_cli ai recommend --output exports/recommendations/Q3_2025.json --format json
```

### Export Portfolio Data
```bash
# Export best strategies to CSV
python -m src.cli.unified_cli reports export-csv --format best-strategies --output exports/best_strategies.csv

# Export quarterly summary
python -m src.cli.unified_cli reports export-csv --format quarterly --quarter Q3 --year 2025
```

### Daily Workflow
```bash
# Check cache status
python -m src.cli.unified_cli cache stats

# Test all portfolios (Sortino ratio default)
python -m src.cli.unified_cli portfolio test-all --metric sortino_ratio --period 1d --open-browser

# Get AI recommendations
python -m src.cli.unified_cli ai recommend --risk-tolerance moderate

# Organize reports
python -m src.cli.unified_cli reports organize
```

## Configuration

Set environment variables in `.env`:
```bash
LOG_LEVEL=INFO
CACHE_ENABLED=true
DEFAULT_PERIOD=1y
BROWSER_AUTO_OPEN=true
```

## Help

Get help for any command:
```bash
python -m src.cli.unified_cli --help
python -m src.cli.unified_cli portfolio --help
python -m src.cli.unified_cli portfolio test --help
```
