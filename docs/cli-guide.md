# CLI Reference

Complete command-line interface reference for the Quant Trading System.

## Quick Start

```bash
# Activate environment
poetry shell

# List available portfolios
python -m src.cli.unified_cli portfolio list

# Test a portfolio
python -m src.cli.unified_cli portfolio test crypto --open-browser
```

## Command Structure

```
python -m src.cli.unified_cli <category> <command> [options]
```

## Portfolio Commands

### List Portfolios
```bash
python -m src.cli.unified_cli portfolio list
```

### Test Portfolio
```bash
python -m src.cli.unified_cli portfolio test <name> [options]

Options:
  --metric METRIC        Performance metric (sharpe_ratio, sortino_ratio)
  --period PERIOD        Time period (1d, 1w, 1m, 3m, 6m, 1y, max)
  --test-timeframes      Test multiple timeframes
  --open-browser         Auto-open results in browser
```

### Test All Portfolios
```bash
python -m src.cli.unified_cli portfolio test-all [options]
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

## Examples

### Test Crypto Portfolio
```bash
python -m src.cli.unified_cli portfolio test crypto \
  --metric sharpe_ratio \
  --period 1y \
  --test-timeframes \
  --open-browser
```

### Download Forex Data
```bash
python -m src.cli.unified_cli data download \
  --symbols EURUSD=X,GBPUSD=X \
  --start-date 2023-01-01 \
  --source twelve_data
```

### Daily Workflow
```bash
# Check cache status
python -m src.cli.unified_cli cache stats

# Test all portfolios
python -m src.cli.unified_cli portfolio test-all --period 1d --open-browser

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
