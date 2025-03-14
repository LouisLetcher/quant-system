# 📊 Quant Trading System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Poetry](https://img.shields.io/badge/Poetry-Package%20Manager-1E293B)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Python-based quantitative trading system for backtesting, optimizing, and analyzing algorithmic trading strategies with professional-grade reports.

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Configuration](#-configuration)
- [CLI Commands](#-cli-commands)
- [Example Workflows](#-example-workflows)
- [Code Quality](#-code-quality)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

## 🔍 Overview

This Quant Trading System enables traders, quants, and financial analysts to:

- Backtest trading strategies against historical market data
- Optimize strategy parameters for maximum performance
- Analyze performance with comprehensive metrics
- Generate professional HTML reports with interactive charts
- Run portfolio-level analysis across multiple assets
- Find optimal combinations of strategies and timeframes

Whether you're a professional trader or a financial enthusiast, this system provides the tools to validate and refine your trading strategies with rigorous quantitative analysis.

## 🔥 Key Features

✅ **Data Acquisition & Management**
- Fetch historical price data from Yahoo Finance (`yfinance`)
- Intelligent caching system for efficient data retrieval
- Data cleaning and preprocessing utilities

✅ **Backtesting Engine**
- Multiple built-in trading strategies
- Custom strategy development framework
- Commission modeling and slippage simulation
- Multi-timeframe analysis

✅ **Strategy Optimization**
- Bayesian optimization for parameter tuning
- Performance metric selection (Sharpe, profit factor, returns)
- Hyperparameter search with constraints

✅ **Portfolio Analysis**
- Multi-asset backtesting
- Portfolio optimization
- Risk assessment and drawdown analysis
- Asset correlation analysis

✅ **Reporting & Visualization**
- Interactive HTML reports with charts
- Detailed portfolio reports with equity curves and drawdown charts
- Trade analysis tables with win/loss highlighting
- Performance metrics dashboards
- Tabbed interface for easy navigation across assets

✅ **API & Integration**
- FastAPI backend for frontend integration
- Database integration for storing results
- Docker support for deployment

## 🏗 Architecture

```
quant-system/
├── src/
│   ├── api/                # FastAPI endpoints
│   ├── backtesting_engine/ # Backtesting functionality
│   ├── cli/                # Command-line interface
│   ├── data_scraper/       # Data acquisition modules
│   ├── optimizer/          # Optimization algorithms
│   ├── reports/            # Report generation & templates
│   └── utils/              # Utility functions
├── config/                 # Configuration files
├── reports_output/         # Generated report output
└── tests/                  # Test suites
```

## 🛠 Tech Stack

- **FastAPI**: Backend API framework for frontend integration
- **Backtesting.py**: Core backtesting engine
- **yfinance**: Market data acquisition
- **Bayesian Optimization**: Parameter tuning algorithms
- **PostgreSQL/MongoDB**: Data storage options
- **Jinja2 + Chart.js**: HTML report generation with interactive charts
- **Docker**: Containerization for deployment
- **Poetry**: Dependency management
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib**: Visualization for equity curves and drawdowns

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Poetry package manager
- Git

### 1️⃣ Install Poetry (if not already installed)
```bash
pip install poetry
```

### 2️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/quant-system.git
cd quant-system
```

### 3️⃣ Install Dependencies
```bash
poetry install
```

### 4️⃣ Activate Virtual Environment
```bash
poetry shell
```

## ⚙️ Configuration

### Portfolio Configuration
Create `config/assets_config.json` with your portfolio settings:

```json
{
    "portfolios": {
        "tech_stocks": {
            "description": "Technology sector stocks",
            "assets": [
                {
                    "ticker": "AAPL",
                    "commission": 0.001,
                    "initial_capital": 10000
                },
                {
                    "ticker": "MSFT",
                    "commission": 0.001,
                    "initial_capital": 10000
                }
            ]
        }
    }
}
```

## 🧪 CLI Commands

### Strategy Backtesting

#### Single Strategy Backtest
```bash
poetry run python -m src.cli.main backtest --strategy mean_reversion --ticker AAPL --period max
```

#### Test All Available Strategies on a Single Asset
```bash
poetry run python -m src.cli.main all-strategies --ticker TSLA --period max --metric profit_factor
```

#### Backtest a Portfolio with All Strategies
```bash
poetry run python -m src.cli.main portfolio --name tech_stocks --period max --metric sharpe --open-browser
```

### Timeframe Analysis

#### Test Different Timeframes for a Strategy
```bash
poetry run python -m src.cli.main intervals --strategy momentum --ticker AAPL
```

#### Find Optimal Strategy and Timeframe Combination
```bash
poetry run python -m src.cli.main portfolio-optimal --name tech_stocks --metric sharpe --intervals 1d 1h 4h --open-browser
```

### Strategy Optimization

#### Optimize Strategy Parameters
```bash
poetry run python -m src.cli.main optimize --strategy mean_reversion --ticker AAPL --metric sharpe --iterations 50
```

### Utility Commands

#### List Available Portfolios
```bash
poetry run python -m src.cli.main list-portfolios
```

#### List Available Strategies
```bash
poetry run python -m src.cli.main list-strategies
```

## 📋 Example Workflows

### Momentum Strategy Development Workflow

1. Create a portfolio configuration in `config/assets_config.json`
```bash
# List available strategies
poetry run python -m src.cli.main list-strategies

# Backtest the momentum strategy on Apple
poetry run python -m src.cli.main backtest --strategy momentum --ticker AAPL --period 5y

# Optimize the strategy parameters
poetry run python -m src.cli.main optimize --strategy momentum --ticker AAPL --metric sharpe --iterations 100

# Test the strategy across different timeframes
poetry run python -m src.cli.main intervals --strategy momentum --ticker AAPL

# Apply the strategy to a portfolio
poetry run python -m src.cli.main portfolio --name tech_stocks --period 5y --metric sharpe --open-browser
```

### Finding the Best Strategy for a Portfolio

```bash
# List available portfolios
poetry run python -m src.cli.main list-portfolios

# Find optimal strategy-timeframe combinations for each asset
poetry run python -m src.cli.main portfolio-optimal --name tech_stocks --metric profit_factor --intervals 1d 1h 4h --open-browser
```

### Detailed Portfolio Analysis

```bash
# Generate a detailed portfolio report with equity curves and trade tables
poetry run python -m src.cli.main portfolio --name tech_stocks --period 5y --metric sharpe --open-browser

# Compare different timeframes for optimal performance
poetry run python -m src.cli.main portfolio-optimal --name tech_stocks --intervals 1d 1h 4h --metric profit_factor --open-browser
```

The detailed reports include:
- Performance summary statistics for the entire portfolio
- Interactive tabs to view each asset's performance
- Equity curves with drawdown visualization
- Detailed trade tables with win/loss highlighting
- Key metrics including Sharpe ratio, profit factor, and maximum drawdown

## 🎯 Code Quality

Run these commands to maintain code quality:

```bash
# Format code
poetry run black src/

# Sort imports
poetry run isort src/

# Run linter
poetry run ruff check src/
```

## 🚀 Deployment

### Deploy with Docker

```bash
# Build Docker image
docker build -t quant-trading-app .

# Run container
docker run -p 8000:8000 quant-trading-app
```

### Access API Endpoints

Once deployed, access the API at:
```
http://localhost:8000/docs
```

## 🔧 Troubleshooting

### Common Issues

#### Module Import Errors
If you encounter "No module named 'src.utils'" or similar:
```bash
# Ensure you have __init__.py files in all directories
touch src/__init__.py
touch src/utils/__init__.py
```

#### Data Fetching Issues
If you encounter problems with data fetching:
```bash
# Check your internet connection
# Try with a different ticker or time period
poetry run python -m src.cli.main backtest --strategy mean_reversion --ticker SPY --period 1y
```

#### Report Generation Errors
Ensure the reports_output directory exists:
```bash
mkdir -p reports_output
```

## 📜 License

Proprietary License - All rights reserved.
