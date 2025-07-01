# Quant System Optimization Guide

## Overview

The optimized quant system provides a comprehensive framework for backtesting thousands of assets with multiple strategies, advanced parameter optimization, and intelligent caching. This guide covers the new features and how to use them effectively.

## Key Features

### 🚀 **Multi-Source Data Management**
- **Yahoo Finance** (primary, free, high-quality data)
- **Alpha Vantage** (optional, requires API key, 500 requests/day free)
- **Twelve Data** (optional, requires API key, 800 requests/day free)
- Intelligent fallback between sources
- Automatic data validation and cleaning

### ⚡ **High-Performance Backtesting**
- Parallel processing with configurable workers
- Memory-efficient batch processing
- Optimized indicators using Numba JIT compilation
- Incremental backtesting for new data only
- Smart batching to handle thousands of assets

### 🧠 **Advanced Optimization**
- **Genetic Algorithm**: Population-based search with evolution
- **Grid Search**: Exhaustive parameter space exploration
- **Bayesian Optimization**: Gaussian Process-based efficient search
- Ensemble strategy creation from top performers
- Multi-objective optimization support

### 💾 **Intelligent Caching**
- SQLite-based metadata management
- Compressed data storage (gzip)
- Hierarchical caching (data, backtests, optimizations)
- Automatic cache cleanup and size management
- TTL-based expiration

### 📊 **Advanced Reporting**
- Interactive HTML reports with Plotly charts
- JSON exports for programmatic access
- Portfolio performance analysis
- Strategy comparison reports
- Optimization convergence analysis
- Cached report generation

## Quick Start

### 1. Setup Environment Variables (Optional)

```bash
# For Alpha Vantage (free tier: 500 requests/day)
export ALPHA_VANTAGE_API_KEY="your_api_key_here"

# For Twelve Data (free tier: 800 requests/day)
export TWELVE_DATA_API_KEY="your_api_key_here"
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Basic Usage Examples

#### Advanced Backtesting
```bash
# Backtest multiple symbols and strategies with caching
python -m src.cli.main advanced-backtest \
    --symbols AAPL MSFT GOOGL AMZN TSLA \
    --strategies rsi macd bollinger_bands \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --max-workers 4 \
    --output-format html
```

#### Portfolio Optimization
```bash
# Optimize strategy parameters using genetic algorithm
python -m src.cli.main optimize \
    --symbols AAPL MSFT GOOGL \
    --strategies rsi macd \
    --param-config config/optimization_config.json \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --method genetic_algorithm \
    --max-iterations 100 \
    --population-size 50
```

#### Data Management
```bash
# Download and cache data
python -m src.cli.main data download \
    --symbols AAPL MSFT GOOGL AMZN TSLA \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --sources yahoo alpha_vantage \
    --interval 1d

# Check cache statistics
python -m src.cli.main data cache stats

# Clear old cache entries
python -m src.cli.main data cache clear --older-than 30
```

## Architecture Overview

### Data Flow
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Multi-Source    │───▶│  Advanced Cache │
│  - Yahoo Finance│    │  Data Manager    │    │  - SQLite DB    │
│  - Alpha Vantage│    │  - Fallback      │    │  - Compressed   │
│  - Twelve Data  │    │  - Validation    │    │  - TTL-based    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Optimized     │◀───│   Backtest       │───▶│   Advanced      │
│   Reporting     │    │   Engine         │    │   Optimizer     │
│  - Interactive  │    │  - Parallel      │    │  - Genetic Algo │
│  - Cached       │    │  - Memory Opt    │    │  - Bayesian     │
│  - Multi-format │    │  - Incremental   │    │  - Grid Search  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Performance Optimization

#### Memory Management
- **Batch Processing**: Processes assets in configurable batches to manage memory
- **Garbage Collection**: Automatic cleanup between batches
- **Memory Monitoring**: Tracks usage and adjusts batch sizes
- **Data Streaming**: Loads only necessary data into memory

#### Parallel Processing
- **Process Pool**: Uses multiprocessing for CPU-intensive tasks
- **Thread Pool**: Uses threading for I/O-bound operations
- **Async Operations**: Async/await for concurrent data fetching
- **Smart Batching**: Optimizes batch sizes based on available resources

#### Caching Strategy
- **L1 Cache**: In-memory pandas DataFrames
- **L2 Cache**: Compressed files on disk
- **L3 Cache**: SQLite database for metadata
- **Smart Invalidation**: TTL-based with dependency tracking

## Configuration

### Optimization Parameters

The system supports extensive parameter optimization for various strategies:

```json
{
  "strategy_parameters": {
    "rsi": {
      "period": [10, 14, 20, 30],
      "overbought": [70, 75, 80],
      "oversold": [20, 25, 30]
    },
    "macd": {
      "fast": [8, 12, 16],
      "slow": [21, 26, 30],
      "signal": [6, 9, 12]
    }
  }
}
```

### Asset Universes

Pre-defined asset universes for easy backtesting:

- **S&P 500 Large Cap**: Top 100 S&P 500 stocks
- **NASDAQ Tech**: Technology stocks from NASDAQ
- **Forex Majors**: Major currency pairs
- **Crypto Major**: Major cryptocurrencies  
- **Commodities**: Futures contracts
- **Sector ETFs**: Sector-specific ETFs

### Risk Management

Built-in risk management constraints:

```json
{
  "risk_management": {
    "max_drawdown_threshold": -20.0,
    "min_sharpe_ratio": 0.5,
    "max_leverage": 2.0,
    "position_size_limits": {
      "min_percentage": 0.01,
      "max_percentage": 0.1
    }
  }
}
```

## Optimization Methods

### 1. Genetic Algorithm (Recommended)

Best for: Large parameter spaces, non-linear optimization

**Advantages:**
- Explores diverse solution space
- Handles discrete and continuous parameters
- Good balance of exploration vs exploitation
- Parallel evaluation of population

**Configuration:**
```python
config = OptimizationConfig(
    method="genetic_algorithm",
    population_size=50,
    max_iterations=100,
    mutation_rate=0.1,
    crossover_rate=0.7,
    early_stopping_patience=20
)
```

### 2. Grid Search

Best for: Small parameter spaces, exhaustive search

**Advantages:**
- Guaranteed to find global optimum in search space
- Fully parallel execution
- Simple and interpretable

**Limitations:**
- Exponential growth with parameter dimensions
- May be computationally expensive

### 3. Bayesian Optimization

Best for: Expensive objective functions, smart sampling

**Advantages:**
- Sample-efficient
- Handles noise well
- Uses Gaussian Processes for uncertainty quantification

**Limitations:**
- Only supports continuous parameters
- Requires more setup

## Advanced Features

### Ensemble Strategies

Create ensemble strategies from optimization results:

```python
from src.portfolio.advanced_optimizer import AdvancedPortfolioOptimizer

optimizer = AdvancedPortfolioOptimizer()
results = optimizer.optimize_portfolio(config)

# Create ensemble from top 5 strategies
ensemble = optimizer.create_ensemble_strategy(results, top_n=5)
```

### Custom Constraints

Add custom constraint functions:

```python
def max_trades_constraint(metrics, parameters):
    return metrics.get('num_trades', 0) <= 100

def sharpe_constraint(metrics, parameters):
    return metrics.get('sharpe_ratio', 0) >= 1.0

config.constraint_functions = [max_trades_constraint, sharpe_constraint]
```

### Incremental Backtesting

Update existing backtests with new data:

```python
from src.backtesting_engine.optimized_engine import OptimizedBacktestEngine

engine = OptimizedBacktestEngine()

# Only backtest new data since last run
result = engine.run_incremental_backtest(
    symbol="AAPL",
    strategy="rsi",
    config=config,
    last_update=datetime(2023, 12, 1)
)
```

## Performance Tips

### 1. Optimize Data Loading
- Use batch data downloads for multiple symbols
- Enable caching for repeated backtests
- Use appropriate data intervals (daily vs intraday)

### 2. Smart Batching
- Adjust batch sizes based on available memory
- Use fewer workers for memory-intensive operations
- Monitor memory usage during large runs

### 3. Caching Strategy
- Enable caching for development and testing
- Clear cache periodically to manage disk space
- Use incremental backtesting for daily updates

### 4. Parameter Optimization
- Start with genetic algorithm for exploration
- Use grid search for final fine-tuning
- Limit parameter ranges to reasonable values

## Troubleshooting

### Common Issues

#### 1. Memory Errors
```bash
# Reduce batch size and workers
python -m src.cli.main advanced-backtest \
    --memory-limit 4.0 \
    --max-workers 2
```

#### 2. API Rate Limits
```bash
# Check data source status
python -m src.cli.main data cache stats

# Use cached data only
python -m src.cli.main advanced-backtest --no-cache
```

#### 3. Cache Issues
```bash
# Clear corrupted cache
python -m src.cli.main data cache clear --type data

# Reset entire cache
rm -rf cache/
```

### Performance Monitoring

Enable detailed logging for performance analysis:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Monitor cache usage:
```bash
python -m src.cli.main data cache stats
```

Check optimization convergence:
```python
# Review optimization history in results
for generation in result.optimization_history:
    print(f"Generation {generation['generation']}: {generation['best_score']}")
```

## Examples

### 1. Large-Scale Portfolio Optimization

```python
from src.portfolio.advanced_optimizer import AdvancedPortfolioOptimizer, OptimizationConfig

# S&P 500 stocks
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", ...]  # 100+ symbols

config = OptimizationConfig(
    symbols=symbols,
    strategies=["rsi", "macd", "bollinger_bands", "turtle_trading"],
    parameter_ranges=optimization_params,
    optimization_metric="sharpe_ratio",
    start_date="2020-01-01",
    end_date="2023-12-31",
    max_iterations=50,
    population_size=30,
    n_jobs=-1
)

optimizer = AdvancedPortfolioOptimizer()
results = optimizer.optimize_portfolio(config, method="genetic_algorithm")

# Generate comprehensive report
from src.reporting.advanced_reporting import AdvancedReportGenerator
report_gen = AdvancedReportGenerator()
report_path = report_gen.generate_optimization_report(results)
```

### 2. Multi-Asset Class Analysis

```python
# Define asset universes
asset_classes = {
    "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
    "crypto": ["BTC-USD", "ETH-USD", "ADA-USD"],
    "commodities": ["GC=F", "CL=F", "SI=F"]
}

# Run backtests for each asset class
results_by_class = {}
for asset_class, symbols in asset_classes.items():
    config = BacktestConfig(
        symbols=symbols,
        strategies=["rsi", "macd"],
        start_date="2022-01-01",
        end_date="2023-12-31"
    )
    
    results_by_class[asset_class] = engine.run_batch_backtests(config)

# Compare performance across asset classes
report_path = report_gen.generate_strategy_comparison_report(results_by_class)
```

### 3. Real-Time Strategy Monitoring

```python
from datetime import datetime, timedelta

def monitor_strategies():
    symbols = ["AAPL", "MSFT", "GOOGL"]
    strategies = ["rsi", "macd"]
    
    for symbol in symbols:
        for strategy in strategies:
            # Check if we have recent data
            last_week = datetime.now() - timedelta(days=7)
            
            result = engine.run_incremental_backtest(
                symbol=symbol,
                strategy=strategy,
                config=config,
                last_update=last_week
            )
            
            if result and not result.error:
                print(f"{symbol}/{strategy}: {result.metrics.get('total_return', 0):.2f}%")

# Run daily monitoring
monitor_strategies()
```

## Best Practices

### 1. Development Workflow
1. Start with small symbol sets for testing
2. Use cached data during development
3. Optimize parameters on training period
4. Validate on out-of-sample data
5. Monitor performance in production

### 2. Production Deployment
1. Set up automated data downloads
2. Schedule regular optimization runs
3. Monitor cache usage and cleanup
4. Set up alerting for failed backtests
5. Regular performance reviews

### 3. Risk Management
1. Always use position sizing limits
2. Monitor maximum drawdown
3. Diversify across strategies and assets
4. Regular strategy performance reviews
5. Implement circuit breakers for extreme losses

### 4. Optimization Guidelines
1. Use walk-forward analysis for robustness
2. Avoid over-optimization (curve fitting)
3. Use out-of-sample testing
4. Consider transaction costs
5. Test on different market regimes

## Support and Contributing

For issues, feature requests, or contributions, please refer to the main repository documentation.
