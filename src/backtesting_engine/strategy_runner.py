import json
import os
from src.backtesting_engine.engine import BacktestEngine
from src.backtesting_engine.data_loader import DataLoader
from src.backtesting_engine.result_analyzer import BacktestResultAnalyzer
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory

class StrategyRunner:
    """Executes a selected trading strategy for backtesting."""

    @staticmethod
    def execute(strategy_name, ticker, period="max", start=None, end=None, commission=0.001, initial_capital=10000, take_profit=None, stop_loss=None):
        """
        Loads data, runs a strategy, and analyzes the result.
        
        Args:
            strategy_name: Name of the strategy to run
            ticker: Stock ticker symbol
            period: Data period (e.g., "max", "1y", "6mo", etc.)
            start: Start date (used only if period is None)
            end: End date (used only if period is None)
            commission: Commission rate
            initial_capital: Initial capital amount
            take_profit: Take profit percentage
            stop_loss: Stop loss percentage
        """
        if period and (start or end):
            print("⚠️ Both period and start/end dates provided. Using period for data fetching.")
        
        # 🔍 Ensure strategy exists
        strategy_class = StrategyFactory.get_strategy(strategy_name)
        if strategy_class is None:
            raise ValueError(f"❌ Strategy '{strategy_name}' not found.")
        
        # Check if ticker is a portfolio from assets_config.json
        config_path = os.path.join('config', 'assets_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                assets_config = json.load(f)
                
            if ticker in assets_config.get('portfolios', {}):
                return StrategyRunner.execute_portfolio(
                    strategy_name=strategy_name,
                    portfolio_name=ticker,
                    portfolio_config=assets_config['portfolios'][ticker],
                    start=start,
                    end=end
                )
        
        # Single asset execution
        # Print what we're loading
        if period:
            print(f"📥 Loading data for {ticker} with period={period}...")
        else:
            print(f"📥 Loading data for {ticker} from {start} to {end}...")
            
        # Load data using period parameter
        data = DataLoader.load_data(ticker, period=period, start=start, end=end)
        print(f"✅ Successfully loaded {len(data)} rows for {ticker}.")

        # 🚀 Initialize Backtrader engine with supported parameters
        print(f"🚀 Running Backtrader Engine for {ticker}...")
        
        # Only pass parameters that BacktestEngine accepts
        engine_params = {
            'ticker': ticker,
        }
        
        # Add commission if it's a supported parameter
        try:
            engine = BacktestEngine(strategy_class, data, ticker=ticker, commission=commission, cash=initial_capital)
            engine_params['commission'] = commission
        except TypeError:
            # If commission is not supported, create without it
            engine = BacktestEngine(strategy_class, data, ticker=ticker, cash=initial_capital)
        
        # Store additional parameters for later use if needed
        engine.params = {
            'initial_capital': initial_capital,
            'take_profit': take_profit,
            'stop_loss': stop_loss
        }

        if not engine or engine is None:
            raise RuntimeError("❌ Engine failed to initialize.")

        # ✅ Run backtest
        results = engine.run()

        # 🔍 Ensure results exist before proceeding
        if results is None:
            raise RuntimeError("❌ No results returned from Backtest Engine.")

        # Add debugging to help troubleshoot
        print(f"Debug - Raw backtest results type: {type(results)}")
        print(f"Debug - Available metrics: {[k for k in results.keys() if not k.startswith('_')]}")
        print(f"Debug - Trade count from raw results: {results.get('# Trades', 'Not found')}")

        print(f"📊 Strategy finished. Analyzing results...")
        analyzed_results = BacktestResultAnalyzer.analyze(results, ticker=ticker, initial_capital=initial_capital)
        
        # Add the additional parameters to the results
        analyzed_results.update({
            'initial_capital': initial_capital,
            'commission': commission
        })
        
        if take_profit:
            analyzed_results['take_profit'] = take_profit
        if stop_loss:
            analyzed_results['stop_loss'] = stop_loss

        if not isinstance(analyzed_results, dict):
            raise TypeError(f"❌ Expected results in dict format, got {type(analyzed_results)}.")

        print(f"✅ Backtest Complete! Results: {analyzed_results}")
        
        return analyzed_results
    
    @staticmethod
    def execute_portfolio(strategy_name, portfolio_name, portfolio_config, start=None, end=None):
        """
        Execute a strategy on a portfolio of assets defined in assets_config.json
        
        Args:
            strategy_name: Name of the strategy to run
            portfolio_name: Name of the portfolio from assets_config.json
            portfolio_config: Portfolio configuration from assets_config.json
            start: Start date for backtest
            end: End date for backtest
        """
        # Get strategy class
        strategy_class = StrategyFactory.get_strategy(strategy_name)
        if strategy_class is None:
            raise ValueError(f"❌ Strategy '{strategy_name}' not found.")
        
        print(f"📂 Running portfolio backtest for '{portfolio_name}' with {len(portfolio_config['assets'])} assets...")
        
        # Extract initial capital from portfolio config
        initial_capital = portfolio_config.get('initial_capital', 10000)
        
        # Load data for each asset in the portfolio
        portfolio_data = {}
        commission_rates = {}
        
        for asset in portfolio_config['assets']:
            ticker = asset['ticker']
            period = asset.get('period', 'max')
            commission = asset.get('commission', 0.001)
            
            print(f"📥 Loading data for {ticker} with period={period}...")
            
            # Use period if provided, otherwise use start/end dates
            if period and not (start or end):
                data = DataLoader.load_data(ticker, period=period)
            else:
                data = DataLoader.load_data(ticker, start=start, end=end)
                
            print(f"✅ Successfully loaded {len(data)} rows for {ticker}.")
            
            portfolio_data[ticker] = data
            commission_rates[ticker] = commission
        
        # Initialize portfolio backtest engine
        print(f"🚀 Initializing portfolio backtest for {portfolio_name}...")
        
        engine = BacktestEngine(
            strategy_class,
            portfolio_data,
            cash=initial_capital,
            commission=commission_rates,
            ticker=portfolio_name,
            is_portfolio=True
        )
        
        # Run the portfolio backtest
        results = engine.run()
        
        # 🔍 Ensure results exist before proceeding
        if results is None:
            raise RuntimeError("❌ No results returned from Portfolio Backtest Engine.")
        
        print(f"📊 Portfolio strategy finished. Analyzing results...")
        analyzed_results = BacktestResultAnalyzer.analyze(
            results, 
            ticker=portfolio_name, 
            initial_capital=initial_capital
        )
        
        # Add portfolio metadata
        analyzed_results.update({
            'portfolio_name': portfolio_name,
            'portfolio_description': portfolio_config.get('description', ''),
            'asset_count': len(portfolio_config['assets']),
            'initial_capital': initial_capital
        })
        
        print(f"✅ Portfolio Backtest Complete! Overall Return: {analyzed_results['return_pct']}")
        
        return analyzed_results

    @staticmethod
    def optimize(strategy_name, ticker, param_space, metric='sharpe', period="max", 
                iterations=50, initial_capital=10000, commission=0.001):
        """
        Optimizes strategy parameters using Bayesian optimization.
        
        Args:
            strategy_name: Name of the strategy to optimize
            ticker: Stock ticker symbol
            param_space: Dictionary of parameter ranges
            metric: Metric to optimize ('sharpe', 'return', etc.)
            period: Data period
            iterations: Number of optimization iterations
            initial_capital: Initial capital amount
            commission: Commission rate
        """
        from src.optimizer.optimization_runner import OptimizationRunner
        
        print(f"🔍 Optimizing {strategy_name} for {ticker} using {metric} metric...")
        
        # Load data
        data = DataLoader.load_data(ticker, period=period)
        
        # Get strategy class
        strategy_class = StrategyFactory.get_strategy(strategy_name)
        if strategy_class is None:
            raise ValueError(f"❌ Strategy '{strategy_name}' not found.")
        
        # Run optimization
        optimizer = OptimizationRunner(strategy_class, data, param_space)
        results = optimizer.run(metric=metric, iterations=iterations, 
                                initial_capital=initial_capital, commission=commission)
        
        print(f"✅ Optimization complete. Best parameters: {results['best_params']}")
        print(f"   Best {metric} score: {results['best_score']:.4f}")
        
        return results
