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
        
        # Print what we're loading
        if period:
            print(f"📥 Loading data for {ticker} with period={period}...")
        else:
            print(f"📥 Loading data for {ticker} from {start} to {end}...")
            
        # Load data using period parameter
        data = DataLoader.load_data(ticker, period=period, start=start, end=end)
        print(f"✅ Successfully loaded {len(data)} rows for {ticker}.")

        # 🔍 Ensure strategy exists
        strategy_class = StrategyFactory.get_strategy(strategy_name)
        if strategy_class is None:
            raise ValueError(f"❌ Strategy '{strategy_name}' not found.")

        # 🚀 Initialize Backtrader engine with supported parameters
        # Check which parameters are supported by BacktestEngine
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