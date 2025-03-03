from src.backtesting_engine.engine import BacktestEngine
from src.backtesting_engine.data_loader import DataLoader
from src.backtesting_engine.result_analyzer import ResultAnalyzer
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory

class StrategyRunner:
    """Executes a selected trading strategy for backtesting."""

    @staticmethod
    def execute(strategy_name, ticker, start, end):
        """Loads data, runs a strategy, and analyzes the result."""
        print(f"📥 Loading data for {ticker} from {start} to {end}...")
        data = DataLoader.load_data(ticker, start, end)
        print(f"✅ Successfully loaded {len(data)} rows for {ticker}.")

        # 🔍 Ensure strategy exists
        strategy_class = StrategyFactory.get_strategy(strategy_name)
        if strategy_class is None:
            raise ValueError(f"❌ Strategy '{strategy_name}' not found.")

        # 🚀 Initialize Backtrader engine
        print(f"🚀 Running Backtrader Engine for {ticker} from {start} to {end}...")
        engine = BacktestEngine(strategy_class, data, ticker=ticker)

        if not engine or engine is None:
            raise RuntimeError("❌ Engine failed to initialize.")

        # ✅ Run backtest
        results = engine.run()

        # 🔍 Ensure results exist before proceeding
        if results is None:  # Changed from "if not results or isinstance(results, list)"
            raise RuntimeError("❌ No results returned from Backtest Engine.")

        print(f"📊 Strategy finished. Analyzing results...")
        analyzed_results = ResultAnalyzer.analyze(results, ticker=ticker)  # Pass ticker directly

        if not isinstance(analyzed_results, dict):
            raise TypeError(f"❌ Expected results in dict format, got {type(analyzed_results)}.")

        print(f"✅ Backtest Complete! Results: {analyzed_results}")
        
        return analyzed_results
