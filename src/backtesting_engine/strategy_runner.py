from backtesting_engine.engine import BacktestEngine
from backtesting_engine.data_loader import DataLoader
from backtesting_engine.result_analyzer import ResultAnalyzer
from backtester.strategies.strategy_factory import StrategyFactory

class StrategyRunner:
    @staticmethod
    def execute(strategy_name, ticker, start, end):
        data = DataLoader.load_data(ticker, start, end)
        strategy = StrategyFactory.get_strategy(strategy_name)
        engine = BacktestEngine(strategy, data)
        results = engine.run()
        return ResultAnalyzer.analyze(results)