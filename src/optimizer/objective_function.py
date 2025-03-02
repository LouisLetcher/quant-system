from optimizer.parameter_tuner import ParameterTuner
from optimizer.objective_function import ObjectiveFunction
from backtesting_engine.engine import BacktestEngine
from backtester.strategies.strategy_factory import StrategyFactory

class OptimizationRunner:
    """Runs optimization for trading strategies"""

    def __init__(self, strategy_name, ticker, start, end):
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.start = start
        self.end = end

    def run(self):
        """Executes parameter optimization"""
        strategy = StrategyFactory.get_strategy(self.strategy_name)
        backtest_runner = lambda params: BacktestEngine(strategy, self.ticker, self.start, self.end, params).run()

        param_bounds = {
            "sma_period": (10, 50),
            "ema_period": (5, 30),
        }

        tuner = ParameterTuner(self.strategy_name, lambda params: ObjectiveFunction.evaluate(strategy, params, backtest_runner))
        best_params, best_score = tuner.tune(param_bounds)

        return {"best_params": best_params, "best_score": best_score}