import logging
from bayes_opt import BayesianOptimization
from typing import Dict, Any, Callable
from src.backtesting_engine.strategy_runner import StrategyRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Handles the optimization of trading strategies using Bayesian Optimization."""

    def __init__(self, strategy_name: str, ticker: str, start_date: str, end_date: str):
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def optimize(self, param_bounds: Dict[str, tuple], evaluation_function: Callable[[Dict[str, Any]], float]):
        """
        Optimizes a given strategy using Bayesian Optimization.

        :param param_bounds: Dictionary with parameter names as keys and value ranges as tuples.
        :param evaluation_function: Function to evaluate the strategy performance given parameter inputs.
        """
        optimizer = BayesianOptimization(
            f=evaluation_function,
            pbounds=param_bounds,
            random_state=42,
            verbose=2
        )

        optimizer.maximize(init_points=5, n_iter=20)

        best_params = optimizer.max["params"]
        best_score = optimizer.max["target"]

        logger.info(f"✅ Optimization complete. Best Parameters: {best_params}, Best Score: {best_score}")

        return best_params, best_score

    def evaluate_strategy(self, **params) -> float:
        """
        Runs a backtest with given parameters and returns a performance score.

        :param params: Optimized strategy parameters.
        :return: Sharpe Ratio or another performance metric.
        """
        logger.info(f"🔍 Evaluating strategy {self.strategy_name} with params: {params}")

        results = StrategyRunner.execute(self.strategy_name, self.ticker, self.start_date, self.end_date, **params)

        if results and hasattr(results, 'analyzers') and 'sharpe' in results.analyzers:
            sharpe_ratio = results.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            logger.info(f"📊 Sharpe Ratio: {sharpe_ratio}")
            return sharpe_ratio
        else:
            logger.warning(f"⚠️ No Sharpe Ratio found, returning -1 as penalty")
            return -1  # Penalize bad configurations