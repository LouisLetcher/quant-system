from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import numpy as np

from src.backtesting_engine.engine import BacktestEngine
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class ParameterTuner:
    """Class for tuning strategy parameters."""

    def __init__(
        self,
        strategy_class,
        data,
        initial_capital=10000,
        commission=0.001,
        ticker="UNKNOWN",
        metric="sharpe",
    ):
        """
        Initialize the parameter tuner.

        Args:
            strategy_class: The strategy class to optimize
            data: Price data for backtesting
            initial_capital: Initial capital for backtesting
            commission: Commission rate for backtesting
            ticker: Ticker symbol for the asset
            metric: Metric to optimize ('sharpe', 'return', 'profit_factor')
        """
        self.strategy_class = strategy_class
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.ticker = ticker
        self.metric = metric
        self.optimization_results = []

    def optimize(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        max_tries: int = 100,
        method: str = "random",
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Optimize strategy parameters.

        Args:
            param_ranges: Dictionary of parameter names and their ranges (min, max)
            max_tries: Maximum number of optimization attempts
            method: Optimization method ('random', 'grid', 'bayesian')

        Returns:
            Tuple of (best_parameters, best_score, optimization_history)
        """
        logger.info(
            f"Starting parameter optimization for {self.ticker} with method={method}, max_tries={max_tries}"
        )

        # Reset optimization results
        self.optimization_results = []

        # Choose optimization method
        if method == "random":
            return self._random_search(param_ranges, max_tries)
        if method == "grid":
            return self._grid_search(param_ranges, max_tries)
        logger.warning(
            f"Unsupported optimization method: {method}. Using random search instead."
        )
        return self._random_search(param_ranges, max_tries)

    def _random_search(
        self, param_ranges: Dict[str, Tuple[float, float]], max_tries: int
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Perform random search optimization.

        Args:
            param_ranges: Dictionary of parameter names and their ranges (min, max)
            max_tries: Maximum number of optimization attempts

        Returns:
            Tuple of (best_parameters, best_score, optimization_history)
        """
        logger.info(
            f"Performing random search optimization with {max_tries} iterations"
        )

        best_score = float("-inf")
        best_params = {}

        # Try random parameter combinations
        for i in range(max_tries):
            # Generate random parameters within the specified ranges
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                # Handle integer parameters
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = random.randint(min_val, max_val)
                else:
                    params[param_name] = min_val + random.random() * (max_val - min_val)

            # Run backtest with these parameters
            score = self._evaluate_parameters(params)

            # Store result
            result = {"params": params.copy(), "score": score}
            self.optimization_results.append(result)

            # Update best if better
            if score > best_score:
                best_score = score
                best_params = params.copy()
                logger.info(
                    f"New best parameters found (iteration {i+1}): score={best_score}"
                )
                logger.debug(f"Parameters: {best_params}")

        logger.info(f"Random search completed. Best score: {best_score}")
        logger.info(f"Best parameters: {best_params}")

        return best_params, best_score, self.optimization_results

    def _grid_search(
        self, param_ranges: Dict[str, Tuple[float, float]], max_points: int
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Perform grid search optimization.

        Args:
            param_ranges: Dictionary of parameter names and their ranges (min, max)
            max_points: Maximum number of grid points to evaluate

        Returns:
            Tuple of (best_parameters, best_score, optimization_history)
        """
        logger.info("Performing grid search optimization")

        # Calculate number of points per dimension
        n_params = len(param_ranges)
        points_per_dim = max(2, int(max_points ** (1 / n_params)))

        logger.info(
            f"Using {points_per_dim} points per dimension for {n_params} parameters"
        )

        # Create grid
        param_values = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                # For integer parameters, ensure we include the endpoints
                step = max(1, (max_val - min_val) // (points_per_dim - 1))
                param_values[param_name] = list(range(min_val, max_val + 1, step))
            else:
                # For float parameters
                param_values[param_name] = np.linspace(
                    min_val, max_val, points_per_dim
                ).tolist()

        # Generate all combinations
        param_names = list(param_ranges.keys())
        best_score = float("-inf")
        best_params = {}

        # Helper function to generate all combinations recursively
        def evaluate_combinations(current_params, param_idx):
            nonlocal best_score, best_params

            if param_idx == len(param_names):
                # We have a complete parameter set, evaluate it
                score = self._evaluate_parameters(current_params)

                # Store result
                result = {"params": current_params.copy(), "score": score}
                self.optimization_results.append(result)

                # Update best if better
                if score > best_score:
                    best_score = score
                    best_params = current_params.copy()
                    logger.info(f"New best parameters found: score={best_score}")
                    logger.debug(f"Parameters: {best_params}")

                return

            # Try each value for the current parameter
            param_name = param_names[param_idx]
            for value in param_values[param_name]:
                current_params[param_name] = value
                evaluate_combinations(current_params, param_idx + 1)

        # Start the recursive evaluation
        evaluate_combinations({}, 0)

        logger.info(f"Grid search completed. Best score: {best_score}")
        logger.info(f"Best parameters: {best_params}")

        return best_params, best_score, self.optimization_results

    def _evaluate_parameters(self, params: Dict[str, Any]) -> float:
        """
        Evaluate a set of parameters by running a backtest.

        Args:
            params: Dictionary of parameter values

        Returns:
            Score based on the selected metric
        """
        try:
            # Create a new strategy class with the specified parameters
            strategy_instance = type(
                "OptimizedStrategy", (self.strategy_class,), params
            )

            # Create backtest instance
            engine = BacktestEngine(
                strategy_instance,
                self.data,
                cash=self.initial_capital,
                commission=self.commission,
                ticker=self.ticker,
            )

            # Run backtest
            result = engine.run()

            # Extract performance metric
            if self.metric == "profit_factor":
                score = result.get("Profit Factor", 0)
            elif self.metric == "sharpe":
                score = result.get("Sharpe Ratio", 0)
            elif self.metric == "return":
                score = result.get("Return [%]", 0)
            else:
                score = result.get(self.metric, 0)

            # Handle invalid scores
            if score is None or (
                isinstance(score, float) and (np.isnan(score) or np.isinf(score))
            ):
                logger.warning(f"Invalid score ({score}) for parameters: {params}")
                return float("-inf")

            return score

        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return float("-inf")

    def objective(self, params):
        """Objective function to minimize (negative of the metric)."""
        return -self.backtest_function(params)
