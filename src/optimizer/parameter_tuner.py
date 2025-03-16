from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class ParameterTuner:
    """Tunes strategy parameters to maximize performance metrics."""

    def __init__(self, backtest_function):
        self.backtest_function = backtest_function

    def objective(self, params):
        """Objective function to minimize (negative Sharpe Ratio)."""
        return -self.backtest_function(params)

    def tune(self, param_bounds, method="L-BFGS-B"):
        """Runs optimization to find best parameters."""
        initial_guess = np.mean(np.array(list(param_bounds.values())), axis=1)
        bounds = list(param_bounds.values())

        result = minimize(self.objective, initial_guess, bounds=bounds, method=method)
        return result.x, -result.fun  # Best parameters and highest return
