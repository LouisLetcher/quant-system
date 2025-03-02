import numpy as np
from scipy.optimize import minimize

class ParameterTuner:
    """Tunes strategy parameters to maximize performance metrics"""

    def __init__(self, strategy, backtest_function):
        self.strategy = strategy
        self.backtest_function = backtest_function

    def objective(self, params):
        """Objective function to minimize/maximize"""
        return -self.backtest_function(params)  # Maximizing returns by minimizing negative

    def tune(self, param_bounds, method="L-BFGS-B"):
        """Runs optimization to find best parameters"""
        initial_guess = np.mean(np.array(list(param_bounds.values())), axis=1)
        bounds = list(param_bounds.values())

        result = minimize(self.objective, initial_guess, bounds=bounds, method=method)
        return result.x, -result.fun  # Best parameters and highest return