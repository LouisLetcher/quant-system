from __future__ import annotations

from src.backtesting_engine.engine import BacktestEngine
from src.backtesting_engine.result_analyzer import BacktestResultAnalyzer

from bayes_opt import BayesianOptimization


class OptimizationRunner:
    """Runs Bayesian optimization to find optimal strategy parameters."""

    def __init__(self, strategy_class, data, param_space):
        """
        Initialize the optimization runner.

        Args:
            strategy_class: Strategy class to optimize
            data: DataFrame with OHLCV data
            param_space: Dictionary of parameter ranges {param_name: (min, max)}
        """
        self.strategy_class = strategy_class
        self.data = data
        self.param_space = param_space

    def run(
        self, metric="sharpe", iterations=50, initial_capital=10000, commission=0.001
    ):
        """
        Run the optimization process.

        Args:
            metric: Performance metric to optimize ("sharpe", "return", "profit_factor")
            iterations: Number of optimization iterations
            initial_capital: Initial capital amount
            commission: Commission rate

        Returns:
            Dictionary with optimization results
        """
        print(f"🔍 Running optimization with {iterations} iterations...")

        def evaluate_params(**params):
            """Evaluate a set of parameters by running a backtest."""

            # Create a subclass with the specified parameters
            class OptimizedStrategy(self.strategy_class):
                def init(self):
                    # Set parameters from optimization
                    for param_name, param_value in params.items():
                        setattr(self, param_name, param_value)

                    # Set initial capital
                    self._initial_capital = initial_capital

                    # Call the original init method
                    super().init()

            # Run backtest with these parameters
            engine = BacktestEngine(
                OptimizedStrategy,
                self.data,
                cash=initial_capital,
                commission=commission,
            )

            result = engine.run()

            # Extract the metric value
            analyzed_result = BacktestResultAnalyzer.analyze(
                result, initial_capital=initial_capital
            )

            if metric == "sharpe":
                score = analyzed_result.get("sharpe_ratio", 0)
                if isinstance(score, str):
                    score = float(score)
            elif metric == "return":
                return_pct = analyzed_result.get("return_pct", "0%")
                if isinstance(return_pct, str) and return_pct.endswith("%"):
                    score = float(return_pct.strip("%"))
                else:
                    score = float(return_pct)
            elif metric == "profit_factor":
                score = analyzed_result.get("profit_factor", 0)
                if isinstance(score, str):
                    score = float(score)
            else:
                score = analyzed_result.get(metric, 0)

            # Check if the result has trades
            trade_count = analyzed_result.get("trades", 0)
            if isinstance(trade_count, str) and trade_count.isdigit():
                trade_count = int(trade_count)

            # Penalize strategies with no trades
            if trade_count == 0:
                score = -100  # Strong penalty for no trades

            print(
                f"  Parameters: {params}, {metric.capitalize()}: {score}, Trades: {trade_count}"
            )
            return score

        # Run Bayesian optimization
        optimizer = BayesianOptimization(
            f=evaluate_params, pbounds=self.param_space, random_state=42, verbose=1
        )

        optimizer.maximize(init_points=5, n_iter=iterations)

        # Get best parameters and score
        best_params = optimizer.max["params"]
        best_score = optimizer.max["target"]

        print(f"✅ Optimization complete. Best parameters: {best_params}")
        print(f"   Best {metric} score: {best_score:.4f}")

        # Run a final backtest with the best parameters to get detailed results
        final_params = {}
        for param_name, param_value in best_params.items():
            # Round integer parameters
            if param_name.endswith("_period") or param_name.endswith("_length"):
                final_params[param_name] = int(round(param_value))
            else:
                final_params[param_name] = param_value

        class FinalStrategy(self.strategy_class):
            def init(self):
                # Set parameters from optimization
                for param_name, param_value in final_params.items():
                    setattr(self, param_name, param_value)

                # Set initial capital
                self._initial_capital = initial_capital

                # Call the original init method
                super().init()

        # Run final backtest
        engine = BacktestEngine(
            FinalStrategy, self.data, cash=initial_capital, commission=commission
        )

        final_result = engine.run()
        analyzed_final = BacktestResultAnalyzer.analyze(
            final_result, initial_capital=initial_capital
        )

        # Prepare optimization results
        optimization_results = {
            "strategy": self.strategy_class.__name__,
            "best_params": final_params,
            "best_score": best_score,
            "metric": metric,
            "iterations": iterations,
            "final_result": analyzed_final,
            "all_trials": [
                {"params": trial["params"], "score": trial["target"]}
                for trial in optimizer.res
            ],
        }

        return optimization_results
