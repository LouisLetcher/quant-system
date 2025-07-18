"""
Unified Result Analyzer - Consolidates all result analysis functionality.
Calculates comprehensive metrics for backtests, portfolios, and optimizations.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


class UnifiedResultAnalyzer:
    """
    Unified result analyzer that consolidates all result analysis functionality.
    Provides comprehensive metrics calculation for different types of results.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(
        self, backtest_result: dict[str, Any], initial_capital: float
    ) -> dict[str, float]:
        """
        Calculate comprehensive metrics for a single backtest result.

        Args:
            backtest_result: Backtest result dictionary with equity_curve and trades
            initial_capital: Initial capital amount

        Returns:
            Dictionary of calculated metrics
        """
        try:
            equity_curve = backtest_result.get("equity_curve")
            trades = backtest_result.get("trades")
            final_capital = backtest_result.get("final_capital", initial_capital)

            if equity_curve is None or equity_curve.empty:
                return self._get_zero_metrics()

            # Convert equity curve to pandas Series if needed
            equity_values = (
                equity_curve["equity"]
                if isinstance(equity_curve, pd.DataFrame)
                else equity_curve
            )

            # Calculate returns
            returns = equity_values.pct_change().dropna()

            # Basic metrics
            metrics = {
                "total_return": ((final_capital - initial_capital) / initial_capital)
                * 100,
                "annualized_return": self._calculate_annualized_return(
                    equity_values, initial_capital
                ),
                "volatility": self._calculate_volatility(returns),
                "sharpe_ratio": self._calculate_sharpe_ratio(returns),
                "sortino_ratio": self._calculate_sortino_ratio(returns),
                "calmar_ratio": self._calculate_calmar_ratio(
                    equity_values, initial_capital
                ),
                "max_drawdown": self._calculate_max_drawdown(equity_values),
                "max_drawdown_duration": self._calculate_max_drawdown_duration(
                    equity_values
                ),
                "var_95": self._calculate_var(returns, 0.05),
                "cvar_95": self._calculate_cvar(returns, 0.05),
                "skewness": self._calculate_skewness(returns),
                "kurtosis": self._calculate_kurtosis(returns),
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "num_trades": 0,
                "avg_trade_duration": 0,
                "expectancy": 0,
            }

            # Trade-specific metrics
            if trades is not None and not trades.empty:
                trade_metrics = self._calculate_trade_metrics(trades)
                metrics.update(trade_metrics)

            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(returns, equity_values)
            metrics.update(risk_metrics)

            return metrics

        except Exception as e:
            self.logger.error("Error calculating metrics: %s", e)
            return self._get_zero_metrics()

    def calculate_portfolio_metrics(
        self, portfolio_data: dict[str, Any], initial_capital: float
    ) -> dict[str, float]:
        """
        Calculate metrics for portfolio backtests.

        Args:
            portfolio_data: Portfolio data with returns, equity_curve, weights
            initial_capital: Initial capital amount

        Returns:
            Dictionary of portfolio metrics
        """
        try:
            returns = portfolio_data.get("returns")
            equity_curve = portfolio_data.get("equity_curve")
            weights = portfolio_data.get("weights", {})

            if returns is None or equity_curve is None:
                return self._get_zero_metrics()

            # Basic portfolio metrics
            return {
                "total_return": (
                    (equity_curve.iloc[-1] - initial_capital) / initial_capital
                )
                * 100,
                "annualized_return": self._calculate_annualized_return(
                    equity_curve, initial_capital
                ),
                "volatility": self._calculate_volatility(returns),
                "sharpe_ratio": self._calculate_sharpe_ratio(returns),
                "sortino_ratio": self._calculate_sortino_ratio(returns),
                "max_drawdown": self._calculate_max_drawdown(equity_curve),
                "var_95": self._calculate_var(returns, 0.05),
                "cvar_95": self._calculate_cvar(returns, 0.05),
                "num_assets": len(weights),
                "effective_assets": self._calculate_effective_number_assets(weights),
                "concentration_ratio": max(weights.values()) if weights else 0,
                "diversification_ratio": self._calculate_diversification_ratio(weights),
            }

        except Exception as e:
            self.logger.error("Error calculating portfolio metrics: %s", e)
            return self._get_zero_metrics()

    def calculate_optimization_metrics(
        self, optimization_results: dict[str, Any]
    ) -> dict[str, float]:
        """
        Calculate metrics for optimization results.

        Args:
            optimization_results: Optimization results data

        Returns:
            Dictionary of optimization metrics
        """
        try:
            history = optimization_results.get("optimization_history", [])
            final_population = optimization_results.get("final_population", [])

            if not history:
                return {}

            # Extract scores from history
            scores = [entry.get("score", 0) for entry in history if "score" in entry]
            best_scores = [
                entry.get("best_score", 0) for entry in history if "best_score" in entry
            ]

            return {
                "convergence_speed": self._calculate_convergence_speed(best_scores),
                "final_diversity": self._calculate_population_diversity(
                    final_population
                ),
                "improvement_rate": self._calculate_improvement_rate(best_scores),
                "stability_ratio": self._calculate_stability_ratio(best_scores),
                "exploration_ratio": self._calculate_exploration_ratio(scores),
                "total_evaluations": len(scores),
                "successful_evaluations": len([s for s in scores if s > 0]),
                "best_score": max(scores) if scores else 0,
                "avg_score": np.mean(scores) if scores else 0,
                "score_std": np.std(scores) if scores else 0,
            }

        except Exception as e:
            self.logger.error("Error calculating optimization metrics: %s", e)
            return {}

    def compare_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Compare multiple backtest results.

        Args:
            results: List of backtest result dictionaries

        Returns:
            Comparison analysis
        """
        if not results:
            return {}

        try:
            # Extract metrics from all results
            all_metrics = []
            for result in results:
                if result.get("metrics"):
                    all_metrics.append(result["metrics"])

            if not all_metrics:
                return {}

            # Calculate statistics across results
            metric_names = set()
            for metrics in all_metrics:
                metric_names.update(metrics.keys())

            comparison = {}
            for metric in metric_names:
                values = [m.get(metric, 0) for m in all_metrics if metric in m]
                if values:
                    comparison[f"{metric}_mean"] = np.mean(values)
                    comparison[f"{metric}_std"] = np.std(values)
                    comparison[f"{metric}_min"] = np.min(values)
                    comparison[f"{metric}_max"] = np.max(values)
                    comparison[f"{metric}_median"] = np.median(values)

            # Ranking analysis
            if "total_return" in metric_names:
                returns = [m.get("total_return", 0) for m in all_metrics]
                comparison["best_performer_idx"] = np.argmax(returns)
                comparison["worst_performer_idx"] = np.argmin(returns)

            return comparison

        except Exception as e:
            self.logger.error("Error comparing results: %s", e)
            return {}

    def _calculate_annualized_return(
        self, equity_curve: pd.Series, initial_capital: float
    ) -> float:
        """Calculate annualized return."""
        if len(equity_curve) < 2:
            return 0

        total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if total_days <= 0:
            return 0

        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
        years = total_days / 365.25

        if years <= 0:
            return 0

        return ((1 + total_return) ** (1 / years) - 1) * 100

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0

        return returns.std() * np.sqrt(252) * 100  # Assuming daily returns

    def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0

        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)

    def _calculate_sortino_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0

        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0

        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)

    def _calculate_calmar_ratio(
        self, equity_curve: pd.Series, initial_capital: float
    ) -> float:
        """Calculate Calmar ratio."""
        annualized_return = self._calculate_annualized_return(
            equity_curve, initial_capital
        )
        max_drawdown = abs(self._calculate_max_drawdown(equity_curve))

        if max_drawdown == 0:
            return 0

        return annualized_return / max_drawdown

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        if len(equity_curve) < 2:
            return 0

        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min() * 100

    def _calculate_max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        if len(equity_curve) < 2:
            return 0

        peak = equity_curve.expanding().max()
        drawdown = equity_curve < peak

        # Find consecutive drawdown periods
        drawdown_periods = []
        current_period = 0

        for is_drawdown in drawdown:
            if is_drawdown:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0

        if current_period > 0:
            drawdown_periods.append(current_period)

        return max(drawdown_periods) if drawdown_periods else 0

    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) < 2:
            return 0

        return np.percentile(returns, confidence * 100) * 100

    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) < 2:
            return 0

        var = np.percentile(returns, confidence * 100)
        cvar = returns[returns <= var].mean()
        return cvar * 100

    def _calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0

        return stats.skew(returns)

    def _calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate excess kurtosis of returns."""
        if len(returns) < 4:
            return 0

        return stats.kurtosis(returns)

    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> dict[str, float]:
        """Calculate trade-specific metrics."""
        if trades.empty:
            return {
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "num_trades": 0,
                "avg_trade_duration": 0,
                "expectancy": 0,
            }

        # Filter trades with PnL information
        trades_with_pnl = (
            trades[trades["pnl"] != 0] if "pnl" in trades.columns else pd.DataFrame()
        )

        if trades_with_pnl.empty:
            return {
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "num_trades": len(trades),
                "avg_trade_duration": 0,
                "expectancy": 0,
            }

        pnl_values = trades_with_pnl["pnl"]
        winning_trades = pnl_values[pnl_values > 0]
        losing_trades = pnl_values[pnl_values < 0]

        num_winning = len(winning_trades)
        len(losing_trades)
        total_trades = len(pnl_values)

        win_rate = (num_winning / total_trades * 100) if total_trades > 0 else 0

        gross_profit = winning_trades.sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades.sum()) if not losing_trades.empty else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

        avg_win = winning_trades.mean() if not winning_trades.empty else 0
        avg_loss = losing_trades.mean() if not losing_trades.empty else 0

        largest_win = winning_trades.max() if not winning_trades.empty else 0
        largest_loss = losing_trades.min() if not losing_trades.empty else 0

        expectancy = pnl_values.mean() if not pnl_values.empty else 0

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "num_trades": total_trades,
            "expectancy": expectancy,
        }

    def _calculate_risk_metrics(
        self, returns: pd.Series, equity_curve: pd.Series
    ) -> dict[str, float]:
        """Calculate additional risk metrics."""
        if len(returns) < 2:
            return {}

        # Beta calculation (simplified, using market proxy)
        # For now, return 1.0 as placeholder
        beta = 1.0

        # Tracking error (simplified)
        tracking_error = returns.std() * np.sqrt(252) * 100

        # Information ratio (simplified)
        information_ratio = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        )

        return {
            "beta": beta,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
        }

    def _calculate_effective_number_assets(self, weights: dict[str, float]) -> float:
        """Calculate effective number of assets (Herfindahl index)."""
        if not weights:
            return 0

        weight_values = list(weights.values())
        sum_squared_weights = sum(w**2 for w in weight_values)
        return 1 / sum_squared_weights if sum_squared_weights > 0 else 0

    def _calculate_diversification_ratio(self, weights: dict[str, float]) -> float:
        """Calculate diversification ratio."""
        if not weights:
            return 0

        # Simplified calculation - would need correlation matrix for full calculation
        num_assets = len(weights)
        equal_weight = 1.0 / num_assets

        # Calculate deviation from equal weighting
        weight_values = list(weights.values())
        return 1 - sum(abs(w - equal_weight) for w in weight_values) / 2

    def _calculate_convergence_speed(self, best_scores: list[float]) -> float:
        """Calculate how quickly optimization converged."""
        if len(best_scores) < 2:
            return 0

        # Find the generation where 95% of final improvement was achieved
        final_score = best_scores[-1]
        initial_score = best_scores[0]
        target_improvement = (final_score - initial_score) * 0.95

        for i, score in enumerate(best_scores):
            if score - initial_score >= target_improvement:
                return i / len(best_scores)

        return 1.0

    def _calculate_population_diversity(self, population: list[dict]) -> float:
        """Calculate diversity in final population."""
        if len(population) < 2:
            return 0

        # Calculate variance in scores as proxy for diversity
        scores = [p.get("score", 0) for p in population if "score" in p]
        if not scores:
            return 0

        return np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0

    def _calculate_improvement_rate(self, best_scores: list[float]) -> float:
        """Calculate rate of improvement over optimization."""
        if len(best_scores) < 2:
            return 0

        improvements = [
            best_scores[i] - best_scores[i - 1] for i in range(1, len(best_scores))
        ]
        positive_improvements = [imp for imp in improvements if imp > 0]

        return len(positive_improvements) / len(improvements) if improvements else 0

    def _calculate_stability_ratio(self, best_scores: list[float]) -> float:
        """Calculate stability of optimization (low variance in later generations)."""
        if len(best_scores) < 10:
            return 0

        # Compare variance in first half vs second half
        mid_point = len(best_scores) // 2
        first_half_var = np.var(best_scores[:mid_point])
        second_half_var = np.var(best_scores[mid_point:])

        if first_half_var == 0:
            return 1.0 if second_half_var == 0 else 0.0

        return 1 - (second_half_var / first_half_var)

    def _calculate_exploration_ratio(self, all_scores: list[float]) -> float:
        """Calculate how well the optimization explored the search space."""
        if len(all_scores) < 2:
            return 0

        # Calculate ratio of unique scores to total evaluations
        unique_scores = len(set(all_scores))
        total_scores = len(all_scores)

        return unique_scores / total_scores

    def _get_zero_metrics(self) -> dict[str, float]:
        """Return dictionary of zero metrics for failed calculations."""
        return {
            "total_return": 0,
            "annualized_return": 0,
            "volatility": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "calmar_ratio": 0,
            "max_drawdown": 0,
            "max_drawdown_duration": 0,
            "var_95": 0,
            "cvar_95": 0,
            "skewness": 0,
            "kurtosis": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "num_trades": 0,
            "avg_trade_duration": 0,
            "expectancy": 0,
        }
