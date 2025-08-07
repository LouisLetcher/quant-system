"""
Tests for the result analyzer module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.result_analyzer import UnifiedResultAnalyzer


class TestUnifiedResultAnalyzer:
    """Test UnifiedResultAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = UnifiedResultAnalyzer()

    def test_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, "logger")

    def test_calculate_metrics_empty_result(self):
        """Test metrics calculation with empty result."""
        backtest_result = {}
        initial_capital = 10000

        metrics = self.analyzer.calculate_metrics(backtest_result, initial_capital)

        # Should return zero metrics for empty result
        assert isinstance(metrics, dict)
        assert metrics["total_return"] == 0
        assert metrics["sharpe_ratio"] == 0

    def test_calculate_metrics_no_equity_curve(self):
        """Test metrics calculation when equity curve is None or empty."""
        backtest_result = {"equity_curve": None, "trades": None}
        initial_capital = 10000

        metrics = self.analyzer.calculate_metrics(backtest_result, initial_capital)

        # Should return zero metrics
        assert metrics["total_return"] == 0
        assert metrics["annualized_return"] == 0

    def test_calculate_metrics_with_simple_equity_curve(self):
        """Test metrics calculation with simple equity curve."""
        # Create simple equity curve: start at 10000, end at 11000 (10% gain)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        equity_values = np.linspace(10000, 11000, 252)
        equity_curve = pd.DataFrame({"equity": equity_values}, index=dates)

        backtest_result = {
            "equity_curve": equity_curve,
            "final_capital": 11000,
            "trades": None,
        }
        initial_capital = 10000

        metrics = self.analyzer.calculate_metrics(backtest_result, initial_capital)

        # Check basic metrics
        assert abs(metrics["total_return"] - 10.0) < 0.1  # 10% return
        assert metrics["sharpe_ratio"] != 0
        assert metrics["volatility"] >= 0
        assert metrics["max_drawdown"] <= 0

    def test_calculate_metrics_with_trades(self):
        """Test metrics calculation including trade data."""
        # Create equity curve
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        equity_values = np.linspace(10000, 12000, 100)
        equity_curve = pd.DataFrame({"equity": equity_values}, index=dates)

        # Create trade data
        trades = pd.DataFrame(
            {
                "entry_date": ["2023-01-01", "2023-02-01", "2023-03-01"],
                "exit_date": ["2023-01-15", "2023-02-15", "2023-03-15"],
                "pnl": [100, 200, -50],
                "return_pct": [1.0, 2.0, -0.5],
            }
        )

        backtest_result = {
            "equity_curve": equity_curve,
            "final_capital": 12000,
            "trades": trades,
        }
        initial_capital = 10000

        metrics = self.analyzer.calculate_metrics(backtest_result, initial_capital)

        # Check that trade metrics are calculated
        assert metrics["num_trades"] == 3
        assert metrics["win_rate"] > 0  # Should have some winning trades
        assert "avg_win" in metrics
        assert "avg_loss" in metrics

    def test_calculate_metrics_with_series_equity_curve(self):
        """Test metrics calculation with equity curve as Series instead of DataFrame."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        equity_curve = pd.Series(np.linspace(10000, 11500, 50), index=dates)

        backtest_result = {
            "equity_curve": equity_curve,
            "final_capital": 11500,
            "trades": None,
        }
        initial_capital = 10000

        metrics = self.analyzer.calculate_metrics(backtest_result, initial_capital)

        # Should work with Series as well
        assert abs(metrics["total_return"] - 15.0) < 0.1  # 15% return
        assert metrics["volatility"] >= 0

    def test_calculate_metrics_exception_handling(self):
        """Test metrics calculation with invalid data that causes exceptions."""
        # Create problematic data
        backtest_result = {
            "equity_curve": "invalid_data",  # This should cause an exception
            "final_capital": 10000,
        }
        initial_capital = 10000

        metrics = self.analyzer.calculate_metrics(backtest_result, initial_capital)

        # Should return zero metrics on exception
        assert isinstance(metrics, dict)
        assert metrics["total_return"] == 0

    def test_calculate_portfolio_metrics_success(self):
        """Test successful portfolio metrics calculation."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        equity_curve = pd.Series(np.cumsum(returns) * 10000 + 10000, index=dates)
        weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}

        portfolio_data = {
            "returns": returns,
            "equity_curve": equity_curve,
            "weights": weights,
        }
        initial_capital = 10000

        metrics = self.analyzer.calculate_portfolio_metrics(
            portfolio_data, initial_capital
        )

        assert isinstance(metrics, dict)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "num_assets" in metrics
        assert metrics["num_assets"] == 3
        assert "concentration_ratio" in metrics
        assert metrics["concentration_ratio"] == 0.4  # Max weight

    def test_calculate_portfolio_metrics_missing_data(self):
        """Test portfolio metrics with missing data."""
        portfolio_data = {"returns": None, "equity_curve": None}
        initial_capital = 10000

        metrics = self.analyzer.calculate_portfolio_metrics(
            portfolio_data, initial_capital
        )

        # Should return zero metrics
        assert metrics["total_return"] == 0

    def test_calculate_optimization_metrics_success(self):
        """Test successful optimization metrics calculation."""
        optimization_results = {
            "optimization_history": [
                {"score": 0.1, "best_score": 0.1},
                {"score": 0.15, "best_score": 0.15},
                {"score": 0.12, "best_score": 0.15},
                {"score": 0.18, "best_score": 0.18},
            ],
            "final_population": [
                {"params": {"a": 1, "b": 2}, "score": 0.18},
                {"params": {"a": 2, "b": 3}, "score": 0.16},
            ],
        }

        metrics = self.analyzer.calculate_optimization_metrics(optimization_results)

        assert isinstance(metrics, dict)
        assert "total_evaluations" in metrics
        assert metrics["total_evaluations"] == 4
        assert "best_score" in metrics
        assert metrics["best_score"] == 0.18
        assert "avg_score" in metrics
        assert "convergence_speed" in metrics

    def test_calculate_optimization_metrics_empty_history(self):
        """Test optimization metrics with empty history."""
        optimization_results = {"optimization_history": []}

        metrics = self.analyzer.calculate_optimization_metrics(optimization_results)

        assert metrics == {}

    def test_compare_results_success(self):
        """Test successful results comparison."""
        results = [
            {"metrics": {"total_return": 10.0, "sharpe_ratio": 1.5}},
            {"metrics": {"total_return": 15.0, "sharpe_ratio": 1.2}},
            {"metrics": {"total_return": 8.0, "sharpe_ratio": 1.8}},
        ]

        comparison = self.analyzer.compare_results(results)

        assert isinstance(comparison, dict)
        assert "total_return_mean" in comparison
        assert abs(comparison["total_return_mean"] - 11.0) < 0.1  # (10+15+8)/3
        assert "total_return_max" in comparison
        assert comparison["total_return_max"] == 15.0
        assert "best_performer_idx" in comparison
        assert comparison["best_performer_idx"] == 1  # Index of best performer

    def test_compare_results_empty_list(self):
        """Test results comparison with empty list."""
        results = []

        comparison = self.analyzer.compare_results(results)

        assert comparison == {}

    def test_compare_results_no_metrics(self):
        """Test results comparison when results have no metrics."""
        results = [
            {"symbol": "AAPL", "strategy": "test"},
            {"symbol": "MSFT", "strategy": "test"},
        ]

        comparison = self.analyzer.compare_results(results)

        assert comparison == {}

    def test_private_methods_edge_cases(self):
        """Test private methods with edge cases."""
        # Test with empty/minimal data
        empty_series = pd.Series([])
        single_value_series = pd.Series([100])

        # These should not crash and return reasonable defaults
        volatility = self.analyzer._calculate_volatility(empty_series)
        assert volatility == 0

        volatility = self.analyzer._calculate_volatility(single_value_series)
        assert volatility == 0

        sharpe = self.analyzer._calculate_sharpe_ratio(empty_series)
        assert sharpe == 0

    def test_calculate_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio calculation when returns have zero standard deviation."""
        # Create returns with zero variance
        returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])

        sharpe = self.analyzer._calculate_sharpe_ratio(returns)

        # Should return 0 when std is 0
        assert sharpe == 0

    def test_calculate_annualized_return(self):
        """Test annualized return calculation."""
        # Create equity curve for one year
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        # 20% total return over one year
        equity_curve = pd.Series(np.linspace(10000, 12000, len(dates)), index=dates)

        annualized = self.analyzer._calculate_annualized_return(equity_curve, 10000)

        # Should be approximately 20% for one year
        assert abs(annualized - 20.0) < 1.0

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create equity curve with a clear drawdown
        equity_values = [10000, 11000, 12000, 10500, 9000, 9500, 11500, 13000]
        dates = pd.date_range("2023-01-01", periods=len(equity_values), freq="D")
        equity_curve = pd.Series(equity_values, index=dates)

        max_dd = self.analyzer._calculate_max_drawdown(equity_curve)

        # Maximum drawdown should be negative (from 12000 to 9000 = -25%)
        assert max_dd < 0
        assert abs(max_dd + 25.0) < 1.0  # Should be approximately -25%

    def test_calculate_var_and_cvar(self):
        """Test VaR and CVaR calculations."""
        # Create returns with known distribution
        np.random.seed(42)  # For reproducible results
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

        var_95 = self.analyzer._calculate_var(returns, 0.05)
        cvar_95 = self.analyzer._calculate_cvar(returns, 0.05)

        # VaR should be negative (represents loss)
        assert var_95 < 0
        # CVaR should be more negative than VaR (expected shortfall)
        assert cvar_95 <= var_95

    def test_get_zero_metrics(self):
        """Test that _get_zero_metrics returns proper structure."""
        zero_metrics = self.analyzer._get_zero_metrics()

        assert isinstance(zero_metrics, dict)
        assert "total_return" in zero_metrics
        assert "sharpe_ratio" in zero_metrics
        assert "volatility" in zero_metrics
        assert zero_metrics["total_return"] == 0
        assert zero_metrics["sharpe_ratio"] == 0

    def test_calculate_skewness_and_kurtosis(self):
        """Test skewness and kurtosis calculations."""
        # Create returns with known characteristics
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))

        skewness = self.analyzer._calculate_skewness(returns)
        kurtosis = self.analyzer._calculate_kurtosis(returns)

        # For normal distribution, skewness should be close to 0
        assert abs(skewness) < 0.5
        # For normal distribution, excess kurtosis should be close to 0
        assert abs(kurtosis) < 1.0

    def test_integration_full_workflow(self):
        """Test complete workflow from equity curve to comprehensive metrics."""
        # Create realistic equity curve with ups and downs
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # Generate realistic returns
        returns = np.random.normal(0.0008, 0.015, 252)  # ~20% annual with 15% vol
        equity_values = np.cumprod(1 + returns) * 10000

        equity_curve = pd.DataFrame({"equity": equity_values}, index=dates)

        # Create some trades
        trades = pd.DataFrame(
            {
                "entry_date": dates[::50][:4],  # Every 50 days
                "exit_date": dates[25::50][:4],
                "pnl": [150, -80, 300, 200],
                "return_pct": [1.5, -0.8, 3.0, 2.0],
            }
        )

        backtest_result = {
            "equity_curve": equity_curve,
            "final_capital": equity_values[-1],
            "trades": trades,
        }

        # Calculate comprehensive metrics
        metrics = self.analyzer.calculate_metrics(backtest_result, 10000)

        # Verify all major metric categories are present
        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "var_95" in metrics
        assert "cvar_95" in metrics
        assert "skewness" in metrics
        assert "kurtosis" in metrics

        # Trade metrics
        assert "num_trades" in metrics
        assert metrics["num_trades"] == 4
        assert "win_rate" in metrics

        # All metrics should be numerical
        for key, value in metrics.items():
            assert isinstance(value, (int, float, np.number)), (
                f"Metric {key} is not numerical: {value}"
            )
            assert not np.isnan(value), f"Metric {key} is NaN"
