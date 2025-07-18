"""Test suite for portfolio metrics processor."""

from __future__ import annotations

import pandas as pd
import pytest

from src.portfolio.metrics_processor import (
    calculate_calmar_ratio,
    calculate_drawdowns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    ensure_all_metrics_exist,
    extract_detailed_metrics,
)


class TestMetricsProcessor:
    """Test class for metrics processor functionality."""

    @pytest.fixture
    def backtest_result(self):
        """Create sample backtest result for testing."""
        return {
            "equity_curve": pd.Series(
                [10000, 10100, 10200, 10150, 10300, 10250, 10400],
                index=pd.date_range("2023-01-01", periods=7),
            ),
            "trades": [
                {
                    "entry_time": pd.Timestamp("2023-01-02"),
                    "exit_time": pd.Timestamp("2023-01-03"),
                    "entry_price": 100,
                    "exit_price": 102,
                    "size": 10,
                    "pnl": 20,
                    "return_pct": 2.0,
                    "type": "long",
                },
                {
                    "entry_time": pd.Timestamp("2023-01-04"),
                    "exit_time": pd.Timestamp("2023-01-05"),
                    "entry_price": 101.5,
                    "exit_price": 103,
                    "size": 10,
                    "pnl": 15,
                    "return_pct": 1.5,
                    "type": "long",
                },
                {
                    "entry_time": pd.Timestamp("2023-01-05"),
                    "exit_time": pd.Timestamp("2023-01-06"),
                    "entry_price": 103,
                    "exit_price": 102.5,
                    "size": 10,
                    "pnl": -5,
                    "return_pct": -0.5,
                    "type": "long",
                },
            ],
            "initial_capital": 10000,
            "total_return": 400,
            "return_pct": 4.0,
            "max_drawdown": 50,
            "max_drawdown_pct": 0.5,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 1.8,
            "calmar_ratio": 8.0,
            "volatility": 0.15,
            "win_rate": 66.67,
            "profit_factor": 7.0,
            "trades_count": 3,
        }

    def test_extract_detailed_metrics(self, backtest_result):
        """Test extraction of detailed metrics from backtest results."""
        # Call function
        metrics = extract_detailed_metrics(backtest_result, 10000)

        # Check basic metrics
        assert "return_pct" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "trades_count" in metrics

        # Check additional metrics
        assert "sortino_ratio" in metrics
        assert "calmar_ratio" in metrics
        assert "volatility" in metrics
        assert "avg_trade_pct" in metrics
        assert "best_trade_pct" in metrics
        assert "worst_trade_pct" in metrics

        # Check values
        assert metrics["trades_count"] == 3
        assert metrics["win_rate"] == 66.67
        assert metrics["best_trade_pct"] == 2.0
        assert metrics["worst_trade_pct"] == -0.5
        assert metrics["avg_trade_pct"] == 1.0

    def test_ensure_all_metrics_exist(self):
        """Test that all required metrics are present with defaults."""
        # Create incomplete metrics
        incomplete_metrics = {"return_pct": 4.0, "sharpe_ratio": 1.5}

        # Ensure all metrics exist
        complete_metrics = ensure_all_metrics_exist(incomplete_metrics)

        # Check that missing metrics were added with default values
        assert "max_drawdown_pct" in complete_metrics
        assert "win_rate" in complete_metrics
        assert "profit_factor" in complete_metrics
        assert "trades_count" in complete_metrics
        assert "sortino_ratio" in complete_metrics
        assert "calmar_ratio" in complete_metrics

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Create returns series
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.02])

        # Calculate Sharpe ratio
        sharpe = calculate_sharpe_ratio(returns)

        # Check result
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Should be positive for this sample

        # Test with empty returns
        empty_returns = pd.Series([])
        sharpe = calculate_sharpe_ratio(empty_returns)
        assert sharpe == 0.0  # Should return 0 for empty series

    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        # Create returns series
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.02])

        # Calculate Sortino ratio
        sortino = calculate_sortino_ratio(returns)

        # Check result
        assert isinstance(sortino, float)
        assert sortino > 0  # Should be positive for this sample

        # Test with no negative returns
        positive_returns = pd.Series([0.01, 0.02, 0.015, 0.02])
        sortino = calculate_sortino_ratio(positive_returns)
        assert sortino > 0  # Should still be positive

        # Test with empty returns
        empty_returns = pd.Series([])
        sortino = calculate_sortino_ratio(empty_returns)
        assert sortino == 0.0  # Should return 0 for empty series

    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        # Create returns series and max drawdown
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.02])
        max_drawdown = 0.05  # 5% max drawdown

        # Calculate Calmar ratio
        calmar = calculate_calmar_ratio(returns, max_drawdown)

        # Check result
        assert isinstance(calmar, float)

        # Test with zero drawdown
        calmar = calculate_calmar_ratio(returns, 0)
        assert calmar == 0.0  # Should return 0 for zero drawdown

    def test_calculate_drawdowns(self):
        """Test drawdown calculation."""
        # Create equity curve
        equity_curve = pd.Series(
            [10000, 10100, 10200, 10150, 10300, 10250, 10400],
            index=pd.date_range("2023-01-01", periods=7),
        )

        # Calculate drawdowns
        drawdowns = calculate_drawdowns(equity_curve)

        # Check result
        assert isinstance(drawdowns, pd.Series)
        assert len(drawdowns) == len(equity_curve)
        assert (drawdowns <= 0).all()  # All drawdowns should be <= 0

        # Check max drawdown
        max_dd = drawdowns.min()
        assert max_dd < 0  # Should be negative

        # Test with empty equity curve
        empty_equity = pd.Series([])
        drawdowns = calculate_drawdowns(empty_equity)
        assert drawdowns.empty  # Should return empty series
