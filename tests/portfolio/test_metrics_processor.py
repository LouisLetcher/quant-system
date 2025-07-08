import unittest

import numpy as np
import pandas as pd

from src.portfolio.metrics_processor import (
    calculate_calmar_ratio,
    calculate_drawdowns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    ensure_all_metrics_exist,
    extract_detailed_metrics,
)


class TestMetricsProcessor(unittest.TestCase):

    def setUp(self):
        # Create sample backtest result
        self.backtest_result = {
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
        }

        # Add metrics
        self.backtest_result["metrics"] = {
            "return_pct": 4.0,
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": 0.5,
            "win_rate": 66.67,
            "profit_factor": 7.0,
            "trades_count": 3,
        }

    def test_extract_detailed_metrics(self):
        # Call function
        metrics = extract_detailed_metrics(self.backtest_result, 10000)

        # Check basic metrics
        self.assertIn("return_pct", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown_pct", metrics)
        self.assertIn("win_rate", metrics)
        self.assertIn("profit_factor", metrics)
        self.assertIn("trades_count", metrics)

        # Check additional metrics
        self.assertIn("sortino_ratio", metrics)
        self.assertIn("calmar_ratio", metrics)
        self.assertIn("volatility", metrics)
        self.assertIn("avg_trade_pct", metrics)
        self.assertIn("best_trade_pct", metrics)
        self.assertIn("worst_trade_pct", metrics)

        # Check values
        self.assertEqual(metrics["trades_count"], 3)
        self.assertEqual(metrics["win_rate"], 66.67)
        self.assertEqual(metrics["best_trade_pct"], 2.0)
        self.assertEqual(metrics["worst_trade_pct"], -0.5)
        self.assertEqual(metrics["avg_trade_pct"], 1.0)

    def test_ensure_all_metrics_exist(self):
        # Create incomplete metrics
        incomplete_metrics = {"return_pct": 4.0, "sharpe_ratio": 1.5}

        # Call function
        complete_metrics = ensure_all_metrics_exist(incomplete_metrics)

        # Check that missing metrics were added with default values
        self.assertIn("max_drawdown_pct", complete_metrics)
        self.assertIn("win_rate", complete_metrics)
        self.assertIn("profit_factor", complete_metrics)
        self.assertIn("trades_count", complete_metrics)
        self.assertIn("sortino_ratio", complete_metrics)
        self.assertIn("calmar_ratio", complete_metrics)

    def test_calculate_sharpe_ratio(self):
        # Create returns series
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.02])

        # Call function
        sharpe = calculate_sharpe_ratio(returns)

        # Check result
        self.assertIsInstance(sharpe, float)
        self.assertGreater(sharpe, 0)  # Should be positive for this sample

        # Test with empty returns
        empty_returns = pd.Series([])
        sharpe = calculate_sharpe_ratio(empty_returns)
        self.assertEqual(sharpe, 0.0)  # Should return 0 for empty series

    def test_calculate_sortino_ratio(self):
        # Create returns series
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.02])

        # Call function
        sortino = calculate_sortino_ratio(returns)

        # Check result
        self.assertIsInstance(sortino, float)
        self.assertGreater(sortino, 0)  # Should be positive for this sample

        # Test with no negative returns
        positive_returns = pd.Series([0.01, 0.02, 0.015, 0.02])
        sortino = calculate_sortino_ratio(positive_returns)
        self.assertGreater(sortino, 0)  # Should still be positive

        # Test with empty returns
        empty_returns = pd.Series([])
        sortino = calculate_sortino_ratio(empty_returns)
        self.assertEqual(sortino, 0.0)  # Should return 0 for empty series

    def test_calculate_calmar_ratio(self):
        # Create returns series and max drawdown
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.02])
        max_drawdown = 0.05

        # Call function
        calmar = calculate_calmar_ratio(returns, max_drawdown)

        # Check result
        self.assertIsInstance(calmar, float)

        # Test with zero drawdown
        calmar = calculate_calmar_ratio(returns, 0)
        self.assertEqual(calmar, 0.0)  # Should return 0 for zero drawdown

    def test_calculate_drawdowns(self):
        # Create equity curve
        equity_curve = pd.Series(
            [10000, 10100, 10200, 10150, 10300, 10250, 10400],
            index=pd.date_range("2023-01-01", periods=7),
        )

        # Call function
        drawdowns = calculate_drawdowns(equity_curve)

        # Check result
        self.assertIsInstance(drawdowns, pd.Series)
        self.assertEqual(len(drawdowns), len(equity_curve))
        self.assertTrue((drawdowns <= 0).all())  # All drawdowns should be <= 0

        # Check max drawdown
        max_dd = drawdowns.min()
        self.assertLess(max_dd, 0)  # Should be negative

        # Test with empty equity curve
        empty_equity = pd.Series([])
        drawdowns = calculate_drawdowns(empty_equity)
        self.assertTrue(drawdowns.empty)  # Should return empty series


if __name__ == "__main__":
    unittest.main()
