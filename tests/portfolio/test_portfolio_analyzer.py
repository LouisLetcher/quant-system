import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.backtesting_engine.strategies.strategy_factory import StrategyFactory
from src.portfolio.portfolio_analyzer import PortfolioAnalyzer


class TestPortfolioAnalyzer(unittest.TestCase):

    def setUp(self):
        # Create sample portfolio config
        self.portfolio_config = {
            "description": "Test portfolio",
            "assets": [
                {"ticker": "AAPL", "commission": 0.001, "initial_capital": 10000},
                {"ticker": "MSFT", "commission": 0.001, "initial_capital": 10000},
            ],
        }

        # Create analyzer instance
        self.analyzer = PortfolioAnalyzer(self.portfolio_config)

    def test_initialization(self):
        self.assertEqual(self.analyzer.portfolio_name, "Test Portfolio")
        self.assertEqual(self.analyzer.description, "Test portfolio")
        self.assertEqual(len(self.analyzer.assets), 2)
        self.assertEqual(self.analyzer.assets[0]["ticker"], "AAPL")
        self.assertEqual(self.analyzer.assets[1]["ticker"], "MSFT")

    @patch("src.portfolio.portfolio_analyzer.DataLoader")
    @patch("src.portfolio.portfolio_analyzer.BacktestEngine")
    def test_run_strategy_on_asset(self, mock_backtest_engine, mock_data_loader):
        # Setup mocks
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [98, 99, 100],
                "Close": [103, 104, 105],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )
        mock_data_loader.load_data.return_value = mock_data

        mock_instance = MagicMock()
        mock_backtest_engine.return_value = mock_instance
        mock_instance.run.return_value = {
            "metrics": {
                "sharpe_ratio": 1.5,
                "return_pct": 10.0,
                "max_drawdown_pct": 5.0,
                "win_rate": 60.0,
                "profit_factor": 2.0,
                "trades_count": 10,
            },
            "trades": [],
            "equity_curve": [],
        }

        # Call method
        strategy_class = StrategyFactory.get_strategy("mean_reversion")
        result = self.analyzer._run_strategy_on_asset(
            ticker="AAPL",
            strategy_class=strategy_class,
            interval="1d",
            period="1mo",
            commission=0.001,
            initial_capital=10000,
        )

        # Assertions
        mock_data_loader.load_data.assert_called_with(
            "AAPL", period="1mo", interval="1d"
        )
        mock_backtest_engine.assert_called_once()
        mock_instance.run.assert_called_once()
        self.assertEqual(result["metrics"]["sharpe_ratio"], 1.5)

    @patch("src.portfolio.portfolio_analyzer.PortfolioAnalyzer._run_strategy_on_asset")
    def test_find_best_strategy_for_asset(self, mock_run_strategy):
        # Setup mock
        mock_run_strategy.side_effect = [
            {
                "metrics": {"sharpe_ratio": 1.2},
                "strategy": "mean_reversion",
                "interval": "1d",
            },
            {
                "metrics": {"sharpe_ratio": 1.5},
                "strategy": "momentum",
                "interval": "1d",
            },
            {
                "metrics": {"sharpe_ratio": 1.0},
                "strategy": "breakout",
                "interval": "1d",
            },
        ]

        # Call method
        strategies = ["mean_reversion", "momentum", "breakout"]
        result = self.analyzer._find_best_strategy_for_asset(
            ticker="AAPL",
            strategies=strategies,
            intervals=["1d"],
            period="1mo",
            metric="sharpe",
            commission=0.001,
            initial_capital=10000,
        )

        # Assertions
        self.assertEqual(mock_run_strategy.call_count, 3)
        self.assertEqual(result["best_strategy"], "momentum")
        self.assertEqual(result["best_score"], 1.5)
        self.assertEqual(result["best_interval"], "1d")


if __name__ == "__main__":
    unittest.main()
