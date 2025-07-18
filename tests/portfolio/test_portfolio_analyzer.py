"""Test suite for portfolio analyzer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.portfolio.portfolio_analyzer import PortfolioAnalyzer


class TestPortfolioAnalyzer:
    """Test class for portfolio analyzer functionality."""

    @pytest.fixture
    def portfolio_config(self):
        """Create sample portfolio configuration."""
        return {
            "description": "Test portfolio",
            "assets": [
                {"ticker": "AAPL", "commission": 0.001, "initial_capital": 10000},
                {"ticker": "MSFT", "commission": 0.001, "initial_capital": 10000},
            ],
        }

    @pytest.fixture
    def analyzer(self, portfolio_config):
        """Create analyzer instance."""
        return PortfolioAnalyzer(portfolio_config)

    def test_initialization(self, analyzer):
        """Test portfolio analyzer initialization."""
        assert analyzer.portfolio_name == "Test Portfolio"
        assert analyzer.description == "Test portfolio"
        assert len(analyzer.assets) == 2
        assert analyzer.assets[0]["ticker"] == "AAPL"
        assert analyzer.assets[1]["ticker"] == "MSFT"

    @patch("src.portfolio.portfolio_analyzer.DataLoader")
    @patch("src.portfolio.portfolio_analyzer.BacktestEngine")
    def test_run_strategy_on_asset(
        self, mock_backtest_engine, mock_data_loader, analyzer
    ):
        """Test running strategy on a specific asset."""
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
        mock_instance.run.return_value = {
            "equity_curve": pd.Series([10000, 10100, 10200]),
            "trades": [],
            "metrics": {
                "total_return": 200,
                "return_pct": 2.0,
                "sharpe_ratio": 1.5,
                "max_drawdown": 50,
                "win_rate": 60.0,
                "profit_factor": 1.2,
                "trades_count": 5,
                "sortino_ratio": 1.8,
                "calmar_ratio": 2.5,
                "volatility": 0.15,
                "avg_trade_pct": 0.4,
                "best_trade_pct": 2.0,
                "worst_trade_pct": -1.0,
            },
        }
        mock_backtest_engine.return_value = mock_instance

        # Run strategy on asset
        asset = analyzer.assets[0]
        result = analyzer._run_strategy_on_asset(asset, "rsi", "1d")

        # Verify mocks called
        mock_data_loader.load_data.assert_called_once()
        mock_backtest_engine.assert_called_once()
        mock_instance.run.assert_called_once()
        assert result["metrics"]["sharpe_ratio"] == 1.5

    @patch("src.portfolio.portfolio_analyzer.PortfolioAnalyzer._run_strategy_on_asset")
    def test_find_best_strategy_for_asset(self, mock_run_strategy, analyzer):
        """Test finding the best strategy for a specific asset."""
        # Setup mock
        mock_run_strategy.side_effect = [
            {
                "strategy": "rsi",
                "interval": "1d",
                "metrics": {
                    "sharpe_ratio": 1.2,
                    "total_return": 150,
                    "return_pct": 1.5,
                    "max_drawdown": 40,
                    "win_rate": 55.0,
                    "profit_factor": 1.1,
                    "trades_count": 4,
                    "sortino_ratio": 1.4,
                    "calmar_ratio": 2.1,
                    "volatility": 0.12,
                    "avg_trade_pct": 0.3,
                    "best_trade_pct": 1.5,
                    "worst_trade_pct": -0.8,
                },
            },
            {
                "strategy": "macd",
                "interval": "1d",
                "metrics": {
                    "sharpe_ratio": 1.0,
                    "total_return": 100,
                    "return_pct": 1.0,
                    "max_drawdown": 30,
                    "win_rate": 50.0,
                    "profit_factor": 1.0,
                    "trades_count": 3,
                    "sortino_ratio": 1.2,
                    "calmar_ratio": 1.8,
                    "volatility": 0.10,
                    "avg_trade_pct": 0.25,
                    "best_trade_pct": 1.2,
                    "worst_trade_pct": -0.6,
                },
            },
            {
                "strategy": "momentum",
                "interval": "1d",
                "metrics": {
                    "sharpe_ratio": 1.5,
                    "total_return": 200,
                    "return_pct": 2.0,
                    "max_drawdown": 60,
                    "win_rate": 60.0,
                    "profit_factor": 1.3,
                    "trades_count": 6,
                    "sortino_ratio": 1.7,
                    "calmar_ratio": 2.3,
                    "volatility": 0.16,
                    "avg_trade_pct": 0.35,
                    "best_trade_pct": 2.0,
                    "worst_trade_pct": -1.0,
                },
            },
        ]

        # Run test
        asset = analyzer.assets[0]
        result = analyzer._find_best_strategy_for_asset(asset)

        # Assertions
        assert mock_run_strategy.call_count == 3
        assert result["best_strategy"] == "momentum"
        assert result["best_score"] == 1.5
        assert result["best_interval"] == "1d"
