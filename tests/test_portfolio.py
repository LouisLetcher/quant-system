"""Unit tests for portfolio module."""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.portfolio import Portfolio, PortfolioManager
from core.strategy import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mock buy signals."""
        return pd.Series([1] * len(data), index=data.index)

    def optimize_parameters(self, data: pd.DataFrame) -> dict:
        """Return mock parameters."""
        return {"test_param": 1.0}


class TestPortfolio:
    """Test suite for Portfolio class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        return pd.DataFrame(
            {
                "Open": range(100, 110),
                "High": range(102, 112),
                "Low": range(98, 108),
                "Close": range(101, 111),
                "Volume": [1000] * 10,
            },
            index=dates,
        )

    @pytest.fixture
    def portfolio(self):
        """Create a portfolio instance for testing."""
        config = {
            "name": "Test Portfolio",
            "symbols": ["AAPL", "GOOGL"],
            "initial_capital": 10000,
            "commission": 0.001,
            "strategy": {"name": "BuyAndHold", "parameters": {}},
            "risk_management": {
                "max_position_size": 0.1,
                "stop_loss": 0.05,
                "take_profit": 0.15,
            },
        }
        return Portfolio(config)

    def test_initialization(self, portfolio):
        """Test proper portfolio initialization."""
        assert portfolio.name == "Test Portfolio"
        assert portfolio.symbols == ["AAPL", "GOOGL"]
        assert portfolio.initial_capital == 10000
        assert portfolio.commission == 0.001
        assert portfolio.current_positions == {}
        assert portfolio.cash == 10000

    def test_calculate_position_size(self, portfolio, sample_data):
        """Test position size calculation."""
        price = 100
        position_size = portfolio.calculate_position_size("AAPL", price)

        # Should respect max_position_size of 10%
        max_value = portfolio.initial_capital * 0.1
        expected_shares = int(max_value / price)
        assert position_size == expected_shares

    def test_execute_buy_order_sufficient_cash(self, portfolio):
        """Test successful buy order execution."""
        symbol = "AAPL"
        shares = 10
        price = 100

        success = portfolio.execute_buy_order(symbol, shares, price)

        assert success
        assert portfolio.current_positions[symbol]["shares"] == shares
        assert portfolio.current_positions[symbol]["avg_price"] == price
        assert portfolio.cash == 10000 - (shares * price) - (shares * price * 0.001)

    def test_execute_buy_order_insufficient_cash(self, portfolio):
        """Test buy order failure due to insufficient cash."""
        symbol = "AAPL"
        shares = 200  # Too many shares for available cash
        price = 100

        success = portfolio.execute_buy_order(symbol, shares, price)

        assert not success
        assert symbol not in portfolio.current_positions
        assert portfolio.cash == 10000

    def test_execute_sell_order_sufficient_shares(self, portfolio):
        """Test successful sell order execution."""
        symbol = "AAPL"

        # First buy some shares
        portfolio.execute_buy_order(symbol, 10, 100)
        initial_cash = portfolio.cash

        # Then sell them
        success = portfolio.execute_sell_order(symbol, 5, 110)

        assert success
        assert portfolio.current_positions[symbol]["shares"] == 5
        # Cash should increase by (shares * price) - commission
        expected_cash_increase = (5 * 110) - (5 * 110 * 0.001)
        assert abs(portfolio.cash - (initial_cash + expected_cash_increase)) < 0.01

    def test_execute_sell_order_insufficient_shares(self, portfolio):
        """Test sell order failure due to insufficient shares."""
        symbol = "AAPL"

        # Buy some shares first
        portfolio.execute_buy_order(symbol, 10, 100)
        initial_positions = portfolio.current_positions[symbol].copy()
        initial_cash = portfolio.cash

        # Try to sell more than we have
        success = portfolio.execute_sell_order(symbol, 15, 110)

        assert not success
        assert (
            portfolio.current_positions[symbol]["shares"] == initial_positions["shares"]
        )
        assert portfolio.cash == initial_cash

    def test_get_portfolio_value(self, portfolio):
        """Test portfolio value calculation."""
        # Add some positions
        portfolio.execute_buy_order("AAPL", 10, 100)
        portfolio.execute_buy_order("GOOGL", 5, 200)

        current_prices = {"AAPL": 110, "GOOGL": 220}
        total_value = portfolio.get_portfolio_value(current_prices)

        expected_value = portfolio.cash + (10 * 110) + (5 * 220)
        assert abs(total_value - expected_value) < 0.01

    def test_get_portfolio_value_no_positions(self, portfolio):
        """Test portfolio value with no positions."""
        current_prices = {}
        total_value = portfolio.get_portfolio_value(current_prices)
        assert total_value == portfolio.initial_capital

    def test_calculate_returns(self, portfolio):
        """Test return calculation."""
        # Simulate portfolio growth
        portfolio.cash = 5000
        portfolio.execute_buy_order("AAPL", 50, 100)  # Invest remaining cash

        current_prices = {"AAPL": 120}
        current_value = portfolio.get_portfolio_value(current_prices)
        returns = portfolio.calculate_returns(current_value)

        expected_return = (
            current_value - portfolio.initial_capital
        ) / portfolio.initial_capital
        assert abs(returns - expected_return) < 0.01

    def test_apply_risk_management_stop_loss(self, portfolio):
        """Test stop loss risk management."""
        symbol = "AAPL"
        portfolio.execute_buy_order(symbol, 10, 100)

        # Price drops below stop loss threshold (5%)
        current_price = 94  # 6% drop

        with patch.object(portfolio, "execute_sell_order") as mock_sell:
            mock_sell.return_value = True
            action = portfolio.apply_risk_management(symbol, current_price)

            assert action == "SELL"
            mock_sell.assert_called_once_with(symbol, 10, current_price)

    def test_apply_risk_management_take_profit(self, portfolio):
        """Test take profit risk management."""
        symbol = "AAPL"
        portfolio.execute_buy_order(symbol, 10, 100)

        # Price rises above take profit threshold (15%)
        current_price = 116  # 16% gain

        with patch.object(portfolio, "execute_sell_order") as mock_sell:
            mock_sell.return_value = True
            action = portfolio.apply_risk_management(symbol, current_price)

            assert action == "SELL"
            mock_sell.assert_called_once_with(symbol, 10, current_price)

    def test_apply_risk_management_hold(self, portfolio):
        """Test holding position within risk thresholds."""
        symbol = "AAPL"
        portfolio.execute_buy_order(symbol, 10, 100)

        # Price within acceptable range
        current_price = 105  # 5% gain
        action = portfolio.apply_risk_management(symbol, current_price)

        assert action == "HOLD"


class TestPortfolioManager:
    """Test suite for PortfolioManager class."""

    @pytest.fixture
    def manager(self):
        """Create a PortfolioManager instance for testing."""
        return PortfolioManager()

    @pytest.fixture
    def portfolio_config(self):
        """Sample portfolio configuration."""
        return {
            "name": "Test Portfolio",
            "symbols": ["AAPL"],
            "initial_capital": 10000,
            "commission": 0.001,
            "strategy": {"name": "BuyAndHold", "parameters": {}},
        }

    def test_create_portfolio(self, manager, portfolio_config):
        """Test portfolio creation."""
        portfolio = manager.create_portfolio(portfolio_config)

        assert isinstance(portfolio, Portfolio)
        assert portfolio.name == "Test Portfolio"
        assert "Test Portfolio" in manager.portfolios

    def test_get_portfolio_existing(self, manager, portfolio_config):
        """Test getting existing portfolio."""
        manager.create_portfolio(portfolio_config)
        retrieved = manager.get_portfolio("Test Portfolio")

        assert retrieved is not None
        assert retrieved.name == "Test Portfolio"

    def test_get_portfolio_nonexistent(self, manager):
        """Test getting non-existent portfolio."""
        retrieved = manager.get_portfolio("Nonexistent")
        assert retrieved is None

    def test_list_portfolios(self, manager, portfolio_config):
        """Test listing all portfolios."""
        manager.create_portfolio(portfolio_config)

        # Create another portfolio
        config2 = portfolio_config.copy()
        config2["name"] = "Test Portfolio 2"
        manager.create_portfolio(config2)

        portfolios = manager.list_portfolios()
        assert len(portfolios) == 2
        assert "Test Portfolio" in portfolios
        assert "Test Portfolio 2" in portfolios

    def test_remove_portfolio(self, manager, portfolio_config):
        """Test portfolio removal."""
        manager.create_portfolio(portfolio_config)
        assert "Test Portfolio" in manager.portfolios

        success = manager.remove_portfolio("Test Portfolio")
        assert success
        assert "Test Portfolio" not in manager.portfolios

    def test_remove_nonexistent_portfolio(self, manager):
        """Test removing non-existent portfolio."""
        success = manager.remove_portfolio("Nonexistent")
        assert not success

    @patch("core.data_manager.UnifiedDataManager")
    def test_backtest_portfolio(self, mock_data_manager, manager, portfolio_config):
        """Test portfolio backtesting."""
        # Setup mock data
        sample_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Low": [99, 100, 101],
                "Close": [101, 102, 103],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        mock_data_manager.return_value.fetch_data.return_value = sample_data

        portfolio = manager.create_portfolio(portfolio_config)

        with patch.object(portfolio.strategy, "generate_signals") as mock_signals:
            mock_signals.return_value = pd.Series([1, 0, -1], index=sample_data.index)

            results = manager.backtest_portfolio(
                "Test Portfolio",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 3),
            )

            assert "portfolio_value" in results
            assert "returns" in results
            assert "trades" in results


if __name__ == "__main__":
    pytest.main([__file__])
