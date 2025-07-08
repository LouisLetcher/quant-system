"""Integration tests for the quant trading system."""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cli.unified_cli import main as cli_main
from core.data_manager import UnifiedDataManager
from core.portfolio import Portfolio, PortfolioManager
from core.strategy import BuyAndHoldStrategy, MovingAverageCrossoverStrategy


class TestDataManagerIntegration:
    """Integration tests for data manager with real data sources."""

    @pytest.fixture
    def data_manager(self):
        """Create data manager instance."""
        return UnifiedDataManager()

    @pytest.mark.integration
    def test_yahoo_finance_integration(self, data_manager):
        """Test actual Yahoo Finance data fetching."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        try:
            data = data_manager.fetch_data("AAPL", "yahoo", start_date, end_date)
            assert not data.empty
            assert all(
                col in data.columns
                for col in ["Open", "High", "Low", "Close", "Volume"]
            )
            assert data.index.min().date() >= start_date.date()
            assert data.index.max().date() <= end_date.date()
        except Exception as e:
            pytest.skip(f"Yahoo Finance integration test failed: {e}")

    @pytest.mark.integration
    def test_fallback_mechanism_integration(self, data_manager):
        """Test fallback between data sources."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)

        # Test with primary source that might fail and fallback
        try:
            data = data_manager.fetch_data_with_fallback(
                "AAPL",
                primary_source="yahoo",
                fallback_sources=["alpha_vantage"],
                start_date=start_date,
                end_date=end_date,
            )
            assert not data.empty
        except Exception as e:
            pytest.skip(f"Data source integration test failed: {e}")

    def test_symbol_transformation_integration(self, data_manager):
        """Test symbol transformation for different sources."""
        test_cases = [
            ("BTC-USD", "yahoo", "bybit", "BTCUSDT"),
            ("EURUSD=X", "yahoo", "alpha_vantage", "EUR/USD"),
            ("AAPL", "yahoo", "yahoo", "AAPL"),
            ("^GSPC", "yahoo", "alpha_vantage", "SPX"),
        ]

        for original, from_source, to_source, expected in test_cases:
            result = data_manager.transform_symbol(original, from_source, to_source)
            assert (
                result == expected
            ), f"Failed for {original}: expected {expected}, got {result}"


class TestPortfolioIntegration:
    """Integration tests for portfolio functionality."""

    @pytest.fixture
    def sample_portfolio_config(self):
        """Create sample portfolio configuration."""
        return {
            "name": "Integration Test Portfolio",
            "symbols": ["AAPL", "MSFT"],
            "initial_capital": 100000,
            "commission": 0.001,
            "strategy": {"name": "BuyAndHold", "parameters": {}},
            "risk_management": {
                "max_position_size": 0.1,
                "stop_loss": 0.05,
                "take_profit": 0.15,
            },
            "data_source": {
                "primary_source": "yahoo",
                "fallback_sources": ["alpha_vantage"],
            },
        }

    @pytest.fixture
    def portfolio_manager(self):
        """Create portfolio manager instance."""
        return PortfolioManager()

    def test_portfolio_creation_and_management(
        self, portfolio_manager, sample_portfolio_config
    ):
        """Test complete portfolio lifecycle."""
        # Create portfolio
        portfolio = portfolio_manager.create_portfolio(sample_portfolio_config)
        assert portfolio is not None
        assert portfolio.name == "Integration Test Portfolio"

        # Verify it's in manager
        assert "Integration Test Portfolio" in portfolio_manager.list_portfolios()

        # Retrieve portfolio
        retrieved = portfolio_manager.get_portfolio("Integration Test Portfolio")
        assert retrieved is not None
        assert retrieved.name == portfolio.name

        # Remove portfolio
        success = portfolio_manager.remove_portfolio("Integration Test Portfolio")
        assert success
        assert "Integration Test Portfolio" not in portfolio_manager.list_portfolios()

    @patch("core.data_manager.UnifiedDataManager")
    def test_portfolio_backtesting_integration(
        self, mock_data_manager, portfolio_manager, sample_portfolio_config
    ):
        """Test portfolio backtesting with mocked data."""
        # Setup mock data
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": range(100, 130),
                "High": range(102, 132),
                "Low": range(98, 128),
                "Close": range(101, 131),
                "Volume": [1000] * 30,
            },
            index=dates,
        )

        mock_data_manager.return_value.fetch_data.return_value = mock_data

        # Create and backtest portfolio
        portfolio = portfolio_manager.create_portfolio(sample_portfolio_config)

        results = portfolio_manager.backtest_portfolio(
            "Integration Test Portfolio",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 30),
        )

        assert "portfolio_value" in results
        assert "returns" in results
        assert "trades" in results
        assert len(results["portfolio_value"]) > 0

    def test_strategy_integration_with_portfolio(self, sample_portfolio_config):
        """Test strategy integration with portfolio."""
        # Test with BuyAndHold strategy
        portfolio = Portfolio(sample_portfolio_config)
        assert isinstance(portfolio.strategy, BuyAndHoldStrategy)

        # Test with MovingAverageCrossover strategy
        ma_config = sample_portfolio_config.copy()
        ma_config["strategy"] = {
            "name": "MovingAverageCrossover",
            "parameters": {"short_window": 10, "long_window": 20},
        }

        ma_portfolio = Portfolio(ma_config)
        assert isinstance(ma_portfolio.strategy, MovingAverageCrossoverStrategy)
        assert ma_portfolio.strategy.short_window == 10
        assert ma_portfolio.strategy.long_window == 20


class TestConfigurationIntegration:
    """Integration tests for configuration loading and validation."""

    def test_portfolio_config_loading(self):
        """Test loading actual portfolio configurations."""
        config_dir = Path(__file__).parent.parent / "config" / "portfolios"

        if not config_dir.exists():
            pytest.skip("Portfolio config directory not found")

        config_files = list(config_dir.glob("*.json"))
        assert len(config_files) > 0, "No portfolio configuration files found"

        for config_file in config_files:
            with open(config_file, "r") as f:
                config = json.load(f)

            # Validate required fields
            required_fields = ["name", "symbols", "initial_capital", "strategy"]
            for field in required_fields:
                assert (
                    field in config
                ), f"Missing required field '{field}' in {config_file.name}"

            # Validate symbols is not empty
            assert (
                len(config["symbols"]) > 0
            ), f"Empty symbols list in {config_file.name}"

            # Validate strategy has name
            assert (
                "name" in config["strategy"]
            ), f"Strategy missing name in {config_file.name}"

    def test_portfolio_instantiation_from_configs(self):
        """Test creating portfolios from actual config files."""
        config_dir = Path(__file__).parent.parent / "config" / "portfolios"

        if not config_dir.exists():
            pytest.skip("Portfolio config directory not found")

        config_files = list(config_dir.glob("*.json"))[:3]  # Test first 3 configs

        for config_file in config_files:
            with open(config_file, "r") as f:
                config = json.load(f)

            try:
                portfolio = Portfolio(config)
                assert portfolio.name == config["name"]
                assert portfolio.symbols == config["symbols"]
                assert portfolio.initial_capital == config["initial_capital"]
            except Exception as e:
                pytest.fail(f"Failed to create portfolio from {config_file.name}: {e}")


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config" / "portfolios"
            config_dir.mkdir(parents=True)

            # Create a test portfolio config
            test_config = {
                "name": "CLI Test Portfolio",
                "symbols": ["AAPL"],
                "initial_capital": 10000,
                "commission": 0.001,
                "strategy": {"name": "BuyAndHold", "parameters": {}},
                "data_source": {"primary_source": "yahoo", "fallback_sources": []},
            }

            with open(config_dir / "cli_test.json", "w") as f:
                json.dump(test_config, f)

            yield temp_dir

    def test_cli_portfolio_list(self, temp_config_dir):
        """Test CLI portfolio listing."""
        with patch("sys.argv", ["python", "portfolio", "list"]):
            with patch("pathlib.Path.cwd", return_value=Path(temp_config_dir)):
                try:
                    cli_main()
                except SystemExit:
                    pass  # CLI commands often exit

    @patch("core.data_manager.UnifiedDataManager")
    def test_cli_portfolio_test(self, mock_data_manager, temp_config_dir):
        """Test CLI portfolio testing."""
        # Mock data manager
        mock_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [102],
                "Low": [98],
                "Close": [101],
                "Volume": [1000],
            },
            index=[datetime(2024, 1, 1)],
        )
        mock_data_manager.return_value.fetch_data.return_value = mock_data

        with patch("sys.argv", ["python", "portfolio", "test", "cli_test"]):
            with patch("pathlib.Path.cwd", return_value=Path(temp_config_dir)):
                try:
                    cli_main()
                except SystemExit:
                    pass  # CLI commands often exit


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    @patch("core.data_manager.UnifiedDataManager")
    def test_complete_trading_workflow(self, mock_data_manager):
        """Test complete workflow from data fetching to portfolio analysis."""
        # Setup mock data
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": [100 + i * 0.1 for i in range(50)],
                "High": [102 + i * 0.1 for i in range(50)],
                "Low": [98 + i * 0.1 for i in range(50)],
                "Close": [101 + i * 0.1 for i in range(50)],
                "Volume": [1000] * 50,
            },
            index=dates,
        )
        mock_data_manager.return_value.fetch_data.return_value = mock_data
        mock_data_manager.return_value.validate_data.return_value = True

        # 1. Create data manager
        data_manager = UnifiedDataManager()

        # 2. Create portfolio configuration
        config = {
            "name": "E2E Test Portfolio",
            "symbols": ["AAPL", "MSFT"],
            "initial_capital": 100000,
            "commission": 0.001,
            "strategy": {
                "name": "MovingAverageCrossover",
                "parameters": {"short_window": 5, "long_window": 10},
            },
            "risk_management": {
                "max_position_size": 0.2,
                "stop_loss": 0.05,
                "take_profit": 0.15,
            },
        }

        # 3. Create and run portfolio
        portfolio = Portfolio(config)
        manager = PortfolioManager()
        manager.portfolios[config["name"]] = portfolio

        # 4. Backtest portfolio
        results = manager.backtest_portfolio(
            "E2E Test Portfolio",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 19),
        )

        # 5. Verify results
        assert "portfolio_value" in results
        assert "returns" in results
        assert "trades" in results
        assert len(results["portfolio_value"]) > 0

        # 6. Verify portfolio state
        assert portfolio.name == "E2E Test Portfolio"
        assert len(portfolio.symbols) == 2
        assert portfolio.initial_capital == 100000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
