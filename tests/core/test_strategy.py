"""
Tests for the strategy module.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.strategy import (
    BaseStrategy,
    SimpleBuyAndHoldStrategy,
    StrategyFactory,
    create_strategy,
    list_available_strategies,
)


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def __init__(self, custom_param: str = "default") -> None:
        super().__init__("MockStrategy")
        self.parameters = {"custom_param": custom_param}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mock signals."""
        # Simple alternating signals for testing
        signals = [1 if i % 2 == 0 else -1 for i in range(len(data))]
        return pd.Series(signals, index=data.index)


class TestBaseStrategy:
    """Test BaseStrategy class."""

    def test_init(self):
        """Test BaseStrategy initialization."""
        strategy = MockStrategy()
        assert strategy.name == "MockStrategy"
        assert isinstance(strategy.parameters, dict)
        assert strategy.parameters["custom_param"] == "default"

    def test_init_with_custom_param(self):
        """Test BaseStrategy initialization with custom parameters."""
        strategy = MockStrategy(custom_param="custom_value")
        assert strategy.parameters["custom_param"] == "custom_value"

    def test_get_strategy_info(self):
        """Test get_strategy_info method."""
        strategy = MockStrategy()
        info = strategy.get_strategy_info()

        assert isinstance(info, dict)
        assert info["name"] == "MockStrategy"
        assert info["type"] == "Base"
        assert "parameters" in info
        assert "description" in info

    def test_validate_data_valid(self):
        """Test validate_data with valid OHLCV data."""
        strategy = MockStrategy()
        data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        assert strategy.validate_data(data) is True

    def test_validate_data_invalid(self):
        """Test validate_data with invalid data."""
        strategy = MockStrategy()

        # Missing volume column
        data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
            }
        )

        assert strategy.validate_data(data) is False

    def test_generate_signals(self):
        """Test signal generation."""
        strategy = MockStrategy()
        data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        signals = strategy.generate_signals(data)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(data)
        assert all(signal in [-1, 0, 1] for signal in signals)


class TestSimpleBuyAndHoldStrategy:
    """Test SimpleBuyAndHoldStrategy class."""

    def test_init(self):
        """Test SimpleBuyAndHoldStrategy initialization."""
        strategy = SimpleBuyAndHoldStrategy()
        assert strategy.name == "Simple Buy and Hold"
        assert strategy.parameters == {}

    def test_generate_signals_empty_data(self):
        """Test signal generation with empty data."""
        strategy = SimpleBuyAndHoldStrategy()
        data = pd.DataFrame()

        signals = strategy.generate_signals(data)
        assert isinstance(signals, pd.Series)
        assert len(signals) == 0

    def test_generate_signals_single_row(self):
        """Test signal generation with single row."""
        strategy = SimpleBuyAndHoldStrategy()
        data = pd.DataFrame(
            {
                "open": [100],
                "high": [105],
                "low": [95],
                "close": [102],
                "volume": [1000],
            }
        )

        signals = strategy.generate_signals(data)
        assert len(signals) == 1
        assert signals.iloc[0] == 1  # Buy signal at start

    def test_generate_signals_multiple_rows(self):
        """Test signal generation with multiple rows."""
        strategy = SimpleBuyAndHoldStrategy()
        data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        signals = strategy.generate_signals(data)
        assert len(signals) == 3
        assert signals.iloc[0] == 1  # Buy signal at start
        assert signals.iloc[1] == 0  # Hold
        assert signals.iloc[2] == 0  # Hold

    def test_get_strategy_info(self):
        """Test get_strategy_info method."""
        strategy = SimpleBuyAndHoldStrategy()
        info = strategy.get_strategy_info()

        assert info["name"] == "Simple Buy and Hold"
        assert info["type"] == "Base"
        assert info["parameters"] == {}


class TestStrategyFactory:
    """Test StrategyFactory class."""

    def test_builtin_strategies_list(self):
        """Test that builtin strategies are properly registered."""
        assert "BuyAndHold" in StrategyFactory.BUILTIN_STRATEGIES
        assert (
            StrategyFactory.BUILTIN_STRATEGIES["BuyAndHold"] is SimpleBuyAndHoldStrategy
        )

    def test_create_builtin_strategy(self):
        """Test creating builtin strategy."""
        strategy = StrategyFactory.create_strategy("BuyAndHold")
        assert isinstance(strategy, SimpleBuyAndHoldStrategy)
        assert strategy.name == "Simple Buy and Hold"

    def test_create_builtin_strategy_with_parameters(self):
        """Test creating builtin strategy with parameters."""
        # SimpleBuyAndHoldStrategy doesn't take parameters, but test the flow
        strategy = StrategyFactory.create_strategy("BuyAndHold", {})
        assert isinstance(strategy, SimpleBuyAndHoldStrategy)

    def test_create_nonexistent_strategy(self):
        """Test creating nonexistent strategy raises ValueError."""
        with pytest.raises(
            ValueError, match="Strategy 'NonExistentStrategy' not found"
        ):
            StrategyFactory.create_strategy("NonExistentStrategy")

    def test_list_strategies(self):
        """Test listing available strategies."""
        strategies = StrategyFactory.list_strategies()

        assert isinstance(strategies, dict)
        assert "builtin" in strategies
        assert "external" in strategies
        assert "all" in strategies

        assert "BuyAndHold" in strategies["builtin"]
        assert "BuyAndHold" in strategies["all"]

    def test_get_strategy_info_builtin(self):
        """Test getting strategy info for builtin strategy."""
        info = StrategyFactory.get_strategy_info("BuyAndHold")

        assert isinstance(info, dict)
        assert info["name"] == "Simple Buy and Hold"

    def test_get_strategy_info_nonexistent(self):
        """Test getting strategy info for nonexistent strategy."""
        with pytest.raises(
            ValueError, match="Strategy 'NonExistentStrategy' not found"
        ):
            StrategyFactory.get_strategy_info("NonExistentStrategy")


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_strategy(self):
        """Test create_strategy convenience function."""
        strategy = create_strategy("BuyAndHold")
        assert isinstance(strategy, SimpleBuyAndHoldStrategy)

    def test_create_strategy_with_parameters(self):
        """Test create_strategy with parameters."""
        strategy = create_strategy("BuyAndHold", {})
        assert isinstance(strategy, SimpleBuyAndHoldStrategy)

    def test_list_available_strategies(self):
        """Test list_available_strategies convenience function."""
        strategies = list_available_strategies()

        assert isinstance(strategies, dict)
        assert "builtin" in strategies
        assert "external" in strategies
        assert "all" in strategies
        assert "BuyAndHold" in strategies["all"]


class TestIntegration:
    """Integration tests for strategy module."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from creation to signal generation."""
        # List available strategies
        strategies = list_available_strategies()
        assert "BuyAndHold" in strategies["all"]

        # Get strategy info
        info = StrategyFactory.get_strategy_info("BuyAndHold")
        assert info["name"] == "Simple Buy and Hold"

        # Create strategy
        strategy = create_strategy("BuyAndHold")
        assert isinstance(strategy, SimpleBuyAndHoldStrategy)

        # Create test data
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [102, 103, 104, 105, 106],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        # Validate data
        assert strategy.validate_data(data) is True

        # Generate signals
        signals = strategy.generate_signals(data)
        assert len(signals) == 5
        assert signals.iloc[0] == 1  # Buy at start
        assert all(signals.iloc[1:] == 0)  # Hold the rest

    def test_strategy_with_custom_parameters_flow(self):
        """Test strategy workflow with custom parameters."""
        # Test that parameter passing works through the factory
        # SimpleBuyAndHoldStrategy doesn't accept parameters, so test with empty dict
        strategy = create_strategy("BuyAndHold", {})
        assert isinstance(strategy, SimpleBuyAndHoldStrategy)

        # Test with None parameters (should work)
        strategy2 = create_strategy("BuyAndHold", None)
        assert isinstance(strategy2, SimpleBuyAndHoldStrategy)

        # The test verifies the parameter passing mechanism works

    def test_multiple_strategy_instances(self):
        """Test creating multiple instances of the same strategy."""
        strategy1 = create_strategy("BuyAndHold")
        strategy2 = create_strategy("BuyAndHold")

        # Should be separate instances
        assert strategy1 is not strategy2
        assert isinstance(strategy1, SimpleBuyAndHoldStrategy)
        assert isinstance(strategy2, SimpleBuyAndHoldStrategy)

        # Should have same configuration
        assert strategy1.name == strategy2.name
        assert strategy1.parameters == strategy2.parameters
