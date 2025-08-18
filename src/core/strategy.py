"""
Trading Strategy Framework

Provides base classes and utilities for implementing trading strategies.
Supports both built-in and external strategies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .external_strategy_loader import get_strategy_loader

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies

    All strategies should inherit from this class and implement
    the required methods.
    """

    def __init__(self, name: str):
        """
        Initialize base strategy

        Args:
            name: Strategy name
        """
        self.name = name
        self.parameters: dict[str, Any] = {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series of signals: 1 (buy), -1 (sell), 0 (hold)
        """

    def get_strategy_info(self) -> dict[str, Any]:
        """Get strategy information"""
        return {
            "name": self.name,
            "type": "Base",
            "parameters": self.parameters,
            "description": f"Trading strategy: {self.name}",
        }

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        return all(col in data.columns for col in required_columns)


class BuyAndHoldStrategy(BaseStrategy):
    """
    Simple Buy and Hold Strategy

    Generates a buy signal at the start and holds the position.
    """

    def __init__(self) -> None:
        super().__init__("Buy and Hold")
        self.parameters = {}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate buy and hold signals"""
        signals = [0] * len(data)
        if len(signals) > 0:
            signals[0] = 1  # Buy at the start
        return pd.Series(signals, index=data.index)


class StrategyFactory:
    """
    Factory class for creating strategy instances

    Supports both built-in and external strategies.
    """

    # Built-in strategies
    BUILTIN_STRATEGIES = {"BuyAndHold": BuyAndHoldStrategy}

    @classmethod
    def create_strategy(
        cls, strategy_name: str, parameters: dict[str, Any] | None = None
    ) -> Any:
        """
        Create a strategy instance

        Args:
            strategy_name: Name of the strategy
            parameters: Strategy parameters

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy not found
        """
        if parameters is None:
            parameters = {}

        # Check built-in strategies first
        if strategy_name in cls.BUILTIN_STRATEGIES:
            strategy_class = cls.BUILTIN_STRATEGIES[strategy_name]
            return strategy_class(**parameters)

        # Try external strategies
        try:
            loader = get_strategy_loader()
            return loader.get_strategy(strategy_name, **parameters)
        except ValueError:
            pass

        # Strategy not found
        available_builtin = list(cls.BUILTIN_STRATEGIES.keys())
        available_external = get_strategy_loader().list_strategies()
        available_all = available_builtin + available_external

        msg = f"Strategy '{strategy_name}' not found. Available strategies: {available_all}"
        raise ValueError(msg)

    @classmethod
    def list_strategies(cls) -> dict[str, list[str]]:
        """
        List all available strategies

        Returns:
            Dictionary with 'builtin' and 'external' strategy lists
        """
        builtin = list(cls.BUILTIN_STRATEGIES.keys())
        external = get_strategy_loader().list_strategies()

        return {"builtin": builtin, "external": external, "all": builtin + external}

    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> dict[str, Any]:
        """
        Get information about a strategy

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with strategy information
        """
        # Check built-in strategies
        if strategy_name in cls.BUILTIN_STRATEGIES:
            strategy = cls.create_strategy(strategy_name)
            strategy_info = strategy.get_strategy_info()
            return strategy_info if strategy_info is not None else {}

        # Check external strategies
        try:
            loader = get_strategy_loader()
            return loader.get_strategy_info(strategy_name)
        except ValueError:
            msg = f"Strategy '{strategy_name}' not found"
            raise ValueError(msg)


def create_strategy(
    strategy_name: str, parameters: dict[str, Any] | None = None
) -> Any:
    """
    Convenience function to create a strategy

    Args:
        strategy_name: Name of the strategy
        parameters: Strategy parameters

    Returns:
        Strategy instance
    """
    return StrategyFactory.create_strategy(strategy_name, parameters)


def list_available_strategies() -> dict[str, list[str]]:
    """
    Convenience function to list available strategies

    Returns:
        Dictionary with strategy lists
    """
    return StrategyFactory.list_strategies()
