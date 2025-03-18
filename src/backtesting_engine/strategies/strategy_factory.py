from __future__ import annotations

import importlib
import inspect
import os
from typing import Dict, Type

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class StrategyFactory:
    """Factory class for creating strategy instances."""

    _strategies: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def _load_strategies(cls):
        """Dynamically load all strategy classes from the strategies folder."""
        if cls._strategies:  # If already loaded, return
            return

        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Get all Python files in the directory (excluding __init__.py and this file)
        strategy_files = [
            f[:-3]
            for f in os.listdir(current_dir)
            if f.endswith(".py")
            and f != "__init__.py"
            and f != "strategy_factory.py"
            and f != "base_strategy.py"
        ]

        # Import each file and add its strategy classes to _strategies
        for file_name in strategy_files:
            try:
                # Import the module
                module = importlib.import_module(
                    f"src.backtesting_engine.strategies.{file_name}"
                )

                # Find all classes in the module that inherit from BaseStrategy
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseStrategy)
                        and obj.__module__ == module.__name__
                        and obj != BaseStrategy
                    ):

                        # Convert CamelCase to snake_case for the strategy name
                        strategy_name = "".join(
                            ["_" + c.lower() if c.isupper() else c for c in name]
                        ).lstrip("_")
                        strategy_name = strategy_name.removesuffix(
                            "_strategy"
                        )  # Remove '_strategy' suffix

                        cls._strategies[strategy_name] = obj
            except Exception as e:
                print(f"❌ Error loading strategy from {file_name}: {e}")

    @classmethod
    def get_strategy(cls, strategy_name):
        """Get a strategy class by name."""
        cls._load_strategies()  # Ensure strategies are loaded
        strategy_class = cls._strategies.get(strategy_name.lower())
        if strategy_class is None:
            print(f"❌ Strategy '{strategy_name}' not found.")
        return strategy_class

    @classmethod
    def get_available_strategies(cls):
        """Get a list of all available strategy names."""
        cls._load_strategies()  # Ensure strategies are loaded
        return list(cls._strategies.keys())
