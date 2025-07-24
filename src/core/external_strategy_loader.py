"""
External Strategy Loader

Loads and manages external trading strategies from separate repositories.
Provides unified interface for strategy testing and execution.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExternalStrategyLoader:
    """
    Loads and manages external trading strategies

    Discovers strategy modules from external repositories and provides
    a unified interface for the quant-system to use them.
    """

    def __init__(self, strategies_path: str | None = None):
        """
        Initialize External Strategy Loader

        Args:
            strategies_path: Path to external strategies directory
                           (defaults to ../quant-strategies relative to project root)
        """
        if strategies_path is None:
            # Default to external_strategies directory (mounted in Docker)
            project_root = Path(__file__).parent.parent.parent
            default_strategies_path = project_root / "external_strategies"
            self.strategies_path = default_strategies_path
        else:
            self.strategies_path = Path(strategies_path)
        self.loaded_strategies: dict[str, type] = {}
        self._discover_strategies()

    def _discover_strategies(self) -> None:
        """Discover available strategy modules"""
        if not self.strategies_path.exists():
            logger.warning("Strategies path does not exist: %s", self.strategies_path)
            return

        # Look for Python strategy files
        for strategy_file in self.strategies_path.glob("*.py"):
            if not strategy_file.name.startswith("_") and strategy_file.name != "README.py":
                self._load_strategy_file(strategy_file)

        # Also look for directory-based strategies with adapters
        for strategy_dir in self.strategies_path.iterdir():
            if strategy_dir.is_dir() and not strategy_dir.name.startswith("."):
                self._load_strategy_dir(strategy_dir)

    def _load_strategy_file(self, strategy_file: Path) -> None:
        """
        Load a single strategy from a Python file

        Args:
            strategy_file: Path to strategy Python file
        """
        try:
            # Load the strategy module
            strategy_name = strategy_file.stem
            spec = importlib.util.spec_from_file_location(
                f"external_strategy_{strategy_name}", strategy_file
            )
            if spec is None or spec.loader is None:
                logger.error("Could not load spec for %s", strategy_name)
                return

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"external_strategy_{strategy_name}"] = module
            spec.loader.exec_module(module)

            # Look for strategy class in the module
            strategy_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and attr_name.lower().endswith("strategy")
                    and attr_name != "BaseStrategy"
                ):
                    strategy_class = attr
                    break

            if strategy_class:
                self.loaded_strategies[strategy_name] = strategy_class
                logger.info("Loaded external strategy: %s", strategy_name)
            else:
                logger.warning("No strategy class found in %s", strategy_file.name)

        except Exception as e:
            logger.error("Failed to load strategy %s: %s", strategy_file.name, e)

    def _load_strategy_dir(self, strategy_dir: Path) -> None:
        """
        Load a single strategy from directory

        Args:
            strategy_dir: Path to strategy directory
        """
        try:
            # Look for quant_system adapter
            adapter_path = strategy_dir / "adapters" / "quant_system.py"
            if not adapter_path.exists():
                logger.warning(
                    "No quant_system adapter found for %s", strategy_dir.name
                )
                return

            # Load the adapter module
            spec = importlib.util.spec_from_file_location(
                f"{strategy_dir.name}_adapter", adapter_path
            )
            if spec is None or spec.loader is None:
                logger.error("Could not load spec for %s", strategy_dir.name)
                return

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"{strategy_dir.name}_adapter"] = module
            spec.loader.exec_module(module)

            # Find the adapter class (should end with 'Adapter')
            adapter_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and attr_name.endswith("Adapter")
                    and attr_name != "Adapter"
                ):
                    adapter_class = attr
                    break

            if adapter_class is None:
                logger.error("No adapter class found in %s", strategy_dir.name)
                return

            # Store the strategy
            strategy_name = strategy_dir.name.replace("-", "_")
            self.loaded_strategies[strategy_name] = adapter_class
            logger.info("Loaded strategy: %s", strategy_name)

        except Exception as e:
            logger.error("Failed to load strategy %s: %s", strategy_dir.name, e)

    def get_strategy(self, strategy_name: str, **kwargs: Any) -> Any:
        """
        Get a strategy instance by name

        Args:
            strategy_name: Name of the strategy
            **kwargs: Parameters for strategy initialization

        Returns:
            Strategy adapter instance

        Raises:
            ValueError: If strategy not found
        """
        if strategy_name not in self.loaded_strategies:
            available = list(self.loaded_strategies.keys())
            msg = f"Strategy '{strategy_name}' not found. Available: {available}"
            raise ValueError(msg)

        strategy_class = self.loaded_strategies[strategy_name]
        return strategy_class(**kwargs)

    def list_strategies(self) -> list[str]:
        """Get list of available strategy names"""
        return list(self.loaded_strategies.keys())

    def get_strategy_info(self, strategy_name: str) -> dict[str, Any]:
        """
        Get information about a strategy

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with strategy information
        """
        if strategy_name not in self.loaded_strategies:
            msg = f"Strategy '{strategy_name}' not found"
            raise ValueError(msg)

        # Create a temporary instance to get info
        strategy = self.get_strategy(strategy_name)
        if hasattr(strategy, "get_strategy_info"):
            strategy_info = strategy.get_strategy_info()
            return strategy_info if strategy_info is not None else {}
        return {
            "name": strategy_name,
            "type": "External",
            "parameters": getattr(strategy, "parameters", {}),
            "description": f"External strategy: {strategy_name}",
        }

    def validate_strategy_data(self, strategy_name: str, data: Any) -> bool:
        """
        Validate data for a specific strategy

        Args:
            strategy_name: Name of the strategy
            data: Data to validate

        Returns:
            True if data is valid, False otherwise
        """
        strategy = self.get_strategy(strategy_name)
        if hasattr(strategy, "validate_data"):
            result = strategy.validate_data(data)
            return result if isinstance(result, bool) else True
        return True


# Global strategy loader instance
_strategy_loader: ExternalStrategyLoader | None = None


def get_strategy_loader(strategies_path: str | None = None) -> ExternalStrategyLoader:
    """
    Get global strategy loader instance

    Args:
        strategies_path: Path to strategies directory (only used on first call)

    Returns:
        ExternalStrategyLoader instance
    """
    global _strategy_loader
    if _strategy_loader is None:
        _strategy_loader = ExternalStrategyLoader(strategies_path)
    return _strategy_loader


def load_external_strategy(strategy_name: str, **kwargs: Any) -> Any:
    """
    Convenience function to load an external strategy

    Args:
        strategy_name: Name of the strategy
        **kwargs: Strategy parameters

    Returns:
        Strategy adapter instance
    """
    loader = get_strategy_loader()
    return loader.get_strategy(strategy_name, **kwargs)
