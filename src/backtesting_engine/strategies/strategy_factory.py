from src.backtesting_engine.strategies.index_trend import IndexTrendStrategy
from src.backtesting_engine.strategies.inside_day import InsideDayStrategy

class StrategyFactory:
    """Factory class for creating strategy instances."""

    _strategies = {
        "index_trend": IndexTrendStrategy,
        "inside_day": InsideDayStrategy
    }

    @classmethod
    def get_strategy(cls, strategy_name):
        """Get a strategy class by name."""
        strategy_class = cls._strategies.get(strategy_name.lower())
        if strategy_class is None:
            print(f"❌ Strategy '{strategy_name}' not found.")
        return strategy_class

    @classmethod
    def get_available_strategies(cls):
        """Get a list of all available strategy names."""
        return list(cls._strategies.keys())
