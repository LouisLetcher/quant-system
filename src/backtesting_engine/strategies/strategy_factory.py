from src.backtesting_engine.strategies.base_strategy import BaseStrategy
from src.backtesting_engine.strategies.mean_reversion import MeanReversionStrategy
from src.backtesting_engine.strategies.momentum import MomentumStrategy

class StrategyFactory:
    """Factory class for creating strategy instances."""
    
    _strategies = {
        "mean_reversion": MeanReversionStrategy,
        # Add other strategies here
    }
    
    @classmethod
    def get_strategy(cls, strategy_name):
        """Get a strategy class by name."""
        strategy_class = cls._strategies.get(strategy_name.lower())
        if strategy_class is None:
            print(f"❌ Strategy '{strategy_name}' not found.")
        return strategy_class
