from src.backtesting_engine.strategies.base_strategy import BaseStrategy
from src.backtesting_engine.strategies.mean_reversion import MeanReversionStrategy
from src.backtesting_engine.strategies.momentum import MomentumStrategy

class StrategyFactory:
    """Factory class to create strategy instances dynamically."""
    
    strategies = {
        "mean_reversion": MeanReversionStrategy,
        "momentum": MomentumStrategy,
    }

    @staticmethod
    def get_strategy(strategy_name):
        """Returns the correct strategy class based on name."""
        return StrategyFactory.strategies.get(strategy_name, BaseStrategy)