import logging
from src.backtesting_engine.strategies.mean_reversion import MeanReversionStrategy
from src.backtesting_engine.strategies.momentum import MomentumStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyService:
    """Handles fetching and instantiating trading strategies."""

    strategies = {
        "mean_reversion": MeanReversionStrategy,
        "momentum": MomentumStrategy
    }

    @staticmethod
    def get_strategy(strategy_name: str):
        """
        Retrieves a strategy class.

        :param strategy_name: Name of the strategy.
        :return: Strategy class or None if not found.
        """
        if strategy_name in StrategyService.strategies:
            logger.info(f"✅ Found strategy: {strategy_name}")
            return StrategyService.strategies[strategy_name]
        else:
            logger.error(f"❌ Strategy {strategy_name} not found.")
            return None