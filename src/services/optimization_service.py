import logging
import backtrader as bt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationService:
    """Optimizes trading strategies."""

    def __init__(self, strategy_class):
        self.strategy_class = strategy_class
        self.cerebro = bt.Cerebro(optreturn=False)

    def optimize(self, data, **params):
        """
        Runs optimization for a given strategy.

        :param data: Backtrader data feed.
        :param params: Dictionary of optimization parameters.
        :return: Optimization results.
        """
        try:
            self.cerebro.optstrategy(self.strategy_class, **params)
            self.cerebro.adddata(data)
            logger.info(f"🔍 Running optimization with params: {params}")
            results = self.cerebro.run()
            logger.info("✅ Optimization complete.")
            return results
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            return None