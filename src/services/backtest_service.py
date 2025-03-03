import logging
import backtrader as bt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestService:
    """Handles backtest execution using Backtrader."""

    def __init__(self, strategy_class):
        self.strategy_class = strategy_class
        self.cerebro = bt.Cerebro()

    def run_backtest(self, data):
        """
        Executes a backtest.

        :param data: Backtrader-compatible data feed.
        :return: Backtest results.
        """
        try:
            self.cerebro.addstrategy(self.strategy_class)
            self.cerebro.adddata(data)
            logger.info("🚀 Running backtest...")
            results = self.cerebro.run()
            logger.info("✅ Backtest complete.")
            return results
        except Exception as e:
            logger.error(f"❌ Error during backtest execution: {e}")
            return None