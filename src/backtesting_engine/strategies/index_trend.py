from src.backtesting_engine.strategies.base_strategy import BaseStrategy
import pandas as pd

class IndexTrendStrategy(BaseStrategy):
    """
    Improved Index Trend Strategy based on SMA crossovers.
    Buys when fast SMA crosses above slow SMA and sells when fast SMA crosses below slow SMA.
    """

    # Define parameters that can be optimized
    fast_sma_period = 57
    slow_sma_period = 194

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate Fast and Slow Simple Moving Averages
        self.fast_sma = self.I(lambda x: pd.Series(x).rolling(self.fast_sma_period).mean(), self.data.Close)
        self.slow_sma = self.I(lambda x: pd.Series(x).rolling(self.slow_sma_period).mean(), self.data.Close)

    def next(self):
        """Trading logic for each bar."""
        # Check for crossovers using the last two periods
        fast_crossed_above_slow = self.fast_sma[-2] < self.slow_sma[-2] and self.fast_sma[-1] > self.slow_sma[-1]
        fast_crossed_below_slow = self.fast_sma[-2] > self.slow_sma[-2] and self.fast_sma[-1] < self.slow_sma[-1]

        # If we don't have a position and fast SMA crosses above slow SMA
        if not self.position and fast_crossed_above_slow:
            print(f"🟢 BUY SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📈 Fast SMA: {self.fast_sma[-1]}, Slow SMA: {self.slow_sma[-1]}")

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # If we have a position and fast SMA crosses below slow SMA
        elif self.position and fast_crossed_below_slow:
            print(f"🔴 SELL SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📉 Fast SMA: {self.fast_sma[-1]}, Slow SMA: {self.slow_sma[-1]}")
            self.position.close()  # Use close() on the position object
