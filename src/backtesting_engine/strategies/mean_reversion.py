from src.backtesting_engine.strategies.base_strategy import BaseStrategy
from backtesting.lib import crossover
import pandas as pd

class MeanReversionStrategy(BaseStrategy):
    """
    A mean reversion strategy that buys when price falls below SMA
    and sells when price rises above SMA.
    """
    
    # Define parameters that can be optimized
    sma_period = 20
    
    def init(self):
        """Initialize strategy indicators."""
        # Calculate Simple Moving Average
        self.sma = self.I(lambda x: pd.Series(x).rolling(self.sma_period).mean(), self.data.Close)
    
    def next(self):
        """Trading logic for each bar."""
        # If we don't have any position and price crosses below SMA, BUY
        if not self.position and self.data.Close[-1] < self.sma[-1]:
            print(f"🟢 BUY SIGNAL TRIGGERED")
            print(f"📉 Current Price: {self.data.Close[-1]}, SMA: {self.sma[-1]}")
            self.buy()
            
        # If we have a position and price crosses above SMA, SELL
        elif self.position and self.data.Close[-1] > self.sma[-1]:
            print(f"🔴 SELL SIGNAL TRIGGERED")
            print(f"📈 Current Price: {self.data.Close[-1]}, SMA: {self.sma[-1]}")
            self.sell()
