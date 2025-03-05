from src.backtesting_engine.strategies.base_strategy import BaseStrategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np

class MomentumStrategy(BaseStrategy):
    """
    A momentum strategy that buys when price rises above SMA
    and sells when price falls below SMA.
    """
    
    # Define parameters that can be optimized
    sma_period = 50
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    
    def init(self):
        """Initialize strategy indicators."""
        # Calculate Simple Moving Average - convert to pandas Series first
        self.sma = self.I(lambda x: pd.Series(x).rolling(self.sma_period).mean(), self.data.Close)
        
        # Calculate RSI using numpy approach
        self.rsi = self.I(self._calculate_rsi, self.data.Close)
    
    def _calculate_rsi(self, price_array):
        """Calculate RSI using numpy operations instead of pandas"""
        # Convert to numpy array for easier manipulation
        prices = np.array(price_array)
        
        # Calculate price changes
        deltas = np.diff(prices)
        deltas = np.append(deltas, 0)  # Add a zero to maintain array length
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.zeros_like(gains)
        avg_loss = np.zeros_like(losses)
        
        for i in range(self.rsi_period, len(gains)):
            avg_gain[i] = np.mean(gains[i-self.rsi_period+1:i+1])
            avg_loss[i] = np.mean(losses[i-self.rsi_period+1:i+1])
        
        # Calculate RS and RSI
        rs = np.zeros_like(gains)
        # Avoid division by zero
        for i in range(self.rsi_period, len(gains)):
            if avg_loss[i] == 0:
                rs[i] = 100
            else:
                rs[i] = avg_gain[i] / avg_loss[i]
        
        rsi = 100 - (100 / (1 + rs))
        
        # Set initial values to NaN to match pandas behavior
        rsi[:self.rsi_period] = float('nan')
        
        return rsi
    
    def next(self):
        """Trading logic for each bar."""
        # If we don't have any position and price crosses above SMA with RSI below overbought
        if not self.position and self.data.Close[-1] > self.sma[-1] and self.rsi[-1] < self.rsi_overbought:
            print(f"🟢 BUY SIGNAL TRIGGERED")
            print(f"📈 Current Price: {self.data.Close[-1]}, SMA: {self.sma[-1]}, RSI: {self.rsi[-1]}")
            self.buy()
            
        # If we have a position and price crosses below SMA or RSI indicates overbought
        elif self.position and (self.data.Close[-1] < self.sma[-1] or self.rsi[-1] > self.rsi_overbought):
            print(f"🔴 SELL SIGNAL TRIGGERED")
            print(f"📉 Current Price: {self.data.Close[-1]}, SMA: {self.sma[-1]}, RSI: {self.rsi[-1]}")
            self.sell()
