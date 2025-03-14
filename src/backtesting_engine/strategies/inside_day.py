from src.backtesting_engine.strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class InsideDayStrategy(BaseStrategy):
    """
    Inside Day Strategy with RSI exit.
    
    Enters long when today's price range is inside yesterday's range (inside day pattern).
    Exits when RSI exceeds the overbought threshold.
    """

    # Define parameters that can be optimized
    rsi_length = 5
    overbought_level = 80

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate RSI indicator
        self.rsi = self.I(self._calculate_rsi, self.data.Close, self.rsi_length)
        
        # Store high and low for inside day detection
        self.highs = self.data.High
        self.lows = self.data.Low

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) < 2:
            return
        
        # Check for inside day pattern
        # Today's high is lower than yesterday's high and today's low is higher than yesterday's low
        inside_day = (self.highs[-1] < self.highs[-2]) and (self.lows[-1] > self.lows[-2])
        
        # Check if RSI is overbought
        rsi_overbought = self.rsi[-1] > self.overbought_level
        
        # Entry logic: Enter long on inside day pattern
        if not self.position and inside_day:
            print(f"🟢 BUY SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 Inside Day Pattern: Yesterday's Range [{self.lows[-2]}-{self.highs[-2]}], Today's Range [{self.lows[-1]}-{self.highs[-1]}]")
            
            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)
            
        # Exit logic: Close position when RSI is overbought
        elif self.position and rsi_overbought:
            print(f"🔴 SELL SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📈 RSI: {self.rsi[-1]}, Overbought threshold: {self.overbought_level}")
            self.position.close()
    
    def _calculate_rsi(self, prices, length=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Price series
            length: RSI period
            
        Returns:
            RSI values
        """
        # Convert to pandas Series if not already
        prices = pd.Series(prices)
        
        # Calculate price changes
        deltas = prices.diff()
        
        # Calculate gains and losses
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=length).mean()
        avg_loss = losses.rolling(window=length).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss.where(avg_loss != 0, 1)
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
