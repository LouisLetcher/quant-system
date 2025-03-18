from __future__ import annotations

import pandas as pd
import numpy as np

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class WeeklyBreakoutStrategy(BaseStrategy):
    """
    Weekly Breakout Strategy.

    Enters long when:
    1. Price closes above the highest close of the last 'lookback_length' periods
    2. Price is above the SMA (confirming uptrend)

    Uses stop loss and take profit levels based on percentages.
    
    This strategy is designed for stocks, indices, forex, and commodities.
    """

    # Define parameters that can be optimized
    lookback_length = 20
    sma_length = 130
    stop_loss_perc = 0.10
    take_profit_perc = 0.40

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate highest close for breakout detection
        # Using a custom function with offset instead of shift
        self.highest_close = self.I(
            lambda x: self._calculate_highest_with_offset(x, self.lookback_length, 1),
            self.data.Close
        )
        
        # Calculate SMA for trend confirmation
        self.sma_value = self.I(
            lambda x: self._calculate_sma(x, self.sma_length),
            self.data.Close
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(self.lookback_length, self.sma_length) + 1:  # +1 for the offset
            return

        # Entry condition: Price closes above highest close and is above SMA
        entry_condition = (self.data.Close[-1] > self.highest_close[-1]) and (self.data.Close[-1] > self.sma_value[-1])

        # Entry logic: Enter long when conditions are met
        if not self.position and entry_condition:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 Close: {self.data.Close[-1]}, Highest Close ({self.lookback_length} periods): {self.highest_close[-1]}")
            print(f"📊 Close: {self.data.Close[-1]}, SMA({self.sma_length}): {self.sma_value[-1]}")
            print(f"📈 Breakout detected with trend confirmation")

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            
            # Calculate stop loss and take profit levels
            stop_price = price * (1 - self.stop_loss_perc)
            take_profit_price = price * (1 + self.take_profit_perc)
            
            print(f"🛑 Stop Loss: {stop_price} ({self.stop_loss_perc*100}% below entry)")
            print(f"🎯 Take Profit: {take_profit_price} ({self.take_profit_perc*100}% above entry)")
            
            # Enter position with stop loss and take profit
            self.buy(size=size, sl=stop_price, tp=take_profit_price)

    def _calculate_highest_with_offset(self, prices, period, offset=0):
        """
        Calculate highest value over a period with an offset
        
        This function calculates the highest value over a period, but excludes
        the most recent 'offset' number of bars from the calculation.

        Args:
            prices: Price series
            period: Lookback period
            offset: Number of recent bars to exclude (default: 0)

        Returns:
            Highest values over the lookback period
        """
        result = np.full_like(prices, np.nan)
        
        for i in range(period + offset - 1, len(prices)):
            # Calculate highest over the period, excluding the most recent 'offset' bars
            highest = np.max(prices[i-period-offset+1:i-offset+1])
            result[i] = highest
            
        return result

    def _calculate_sma(self, prices, period):
        """
        Calculate Simple Moving Average (SMA)

        Args:
            prices: Price series
            period: SMA period

        Returns:
            SMA values
        """
        result = np.full_like(prices, np.nan)
        
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i-period+1:i+1])
            
        return result
