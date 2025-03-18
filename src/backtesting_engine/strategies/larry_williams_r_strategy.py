from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class LarryWilliamsRStrategy(BaseStrategy):
    """
    Larry Williams %R Strategy.

    Enters long when:
    1. The Williams %R indicator was extremely oversold (< -95) 'lookback_index' bars ago
    2. The current Williams %R has recovered above -85

    Uses stop loss and take profit levels based on percentages.
    """

    # Define parameters that can be optimized
    wpr_period = 10
    lookback_index = 5
    stop_loss_perc = 0.10
    take_profit_perc = 0.30

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate Williams %R
        self.wpr = self.I(
            self._calculate_williams_r,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.wpr_period,
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= self.wpr_period + self.lookback_index:
            return

        # Entry condition: WPR was extremely oversold and has now recovered
        entry_condition = (self.wpr[-self.lookback_index - 1] < -95) and (
            self.wpr[-1] > -85
        )

        # Entry logic: Enter long when conditions are met
        if not self.position and entry_condition:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Williams %R {self.lookback_index} bars ago: {self.wpr[-self.lookback_index-1]} (< -95)"
            )
            print(f"📊 Current Williams %R: {self.wpr[-1]} (> -85)")

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)

            # Calculate stop loss and take profit levels
            stop_price = price * (1 - self.stop_loss_perc)
            take_profit_price = price * (1 + self.take_profit_perc)

            print(
                f"🛑 Stop Loss: {stop_price} ({self.stop_loss_perc*100}% below entry)"
            )
            print(
                f"🎯 Take Profit: {take_profit_price} ({self.take_profit_perc*100}% above entry)"
            )

            # Enter position with stop loss and take profit
            self.buy(size=size, sl=stop_price, tp=take_profit_price)

    def _calculate_williams_r(self, high, low, close, period=14):
        """
        Calculate Williams %R

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period

        Returns:
            Williams %R values
        """
        # Convert to pandas Series if not already
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)

        # Calculate highest high and lowest low over the period
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        # Calculate Williams %R
        # Formula: %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)
        williams_r = (
            -100
            * (highest_high - close)
            / (highest_high - lowest_low).replace(0, 0.00001)
        )

        return williams_r
