from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class PullbackTradingStrategy(BaseStrategy):
    """
    Pullback Trading Strategy.

    Enters long when:
    1. Price is above the slow SMA (200) - indicating a long-term uptrend
    2. Price is below the fast SMA (50) - indicating a short-term pullback

    Uses stop loss and take profit levels based on percentages.
    """

    # Define parameters that can be optimized
    slow_sma_length = 200
    fast_sma_length = 50
    stop_loss_perc = 0.10
    take_profit_perc = 0.30

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate SMAs
        self.slow_sma = self.I(
            lambda x: self._calculate_sma(x, self.slow_sma_length), self.data.Close
        )

        self.fast_sma = self.I(
            lambda x: self._calculate_sma(x, self.fast_sma_length), self.data.Close
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(self.slow_sma_length, self.fast_sma_length):
            return

        # Entry condition: Price is above slow SMA and below fast SMA
        entry_condition = (self.data.Close[-1] > self.slow_sma[-1]) and (
            self.data.Close[-1] < self.fast_sma[-1]
        )

        # Entry logic: Enter long when conditions are met
        if not self.position and entry_condition:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Close: {self.data.Close[-1]}, Slow SMA({self.slow_sma_length}): {self.slow_sma[-1]}"
            )
            print(
                f"📊 Close: {self.data.Close[-1]}, Fast SMA({self.fast_sma_length}): {self.fast_sma[-1]}"
            )
            print(
                "📈 Pullback detected: Price above long-term trend but pulling back in short-term"
            )

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

    def _calculate_sma(self, prices, period):
        """
        Calculate Simple Moving Average (SMA)

        Args:
            prices: Price series
            period: SMA period

        Returns:
            SMA values
        """
        return pd.Series(prices).rolling(window=period).mean()
