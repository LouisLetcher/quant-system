from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class BitcoinStrategy(BaseStrategy):
    """
    Bitcoin Strategy.

    Enters long when:
    1. Current close is greater than the close 'lookback_period' bars ago
    2. Current close is greater than the SMA of the lookback period

    Exits when:
    1. Current close is less than the close 'lookback_period' bars ago
    2. Current close is less than the SMA of the lookback period
    """

    # Define parameters that can be optimized
    lookback_period = 50

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate SMA for the lookback period
        self.sma = self.I(
            lambda x: self._calculate_sma(x, self.lookback_period), self.data.Close
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= self.lookback_period:
            return

        # Entry conditions
        close_higher_than_past = (
            self.data.Close[-1] > self.data.Close[-self.lookback_period - 1]
        )
        close_higher_than_sma = self.data.Close[-1] > self.sma[-1]

        long_condition = close_higher_than_past and close_higher_than_sma

        # Exit conditions
        close_lower_than_past = (
            self.data.Close[-1] < self.data.Close[-self.lookback_period - 1]
        )
        close_lower_than_sma = self.data.Close[-1] < self.sma[-1]

        exit_condition = close_lower_than_past or close_lower_than_sma

        # Entry logic: Enter long when conditions are met
        if not self.position and long_condition:
            print(f"🟢 BUY SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📈 Current close: {self.data.Close[-1]}, Close {self.lookback_period} bars ago: {self.data.Close[-self.lookback_period-1]}"
            )
            print(
                f"📊 Current close: {self.data.Close[-1]}, SMA({self.lookback_period}): {self.sma[-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Exit logic: Close position when exit conditions are met
        elif self.position and exit_condition:
            print(f"🔴 SELL SIGNAL TRIGGERED at price: {self.data.Close[-1]}")

            if close_lower_than_past:
                print(
                    f"📉 Current close: {self.data.Close[-1]}, Close {self.lookback_period} bars ago: {self.data.Close[-self.lookback_period-1]}"
                )

            if close_lower_than_sma:
                print(
                    f"📉 Current close: {self.data.Close[-1]}, SMA({self.lookback_period}): {self.sma[-1]}"
                )

            self.position.close()

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
