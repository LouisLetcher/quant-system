from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class DonchianChannelsStrategy(BaseStrategy):
    """
    Donchian Channels Breakout Strategy.

    Enters long when:
    1. Price closes above the upper Donchian channel (highest high of the last 'period' bars)

    Exits when:
    1. Price closes below the exit level (lowest low of the last 'exit_period' bars)

    The strategy uses previous bar's data to avoid lookahead bias.
    """

    # Define parameters that can be optimized
    period = 20  # Donchian lookback period

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate Donchian Channels
        # Note: Using previous bar's data to avoid lookahead bias
        self.upper_channel = self.I(
            lambda x: self._calculate_highest(x.shift(1), self.period), self.data.High
        )

        self.lower_channel = self.I(
            lambda x: self._calculate_lowest(x.shift(1), self.period), self.data.Low
        )

        self.mid_channel = self.I(
            lambda x, y: (x + y) / 2, self.upper_channel, self.lower_channel
        )

        # Calculate exit level
        # Use a longer period for exit to allow for trend continuation
        self.exit_period = self.period * 2
        self.exit_level = self.I(
            lambda x: self._calculate_lowest(x.shift(1), self.exit_period),
            self.data.Low,
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(self.period, self.exit_period) + 1:  # +1 for the shift
            return

        # Entry condition: Price closes above the upper Donchian channel
        long_condition = self.data.Close[-1] > self.upper_channel[-1]

        # Exit condition: Price closes below the exit level
        exit_condition = self.data.Close[-1] < self.exit_level[-1]

        # Entry logic: Enter long when price closes above the upper channel
        if not self.position and long_condition:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Close: {self.data.Close[-1]}, Upper Channel: {self.upper_channel[-1]}"
            )
            print("📈 Price has broken above the upper Donchian channel")

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Exit logic: Close position when price closes below the exit level
        elif self.position and exit_condition:
            print(f"🔴 EXIT TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 Close: {self.data.Close[-1]}, Exit Level: {self.exit_level[-1]}")
            print(
                f"📉 Price has closed below the exit level (lowest low of the last {self.exit_period} bars)"
            )
            self.position.close()

    def _calculate_highest(self, prices, period):
        """
        Calculate highest value over a period

        Args:
            prices: Price series
            period: Lookback period

        Returns:
            Highest values over the lookback period
        """
        return pd.Series(prices).rolling(window=period).max()

    def _calculate_lowest(self, prices, period):
        """
        Calculate lowest value over a period

        Args:
            prices: Price series
            period: Lookback period

        Returns:
            Lowest values over the lookback period
        """
        return pd.Series(prices).rolling(window=period).min()
