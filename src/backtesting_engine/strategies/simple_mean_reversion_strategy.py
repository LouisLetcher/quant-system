from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class SimpleMeanReversionStrategy(BaseStrategy):
    """
    Simple Mean Reversion Strategy.

    This strategy calculates the mean and standard deviation of closing prices over a
    specified lookback period to determine upper and lower thresholds. It generates
    long and short signals based on price deviations from these thresholds.

    Enters long when:
    1. Price falls below the lower threshold (mean - threshold_multiplier * stdDev)

    Enters short when:
    1. Price rises above the upper threshold (mean + threshold_multiplier * stdDev)

    Exits long when:
    1. Price rises above the upper threshold

    Exits short when:
    1. Price falls below the lower threshold
    """

    # Define parameters that can be optimized
    lookback = 30
    threshold_multiplier = 2.0

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate the mean (SMA)
        self.mean = self.I(
            lambda x: self._calculate_sma(x, self.lookback), self.data.Close
        )

        # Calculate the standard deviation
        self.std_dev = self.I(
            lambda x: self._calculate_std_dev(x, self.lookback), self.data.Close
        )

        # Calculate upper and lower thresholds
        self.upper_threshold = self.I(
            lambda x, y: x + self.threshold_multiplier * y, self.mean, self.std_dev
        )

        self.lower_threshold = self.I(
            lambda x, y: x - self.threshold_multiplier * y, self.mean, self.std_dev
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= self.lookback:
            return

        # Generating signals
        long_condition = self.data.Close[-1] < self.lower_threshold[-1]
        short_condition = self.data.Close[-1] > self.upper_threshold[-1]

        # Entry logic for long positions
        if long_condition and not self.position:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Close: {self.data.Close[-1]}, Lower Threshold: {self.lower_threshold[-1]}"
            )
            print(
                f"📉 Price has fallen below the lower threshold (mean - {self.threshold_multiplier} * stdDev)"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Entry logic for short positions
        elif short_condition and not self.position:
            print(f"🔴 SHORT ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Close: {self.data.Close[-1]}, Upper Threshold: {self.upper_threshold[-1]}"
            )
            print(
                f"📈 Price has risen above the upper threshold (mean + {self.threshold_multiplier} * stdDev)"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.sell(size=size)

        # Exit logic for long positions
        elif self.position and self.position.is_long and short_condition:
            print(f"🟡 LONG EXIT TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Close: {self.data.Close[-1]}, Upper Threshold: {self.upper_threshold[-1]}"
            )
            print("📈 Price has risen above the upper threshold")
            self.position.close()

        # Exit logic for short positions
        elif self.position and self.position.is_short and long_condition:
            print(f"🟡 SHORT EXIT TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Close: {self.data.Close[-1]}, Lower Threshold: {self.lower_threshold[-1]}"
            )
            print("📉 Price has fallen below the lower threshold")
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

    def _calculate_std_dev(self, prices, period):
        """
        Calculate Standard Deviation

        Args:
            prices: Price series
            period: Period for standard deviation calculation

        Returns:
            Standard deviation values
        """
        return pd.Series(prices).rolling(window=period).std()
