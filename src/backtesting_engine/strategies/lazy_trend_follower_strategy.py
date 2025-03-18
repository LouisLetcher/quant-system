from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class LazyTrendFollowerStrategy(BaseStrategy):
    """
    Lazy Trend Follower Strategy.

    Enters long when:
    1. Close is above Moving Average for the current and previous 'confirmation' bars

    Exits when:
    1. Close is below Moving Average for the current and previous 'confirmation' bars

    The strategy aims to follow established trends with minimal effort, hence the name "Lazy Trend Follower".
    """

    # Define parameters that can be optimized
    ma_period = 10
    ma_type = "SMA"  # Options: "SMA", "EMA"
    confirmation = 1  # Number of consecutive bars to confirm trend

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate Moving Average based on type
        if self.ma_type == "SMA":
            self.moving_average = self.I(
                lambda x: self._calculate_sma(x, self.ma_period), self.data.Close
            )
        else:  # EMA
            self.moving_average = self.I(
                lambda x: self._calculate_ema(x, self.ma_period), self.data.Close
            )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= self.ma_period + self.confirmation + 1:
            return

        # Check if close is above moving average for the required number of bars
        bars_above_ma = self._count_bars_above_ma()
        bars_below_ma = self._count_bars_below_ma()

        # Long entry condition: Close is above Moving Average for the current and previous 'confirmation' bars
        long_condition = bars_above_ma >= (self.confirmation + 1)

        # Exit condition: Close is below Moving Average for the current and previous 'confirmation' bars
        exit_condition = bars_below_ma >= (self.confirmation + 1)

        # Entry logic: Enter long when conditions are met
        if not self.position and long_condition:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📈 Close: {self.data.Close[-1]}, {self.ma_type}({self.ma_period}): {self.moving_average[-1]}"
            )
            print(
                f"📊 Bars above MA: {bars_above_ma}, Confirmation required: {self.confirmation + 1}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Exit logic: Close position when exit conditions are met
        elif self.position and exit_condition:
            print(f"🔴 EXIT TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📉 Close: {self.data.Close[-1]}, {self.ma_type}({self.ma_period}): {self.moving_average[-1]}"
            )
            print(
                f"📊 Bars below MA: {bars_below_ma}, Confirmation required: {self.confirmation + 1}"
            )
            self.position.close()

    def _count_bars_above_ma(self):
        """
        Count consecutive bars where close is above moving average

        Returns:
            Number of consecutive bars where close is above moving average
        """
        count = 0
        for i in range(len(self.data) - 1, -1, -1):
            if (
                i >= len(self.moving_average)
                or self.data.Close[i] < self.moving_average[i]
            ):
                break
            count += 1
        return count

    def _count_bars_below_ma(self):
        """
        Count consecutive bars where close is below moving average

        Returns:
            Number of consecutive bars where close is below moving average
        """
        count = 0
        for i in range(len(self.data) - 1, -1, -1):
            if (
                i >= len(self.moving_average)
                or self.data.Close[i] > self.moving_average[i]
            ):
                break
            count += 1
        return count

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

    def _calculate_ema(self, prices, period):
        """
        Calculate Exponential Moving Average (EMA)

        Args:
            prices: Price series
            period: EMA period

        Returns:
            EMA values
        """
        return pd.Series(prices).ewm(span=period, adjust=False).mean()
