from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class ConfidentTrendStrategy(BaseStrategy):
    """
    Confident Trend Strategy.

    Uses a comparative symbol (e.g., SPY) to confirm trend direction.

    Enters long when:
    1. Current close > highest high in the lookback period (excluding current bar)
    2. Comparative symbol is bullish (above its long-term SMA)

    Exits when:
    1. Comparative symbol turns bearish (below its long-term SMA)
    2. Current close < lowest low in the lookback period (excluding current bar)
    """

    # Define parameters that can be optimized
    comparative_symbol = "SPY"
    long_term_ma_period = 200
    lookback_period = 365

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Note: In a real implementation, you would need to fetch the comparative symbol data
        # This is a simplified version assuming you have access to the comparative data
        # self.comparative_data = fetch_data(self.comparative_symbol)

        # For demonstration purposes, we'll assume the comparative data is available
        # In a real implementation, you would need to modify this to fetch actual data
        self.comparative_close = self.data.Close  # Placeholder

        # Calculate SMA for the comparative symbol
        self.comparative_sma = self.I(
            lambda x: self._calculate_sma(x, self.long_term_ma_period),
            self.comparative_close,
        )

        # Calculate highest high and lowest low for the lookback period
        self.highest_high = self.I(
            lambda x: self._calculate_highest(x, self.lookback_period), self.data.High
        )

        self.lowest_low = self.I(
            lambda x: self._calculate_lowest(x, self.lookback_period), self.data.Low
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(self.long_term_ma_period, self.lookback_period):
            return

        # Determine comparative trend conditions
        is_comparative_bullish = self.comparative_close[-1] > self.comparative_sma[-1]
        is_comparative_bearish = self.comparative_close[-1] < self.comparative_sma[-1]

        # Long entry condition: Current close > highest high in the lookback period (excluding current bar)
        # and comparative symbol is bullish
        close_above_highest = (
            self.data.Close[-1] > self.highest_high[-2]
        )  # Using -2 to exclude current bar
        long_condition = close_above_highest and is_comparative_bullish

        # Long exit condition: Comparative symbol turns bearish or
        # current close < lowest low in the lookback period (excluding current bar)
        close_below_lowest = (
            self.data.Close[-1] < self.lowest_low[-2]
        )  # Using -2 to exclude current bar
        long_exit_condition = is_comparative_bearish or close_below_lowest

        # Entry logic: Enter long when conditions are met
        if not self.position and long_condition:
            print(f"🟢 BUY SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📈 Current close: {self.data.Close[-1]}, Highest high (lookback): {self.highest_high[-2]}"
            )
            print(
                f"📊 Comparative close: {self.comparative_close[-1]}, Comparative SMA: {self.comparative_sma[-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Exit logic: Close position when exit conditions are met
        elif self.position and long_exit_condition:
            print(f"🔴 SELL SIGNAL TRIGGERED at price: {self.data.Close[-1]}")

            if is_comparative_bearish:
                print(
                    f"📉 Comparative turned bearish: {self.comparative_close[-1]} < {self.comparative_sma[-1]}"
                )

            if close_below_lowest:
                print(
                    f"📉 Close below lowest low: {self.data.Close[-1]} < {self.lowest_low[-2]}"
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
