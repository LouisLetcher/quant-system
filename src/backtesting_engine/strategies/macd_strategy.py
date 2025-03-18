from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy.

    Enters long when:
    1. MACD histogram is positive (MACD line above signal line)
    2. Price is above the SMA filter

    Exits when:
    1. Price falls below the SMA filter
    """

    # Define parameters that can be optimized
    fast_length = 50
    slow_length = 75
    signal_length = 35
    sma_length = 250

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate MACD components
        self.macd_line, self.signal_line, self.histogram = self.I(
            self._calculate_macd,
            self.data.Close,
            self.fast_length,
            self.slow_length,
            self.signal_length,
        )

        # Calculate SMA filter
        self.sma = self.I(
            lambda x: self._calculate_sma(x, self.sma_length), self.data.Close
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(
            self.fast_length, self.slow_length, self.signal_length, self.sma_length
        ):
            return

        # Entry condition: MACD histogram is positive and price is above the SMA
        histogram_positive = self.histogram[-1] > 0
        price_above_sma = self.data.Close[-1] > self.sma[-1]

        long_condition = histogram_positive and price_above_sma

        # Exit condition: price falls below the SMA
        exit_condition = self.data.Close[-1] < self.sma[-1]

        # Entry logic: Enter long when conditions are met
        if not self.position and long_condition:
            print(f"🟢 BUY SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 MACD Histogram: {self.histogram[-1]} (positive)")
            print(
                f"📈 Close: {self.data.Close[-1]}, SMA({self.sma_length}): {self.sma[-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Exit logic: Close position when price falls below the SMA
        elif self.position and exit_condition:
            print(f"🔴 SELL SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📉 Close: {self.data.Close[-1]}, SMA({self.sma_length}): {self.sma[-1]}"
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

    def _calculate_macd(self, prices, fast_length, slow_length, signal_length):
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            prices: Price series
            fast_length: Fast EMA period
            slow_length: Slow EMA period
            signal_length: Signal EMA period

        Returns:
            Tuple of (MACD line, signal line, histogram)
        """
        # Convert to pandas Series if not already
        prices = pd.Series(prices)

        # Calculate fast and slow EMAs
        fast_ema = self._calculate_ema(prices, fast_length)
        slow_ema = self._calculate_ema(prices, slow_length)

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line
        signal_line = self._calculate_ema(macd_line, signal_length)

        # Calculate histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram
