from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class TrendRiskProtectionStrategy(BaseStrategy):
    """
    Trend Risk Protection Strategy.

    Enters long when:
    1. Close is above the long-term SMA for the current and previous 'confirmation_bars' bars

    Exits when:
    1. Close is below the long-term SMA for the current and previous 'confirmation_bars' bars

    The strategy aims to follow established trends while providing protection against false signals
    by requiring multiple bars of confirmation.
    """

    # Define parameters that can be optimized
    sma_length = 200
    confirmation_bars = 4

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate long-term SMA
        self.long_term_sma = self.I(
            lambda x: self._calculate_sma(x, self.sma_length), self.data.Close
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= self.sma_length + self.confirmation_bars:
            return

        # Check if close is above/below SMA for the required number of bars
        bars_above_sma = self._count_bars_above_sma()
        bars_below_sma = self._count_bars_below_sma()

        # Long condition: Close is above SMA for current and previous 'confirmation_bars' bars
        long_condition = bars_above_sma >= (self.confirmation_bars + 1)

        # Sell condition: Close is below SMA for current and previous 'confirmation_bars' bars
        sell_condition = bars_below_sma >= (self.confirmation_bars + 1)

        # Entry logic: Enter long when conditions are met
        if not self.position and long_condition:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📈 Close: {self.data.Close[-1]}, Long-Term SMA({self.sma_length}): {self.long_term_sma[-1]}"
            )
            print(
                f"📊 Bars above SMA: {bars_above_sma}, Confirmation required: {self.confirmation_bars + 1}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Exit logic: Close position when sell conditions are met
        elif self.position and sell_condition:
            print(f"🔴 SELL SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📉 Close: {self.data.Close[-1]}, Long-Term SMA({self.sma_length}): {self.long_term_sma[-1]}"
            )
            print(
                f"📊 Bars below SMA: {bars_below_sma}, Confirmation required: {self.confirmation_bars + 1}"
            )
            self.position.close()

    def _count_bars_above_sma(self):
        """
        Count consecutive bars where close is above long-term SMA

        Returns:
            Number of consecutive bars where close is above long-term SMA
        """
        count = 0
        for i in range(len(self.data) - 1, -1, -1):
            if (
                i >= len(self.long_term_sma)
                or self.data.Close[i] < self.long_term_sma[i]
            ):
                break
            count += 1
        return count

    def _count_bars_below_sma(self):
        """
        Count consecutive bars where close is below long-term SMA

        Returns:
            Number of consecutive bars where close is below long-term SMA
        """
        count = 0
        for i in range(len(self.data) - 1, -1, -1):
            if (
                i >= len(self.long_term_sma)
                or self.data.Close[i] > self.long_term_sma[i]
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
