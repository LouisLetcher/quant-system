from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class CounterPunchStrategy(BaseStrategy):
    """
    Counter Punch Strategy.

    Enters long when:
    1. RSI is below 10 (extremely oversold)
    2. Price is above the Long-Term SMA (200)

    Exits long when:
    1. Price crosses above the Short-Term SMA (9)

    Enters short when:
    1. RSI is above 90 (extremely overbought)
    2. Price is below the Long-Term SMA (200)

    Exits short when:
    1. Price crosses below the Short-Term SMA (9)
    """

    # Define parameters that can be optimized
    rsi_period = 2
    ma_period = 9
    long_term_ma_period = 200

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate RSI
        self.rsi = self.I(self._calculate_rsi, self.data.Close, self.rsi_period)

        # Calculate Short-Term SMA
        self.short_term_sma = self.I(
            lambda x: self._calculate_sma(x, self.ma_period), self.data.Close
        )

        # Calculate Long-Term SMA
        self.long_term_sma = self.I(
            lambda x: self._calculate_sma(x, self.long_term_ma_period), self.data.Close
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(
            self.rsi_period, self.ma_period, self.long_term_ma_period
        ):
            return

        # Long entry condition: RSI below 10 and price above Long-Term SMA
        rsi_oversold = self.rsi[-1] < 10
        price_above_long_term_sma = self.data.Close[-1] > self.long_term_sma[-1]

        long_condition = rsi_oversold and price_above_long_term_sma

        # Long exit condition: Price crosses above Short-Term SMA
        # We need to check if the previous close was below the SMA and current close is above
        long_exit_condition = self.data.Close[-1] > self.short_term_sma[-1]

        # Short entry condition: RSI above 90 and price below Long-Term SMA
        rsi_overbought = self.rsi[-1] > 90
        price_below_long_term_sma = self.data.Close[-1] < self.long_term_sma[-1]

        short_condition = rsi_overbought and price_below_long_term_sma

        # Short exit condition: Price crosses below Short-Term SMA
        # We need to check if the previous close was above the SMA and current close is below
        short_exit_condition = self.data.Close[-1] < self.short_term_sma[-1]

        # Long entry logic
        if not self.position and long_condition:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 RSI: {self.rsi[-1]} (below 10)")
            print(
                f"📈 Close: {self.data.Close[-1]}, Long-Term SMA({self.long_term_ma_period}): {self.long_term_sma[-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Long exit logic
        elif self.position and self.position.is_long and long_exit_condition:
            print(f"🟡 LONG EXIT TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📈 Close: {self.data.Close[-1]}, Short-Term SMA({self.ma_period}): {self.short_term_sma[-1]}"
            )
            self.position.close()

        # Short entry logic
        elif not self.position and short_condition:
            print(f"🔴 SHORT ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 RSI: {self.rsi[-1]} (above 90)")
            print(
                f"📉 Close: {self.data.Close[-1]}, Long-Term SMA({self.long_term_ma_period}): {self.long_term_sma[-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.sell(size=size)

        # Short exit logic
        elif self.position and self.position.is_short and short_exit_condition:
            print(f"🟡 SHORT EXIT TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📉 Close: {self.data.Close[-1]}, Short-Term SMA({self.ma_period}): {self.short_term_sma[-1]}"
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

    def _calculate_rsi(self, prices, length=14):
        """
        Calculate Relative Strength Index (RSI)

        Args:
            prices: Price series
            length: RSI period

        Returns:
            RSI values
        """
        # Convert to pandas Series if not already
        prices = pd.Series(prices)

        # Calculate price changes
        deltas = prices.diff()

        # Calculate gains and losses
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)

        # Calculate average gains and losses
        avg_gain = gains.rolling(window=length).mean()
        avg_loss = losses.rolling(window=length).mean()

        # Calculate RS
        rs = avg_gain / avg_loss.where(avg_loss != 0, 1)

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi
