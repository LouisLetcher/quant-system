from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class BullishEngulfingStrategy(BaseStrategy):
    """
    Bullish Engulfing Strategy.

    Enters long when a bullish engulfing pattern is detected:
    1. The previous candle is bearish (close[1] < open[1])
    2. The current candle's close exceeds the previous candle's open (close > open[1])
    3. The current candle's open is below the previous candle's close (open < close[1])

    Exits when:
    1. RSI exceeds the specified threshold
    """

    # Define parameters that can be optimized
    rsi_length = 2
    rsi_exit_threshold = 90

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate RSI
        self.rsi = self.I(self._calculate_rsi, self.data.Close, self.rsi_length)

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) < 2:
            return

        # Check for bullish engulfing pattern
        # 1. The previous candle is bearish (close[1] < open[1])
        # 2. The current candle's close exceeds the previous candle's open (close > open[1])
        # 3. The current candle's open is below the previous candle's close (open < close[1])
        is_bullish_engulfing = (
            self.data.Close[-2] < self.data.Open[-2]  # Previous candle is bearish
            and self.data.Close[-1]
            > self.data.Open[-2]  # Current close exceeds previous open
            and self.data.Open[-1]
            < self.data.Close[-2]  # Current open is below previous close
        )

        # Entry logic: Enter long on a bullish engulfing pattern
        if not self.position and is_bullish_engulfing:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print("📊 Bullish Engulfing Pattern Detected")
            print(
                f"📈 Previous Candle: Open={self.data.Open[-2]}, Close={self.data.Close[-2]}"
            )
            print(
                f"📈 Current Candle: Open={self.data.Open[-1]}, Close={self.data.Close[-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Exit logic: Close position when RSI exceeds the threshold
        elif self.position and self.rsi[-1] > self.rsi_exit_threshold:
            print(f"🔴 EXIT TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 RSI: {self.rsi[-1]}, Exit Threshold: {self.rsi_exit_threshold}")
            self.position.close()

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
