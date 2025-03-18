from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    """
    Relative Strength Index (RSI) Strategy.

    Enters long when:
    1. RSI is below 30 (oversold condition)
    2. Price is above the slow SMA (150)
    3. Price is below the fast SMA (30)

    Uses stop loss and take profit levels based on percentages.
    """

    # Define parameters that can be optimized
    rsi_period = 5
    sma_slow_period = 150
    sma_fast_period = 30
    stop_loss_perc = 0.10
    take_profit_perc = 0.30

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate RSI
        self.rsi = self.I(self._calculate_rsi, self.data.Close, self.rsi_period)

        # Calculate slow SMA
        self.sma_slow = self.I(
            lambda x: self._calculate_sma(x, self.sma_slow_period), self.data.Close
        )

        # Calculate fast SMA
        self.sma_fast = self.I(
            lambda x: self._calculate_sma(x, self.sma_fast_period), self.data.Close
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(
            self.rsi_period, self.sma_slow_period, self.sma_fast_period
        ):
            return

        # Entry conditions
        rsi_oversold = self.rsi[-1] < 30
        price_above_slow_sma = self.data.Close[-1] > self.sma_slow[-1]
        price_below_fast_sma = self.data.Close[-1] < self.sma_fast[-1]

        enter_long = rsi_oversold and price_above_slow_sma and price_below_fast_sma

        # Entry logic: Enter long when all conditions are met
        if not self.position and enter_long:
            print(f"🟢 BUY SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 RSI: {self.rsi[-1]} (below 30)")
            print(
                f"📈 Close: {self.data.Close[-1]}, Slow SMA({self.sma_slow_period}): {self.sma_slow[-1]}"
            )
            print(
                f"📉 Close: {self.data.Close[-1]}, Fast SMA({self.sma_fast_period}): {self.sma_fast[-1]}"
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
