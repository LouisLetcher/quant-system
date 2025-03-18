from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class MovingAverageTrendStrategy(BaseStrategy):
    """
    Moving Average Trend Strategy.

    A trend following strategy that trades based on the 200-day moving average crossover.

    Enters long when:
    1. Price crosses above the 200-day moving average

    Exits when:
    1. Price crosses below the 200-day moving average

    This strategy is designed to capture long-term trends in the market while
    using leverage to amplify returns.
    """

    # Define parameters that can be optimized
    ma_length = 200

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate the 200-day moving average
        self.ma200 = self.I(
            lambda x: self._calculate_sma(x, self.ma_length), self.data.Close
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= self.ma_length:
            return

        # Define the buy and sell conditions
        # Buy: Price crosses above the 200-day MA
        buy_condition = (
            self.data.Close[-2] <= self.ma200[-2]
            and self.data.Close[-1] > self.ma200[-1]
        )

        # Sell: Price crosses below the 200-day MA
        sell_condition = (
            self.data.Close[-2] >= self.ma200[-2]
            and self.data.Close[-1] < self.ma200[-1]
        )

        # Entry logic: Enter long when price crosses above the 200-day MA
        if not self.position and buy_condition:
            print(f"🟢 BUY SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📈 Close: {self.data.Close[-1]}, 200-day MA: {self.ma200[-1]}")
            print("📊 Price crossed above the 200-day moving average")

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size, comment="Buy Signal")

        # Exit logic: Close position when price crosses below the 200-day MA
        elif self.position and sell_condition:
            print(f"🔴 SELL SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📉 Close: {self.data.Close[-1]}, 200-day MA: {self.ma200[-1]}")
            print("📊 Price crossed below the 200-day moving average")
            self.position.close(comment="Sell Signal")

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
