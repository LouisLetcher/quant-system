from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class MFIStrategy(BaseStrategy):
    """
    Money Flow Index (MFI) Strategy.

    Enters long when:
    1. MFI is below 50 (potential oversold condition)
    2. Price is above the SMA (uptrend confirmation)

    Uses stop loss and take profit levels based on percentages.
    """

    # Define parameters that can be optimized
    mfi_length = 14
    sma_period = 200
    sl_percent = 0.10
    tp_percent = 0.30

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate Money Flow Index
        self.mfi = self.I(
            self._calculate_mfi,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume,
            self.mfi_length,
        )

        # Calculate SMA for trend filter
        self.sma = self.I(
            lambda x: self._calculate_sma(x, self.sma_period), self.data.Close
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(self.mfi_length, self.sma_period):
            return

        # Entry condition: MFI is below 50 and price is above the SMA
        mfi_below_50 = self.mfi[-1] < 50
        price_above_sma = self.data.Close[-1] > self.sma[-1]

        long_condition = mfi_below_50 and price_above_sma

        # Entry logic: Enter long when conditions are met
        if not self.position and long_condition:
            print(f"🟢 BUY SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 MFI: {self.mfi[-1]} (below 50)")
            print(
                f"📈 Close: {self.data.Close[-1]}, SMA({self.sma_period}): {self.sma[-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)

            # Calculate stop loss and take profit levels
            stop_price = price * (1 - self.sl_percent)
            take_profit_price = price * (1 + self.tp_percent)

            print(f"🛑 Stop Loss: {stop_price} ({self.sl_percent*100}% below entry)")
            print(
                f"🎯 Take Profit: {take_profit_price} ({self.tp_percent*100}% above entry)"
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

    def _calculate_mfi(self, high, low, close, volume, length=14):
        """
        Calculate Money Flow Index (MFI)

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            length: MFI period

        Returns:
            MFI values
        """
        # Convert to pandas Series if not already
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        volume = pd.Series(volume)

        # Calculate typical price
        typical_price = (high + low + close) / 3

        # Calculate raw money flow
        raw_money_flow = typical_price * volume

        # Calculate money flow direction
        direction = np.zeros_like(typical_price)
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                direction[i] = 1  # Positive money flow
            elif typical_price.iloc[i] < typical_price.iloc[i - 1]:
                direction[i] = -1  # Negative money flow

        # Calculate positive and negative money flows
        positive_flow = pd.Series(
            [
                raw_money_flow.iloc[i] if direction[i] > 0 else 0
                for i in range(len(raw_money_flow))
            ]
        )

        negative_flow = pd.Series(
            [
                raw_money_flow.iloc[i] if direction[i] < 0 else 0
                for i in range(len(raw_money_flow))
            ]
        )

        # Calculate the sum of positive and negative money flows over the period
        positive_sum = positive_flow.rolling(window=length).sum()
        negative_sum = (
            negative_flow.rolling(window=length).sum().abs()
        )  # Use absolute value

        # Calculate money ratio
        money_ratio = pd.Series(np.zeros_like(positive_sum))
        valid_indices = negative_sum > 0
        money_ratio[valid_indices] = (
            positive_sum[valid_indices] / negative_sum[valid_indices]
        )

        # Calculate MFI
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi
