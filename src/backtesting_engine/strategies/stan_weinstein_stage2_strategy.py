from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class StanWeinsteinStage2Strategy(BaseStrategy):
    """
    Stan Weinstein Stage 2 Breakout Strategy.

    Based on Stan Weinstein's four-stage approach to stock market cycles, focusing on Stage 2 (uptrend).

    Enters long when:
    1. Price is above its MA (bullish trend)
    2. Relative Strength (RS) is above 0 (outperforming the comparative symbol)
    3. Current volume is above its MA (strong buying interest)
    4. Price is breaking above recent highest high (breakout)

    Exits when:
    1. Price crosses below its MA (trend reversal)
    """

    # Define parameters that can be optimized
    comparative_ticker = "SPY"
    rs_period = 50
    volume_ma_length = 5
    price_ma_length = 30
    highest_lookback = 52

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Note: In a real implementation, you would need to fetch the comparative symbol data
        # This is a simplified version assuming you have access to the comparative data
        # self.comparative_data = fetch_data(self.comparative_ticker)

        # For demonstration purposes, we'll assume the comparative data is available
        # In a real implementation, you would need to modify this to fetch actual data
        self.comparative_close = self.data.Close  # Placeholder

        # Calculate Relative Strength (RS)
        self.rs_value = self.I(
            self._calculate_relative_strength,
            self.data.Close,
            self.comparative_close,
            self.rs_period,
        )

        # Calculate Volume MA
        self.vol_ma = self.I(
            lambda x: self._calculate_sma(x, self.volume_ma_length), self.data.Volume
        )

        # Calculate Price MA
        self.price_ma = self.I(
            lambda x: self._calculate_sma(x, self.price_ma_length), self.data.Close
        )

        # Calculate Highest High (ignoring current bar)
        self.highest_high = self.I(
            lambda x: self._calculate_highest(x.shift(1), self.highest_lookback),
            self.data.High,
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if (
            len(self.data)
            <= max(
                self.rs_period,
                self.volume_ma_length,
                self.price_ma_length,
                self.highest_lookback,
            )
            + 1
        ):  # +1 for the shift
            return

        # Long entry condition:
        # 1. Price above its MA
        # 2. RS above 0
        # 3. Current volume above its MA
        # 4. Price breaking above recent highest high
        price_above_ma = self.data.Close[-1] > self.price_ma[-1]
        rs_above_zero = self.rs_value[-1] > 0
        volume_above_ma = self.data.Volume[-1] > self.vol_ma[-1]
        price_above_highest = self.data.Close[-1] > self.highest_high[-1]

        long_entry_condition = (
            price_above_ma and rs_above_zero and volume_above_ma and price_above_highest
        )

        # Exit condition: Price crosses below its MA
        long_exit_condition = self.data.Close[-1] < self.price_ma[-1]

        # Entry logic: Enter long when all conditions are met
        if not self.position and long_entry_condition:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 Close: {self.data.Close[-1]}, Price MA: {self.price_ma[-1]}")
            print(f"📊 RS Value: {self.rs_value[-1]} (> 0)")
            print(f"📊 Volume: {self.data.Volume[-1]}, Volume MA: {self.vol_ma[-1]}")
            print(
                f"📊 Close: {self.data.Close[-1]}, Highest High: {self.highest_high[-1]}"
            )
            print("📈 Stan Weinstein Stage 2 Breakout detected")

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Exit logic: Close position when price crosses below its MA
        elif self.position and long_exit_condition:
            print(f"🔴 EXIT TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📉 Close: {self.data.Close[-1]}, Price MA: {self.price_ma[-1]}")
            print("📉 Price crossed below its MA")
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

    def _calculate_relative_strength(self, base_close, comparative_close, period):
        """
        Calculate Relative Strength (RS)

        RS = (baseClose/baseClose[rsPeriod]) / (comparativeClose/comparativeClose[rsPeriod]) - 1
        A value above 0 indicates the base asset is outperforming the comparative asset.

        Args:
            base_close: Close prices of the base asset
            comparative_close: Close prices of the comparative asset
            period: Lookback period for RS calculation

        Returns:
            RS values
        """
        # Convert to pandas Series if not already
        base_close = pd.Series(base_close)
        comparative_close = pd.Series(comparative_close)

        # Calculate RS
        base_ratio = base_close / base_close.shift(period)
        comparative_ratio = comparative_close / comparative_close.shift(period)
        rs = (base_ratio / comparative_ratio) - 1

        return rs
