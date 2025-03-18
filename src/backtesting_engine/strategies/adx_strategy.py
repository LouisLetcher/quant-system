from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class ADXStrategy(BaseStrategy):
    """
    Average Directional Index (ADX) Strategy.

    Enters long when:
    1. Current close is greater than the close from lookback_days ago (momentum)
    2. ADX is above the threshold (trend strength)
    3. Price is above the 200-period moving average (trend direction)

    Exits when price falls below the 200-period moving average.
    """

    # Define parameters that can be optimized
    lookback_days = 30
    ma_period = 200
    adx_period = 14
    adx_threshold = 50
    di_period = 5

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate 200-period moving average
        self.ma = self.I(
            lambda x: self._calculate_sma(x, self.ma_period), self.data.Close
        )

        # Calculate ADX and DI indicators
        self.adx, self.di_plus, self.di_minus = self.I(
            self._calculate_dmi,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.di_period,
            self.adx_period,
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(self.lookback_days, self.ma_period, self.adx_period):
            return

        # Check for entry conditions
        momentum_condition = (
            self.data.Close[-1] > self.data.Close[-self.lookback_days - 1]
        )
        adx_condition = self.adx[-1] > self.adx_threshold
        ma_condition = self.data.Close[-1] > self.ma[-1]

        # Entry logic: Enter long when all conditions are met
        if not self.position and momentum_condition and adx_condition and ma_condition:
            print(f"🟢 BUY SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 ADX: {self.adx[-1]}, Threshold: {self.adx_threshold}")
            print(
                f"📈 Close: {self.data.Close[-1]}, MA({self.ma_period}): {self.ma[-1]}"
            )
            print(
                f"📉 Current close vs {self.lookback_days} days ago: {self.data.Close[-1]} > {self.data.Close[-self.lookback_days-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Exit logic: Close position when price falls below the moving average
        elif self.position and self.data.Close[-1] < self.ma[-1]:
            print(f"🔴 SELL SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📉 Close: {self.data.Close[-1]}, MA({self.ma_period}): {self.ma[-1]}"
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

    def _calculate_dmi(self, high, low, close, di_period=14, adx_period=14):
        """
        Calculate Directional Movement Index (DMI) components: ADX, DI+, DI-

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            di_period: Period for DI calculation
            adx_period: Period for ADX calculation

        Returns:
            Tuple of (ADX, DI+, DI-)
        """
        # Convert to pandas Series
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=di_period).mean()

        # Plus Directional Movement (+DM)
        plus_dm = high.diff()
        minus_dm = low.diff().multiply(-1)

        # When +DM > -DM and +DM > 0, +DM = +DM, else +DM = 0
        plus_dm = pd.Series(
            [
                (
                    plus_dm.iloc[i]
                    if plus_dm.iloc[i] > minus_dm.iloc[i] and plus_dm.iloc[i] > 0
                    else 0
                )
                for i in range(len(plus_dm))
            ]
        )

        # When -DM > +DM and -DM > 0, -DM = -DM, else -DM = 0
        minus_dm = pd.Series(
            [
                (
                    minus_dm.iloc[i]
                    if minus_dm.iloc[i] > plus_dm.iloc[i] and minus_dm.iloc[i] > 0
                    else 0
                )
                for i in range(len(minus_dm))
            ]
        )

        # Smooth +DM and -DM
        plus_di = 100 * (plus_dm.rolling(window=di_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=di_period).mean() / atr)

        # Directional Index (DX)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)

        # Average Directional Index (ADX)
        adx = dx.rolling(window=adx_period).mean()

        return adx, plus_di, minus_di
