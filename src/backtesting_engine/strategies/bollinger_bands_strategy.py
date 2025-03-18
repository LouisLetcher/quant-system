from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy.

    Enters long when:
    1. Previous bar's close is below the lower Bollinger Band
    2. Current close is above the SMA

    Uses stop loss and take profit levels based on percentages.
    """

    # Define parameters that can be optimized
    bb_period = 10
    bb_std_dev = 2.0
    sma_period = 200
    stop_loss_perc = 0.10
    take_profit_perc = 0.40

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate Bollinger Bands
        self.bb_middle, self.bb_upper, self.bb_lower = self.I(
            self._calculate_bollinger_bands,
            self.data.Close,
            self.bb_period,
            self.bb_std_dev,
        )

        # Calculate SMA for trend filter
        self.sma = self.I(
            lambda x: self._calculate_sma(x, self.sma_period), self.data.Close
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(self.bb_period, self.sma_period):
            return

        # Entry condition: previous bar's close is below the lower Bollinger Band
        # and current close is above the SMA
        prev_close_below_lower_bb = self.data.Close[-2] < self.bb_lower[-2]
        current_close_above_sma = self.data.Close[-1] > self.sma[-1]

        entry_condition = prev_close_below_lower_bb and current_close_above_sma

        # Entry logic: Enter long when conditions are met
        if not self.position and entry_condition:
            print(f"🟢 BUY SIGNAL TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Previous close: {self.data.Close[-2]}, Lower BB: {self.bb_lower[-2]}"
            )
            print(
                f"📈 Current close: {self.data.Close[-1]}, SMA({self.sma_period}): {self.sma[-1]}"
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

    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2.0):
        """
        Calculate Bollinger Bands

        Args:
            prices: Price series
            period: Bollinger Bands period
            std_dev: Number of standard deviations

        Returns:
            Tuple of (middle band, upper band, lower band)
        """
        # Convert to pandas Series if not already
        prices = pd.Series(prices)

        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=period).mean()

        # Calculate standard deviation
        rolling_std = prices.rolling(window=period).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)

        return middle_band, upper_band, lower_band
