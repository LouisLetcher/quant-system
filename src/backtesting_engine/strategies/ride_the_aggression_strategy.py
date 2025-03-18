from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class RideTheAggressionStrategy(BaseStrategy):
    """
    Ride the Aggression Strategy.

    Uses a comparative symbol (e.g., SPY) to confirm market direction and Bollinger Bands for entry signals.

    Enters long when:
    1. Price crosses above upper Bollinger Band (aggressive move)
    2. Comparative symbol is bullish (above its long-term SMA)

    Exits long when:
    1. Comparative symbol turns bearish
    2. Trailing stop is hit

    Enters short when:
    1. Price crosses below lower Bollinger Band (aggressive move)
    2. Comparative symbol is bearish (below its long-term SMA)

    Exits short when:
    1. Comparative symbol turns bullish
    2. Trailing stop is hit
    """

    # Define parameters that can be optimized
    comparative_symbol = "SPY"
    long_term_ma_period = 200
    bb_length = 15
    bb_std_dev = 3.0
    trail_perc = 0.4  # 0.4% trailing stop

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Note: In a real implementation, you would need to fetch the comparative symbol data
        # This is a simplified version assuming you have access to the comparative data
        # self.comparative_data = fetch_data(self.comparative_symbol)

        # For demonstration purposes, we'll assume the comparative data is available
        # In a real implementation, you would need to modify this to fetch actual data
        self.comparative_close = self.data.Close  # Placeholder

        # Calculate Long-Term SMA for current symbol
        self.long_term_sma = self.I(
            lambda x: self._calculate_sma(x, self.long_term_ma_period), self.data.Close
        )

        # Calculate Long-Term SMA for comparative symbol
        self.comparative_sma = self.I(
            lambda x: self._calculate_sma(x, self.long_term_ma_period),
            self.comparative_close,
        )

        # Calculate Bollinger Bands
        self.bb_middle, self.bb_upper, self.bb_lower = self.I(
            self._calculate_bollinger_bands,
            self.data.Close,
            self.bb_length,
            self.bb_std_dev,
        )

        # Initialize trailing stops
        self.long_trailing_stop = None
        self.short_trailing_stop = None

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(self.long_term_ma_period, self.bb_length):
            return

        # Determine comparative trend conditions
        is_comparative_bullish = self.comparative_close[-1] > self.comparative_sma[-1]
        is_comparative_bearish = self.comparative_close[-1] < self.comparative_sma[-1]

        # Entry conditions
        # Long: Price crosses above upper Bollinger Band and Comparative Symbol is bullish
        # Note: We use the previous bar's BB to avoid look-ahead bias
        long_condition = (
            self.data.Close[-1] > self.bb_upper[-2]
        ) and is_comparative_bullish

        # Short: Price crosses below lower Bollinger Band and Comparative Symbol is bearish
        short_condition = (
            self.data.Close[-1] < self.bb_lower[-2]
        ) and is_comparative_bearish

        # Exit conditions
        long_exit_condition = is_comparative_bearish
        short_exit_condition = is_comparative_bullish

        # Update trailing stops
        if self.position and self.position.is_long:
            # Initialize trailing stop if not set
            if self.long_trailing_stop is None:
                self.long_trailing_stop = self.data.Close[-1] * (
                    1 - self.trail_perc / 100
                )
            else:
                # Update trailing stop to higher value
                self.long_trailing_stop = max(
                    self.long_trailing_stop,
                    self.data.Close[-1] * (1 - self.trail_perc / 100),
                )
        else:
            self.long_trailing_stop = None

        if self.position and self.position.is_short:
            # Initialize trailing stop if not set
            if self.short_trailing_stop is None:
                self.short_trailing_stop = self.data.Close[-1] * (
                    1 + self.trail_perc / 100
                )
            else:
                # Update trailing stop to lower value
                self.short_trailing_stop = min(
                    self.short_trailing_stop,
                    self.data.Close[-1] * (1 + self.trail_perc / 100),
                )
        else:
            self.short_trailing_stop = None

        # Check if trailing stop is hit
        long_stop_hit = (
            self.position
            and self.position.is_long
            and self.data.Close[-1] <= self.long_trailing_stop
        )
        short_stop_hit = (
            self.position
            and self.position.is_short
            and self.data.Close[-1] >= self.short_trailing_stop
        )

        # Long entry logic
        if not self.position and long_condition:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 Close: {self.data.Close[-1]}, Upper BB: {self.bb_upper[-2]}")
            print(
                f"📈 Comparative is bullish: {self.comparative_close[-1]} > {self.comparative_sma[-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

            # Initialize trailing stop
            self.long_trailing_stop = price * (1 - self.trail_perc / 100)
            print(f"🛑 Initial trailing stop set at: {self.long_trailing_stop}")

        # Short entry logic
        elif not self.position and short_condition:
            print(f"🔴 SHORT ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 Close: {self.data.Close[-1]}, Lower BB: {self.bb_lower[-2]}")
            print(
                f"📉 Comparative is bearish: {self.comparative_close[-1]} < {self.comparative_sma[-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.sell(size=size)

            # Initialize trailing stop
            self.short_trailing_stop = price * (1 + self.trail_perc / 100)
            print(f"🛑 Initial trailing stop set at: {self.short_trailing_stop}")

        # Long exit logic
        elif (
            self.position
            and self.position.is_long
            and (long_exit_condition or long_stop_hit)
        ):
            if long_exit_condition:
                print(f"🟡 LONG EXIT TRIGGERED at price: {self.data.Close[-1]}")
                print(
                    f"📉 Comparative turned bearish: {self.comparative_close[-1]} < {self.comparative_sma[-1]}"
                )
            else:
                print(f"🛑 LONG TRAILING STOP HIT at price: {self.data.Close[-1]}")
                print(
                    f"📉 Close: {self.data.Close[-1]}, Trailing Stop: {self.long_trailing_stop}"
                )

            self.position.close()
            self.long_trailing_stop = None

        # Short exit logic
        elif (
            self.position
            and self.position.is_short
            and (short_exit_condition or short_stop_hit)
        ):
            if short_exit_condition:
                print(f"🟡 SHORT EXIT TRIGGERED at price: {self.data.Close[-1]}")
                print(
                    f"📈 Comparative turned bullish: {self.comparative_close[-1]} > {self.comparative_sma[-1]}"
                )
            else:
                print(f"🛑 SHORT TRAILING STOP HIT at price: {self.data.Close[-1]}")
                print(
                    f"📈 Close: {self.data.Close[-1]}, Trailing Stop: {self.short_trailing_stop}"
                )

            self.position.close()
            self.short_trailing_stop = None

        # Update trailing stop log
        if (
            self.position
            and self.position.is_long
            and self.long_trailing_stop is not None
        ):
            print(f"🔄 Updated long trailing stop: {self.long_trailing_stop}")

        if (
            self.position
            and self.position.is_short
            and self.short_trailing_stop is not None
        ):
            print(f"🔄 Updated short trailing stop: {self.short_trailing_stop}")

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

    def _calculate_bollinger_bands(self, prices, length=20, std_dev=2.0):
        """
        Calculate Bollinger Bands

        Args:
            prices: Price series
            length: Bollinger Bands period
            std_dev: Number of standard deviations

        Returns:
            Tuple of (middle band, upper band, lower band)
        """
        # Convert to pandas Series if not already
        prices = pd.Series(prices)

        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=length).mean()

        # Calculate standard deviation
        rolling_std = prices.rolling(window=length).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)

        return middle_band, upper_band, lower_band
