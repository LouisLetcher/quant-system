from __future__ import annotations

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class CrudeOilStrategy(BaseStrategy):
    """
    Crude Oil Strategy.

    Enters long when:
    1. Price pattern condition: Inside day pattern in any of the last three days
    2. Bullish condition: Current close > close 'lookback_period' bars ago

    Enters short when:
    1. Price pattern condition: Inside day pattern in any of the last three days
    2. Bearish condition: Current close < close 'lookback_period' bars ago
    """

    # Define parameters that can be optimized
    lookback_period = 50

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Store high and low for inside day detection
        self.highs = self.data.High
        self.lows = self.data.Low

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= self.lookback_period + 3:
            return

        # Check for inside day pattern in any of the last three days
        inside_day_1 = (self.highs[-1] < self.highs[-2]) and (
            self.lows[-1] > self.lows[-2]
        )
        inside_day_2 = (self.highs[-2] < self.highs[-3]) and (
            self.lows[-2] > self.lows[-3]
        )
        inside_day_3 = (self.highs[-3] < self.highs[-4]) and (
            self.lows[-3] > self.lows[-4]
        )

        price_pattern_condition = inside_day_1 or inside_day_2 or inside_day_3

        # Check for bullish/bearish conditions
        bullish_condition = (
            self.data.Close[-1] > self.data.Close[-self.lookback_period - 1]
        )
        bearish_condition = (
            self.data.Close[-1] < self.data.Close[-self.lookback_period - 1]
        )

        # Entry logic for long position
        if not self.position and price_pattern_condition and bullish_condition:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print("📊 Inside Day Pattern detected in one of the last three days")
            print(
                f"📈 Current close: {self.data.Close[-1]}, Close {self.lookback_period} bars ago: {self.data.Close[-self.lookback_period-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Entry logic for short position
        elif not self.position and price_pattern_condition and bearish_condition:
            print(f"🔴 SHORT ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print("📊 Inside Day Pattern detected in one of the last three days")
            print(
                f"📉 Current close: {self.data.Close[-1]}, Close {self.lookback_period} bars ago: {self.data.Close[-self.lookback_period-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.sell(size=size)

    def _describe_inside_day_pattern(self):
        """
        Describe which inside day pattern was detected

        Returns:
            String describing the detected pattern
        """
        inside_day_1 = (self.highs[-1] < self.highs[-2]) and (
            self.lows[-1] > self.lows[-2]
        )
        inside_day_2 = (self.highs[-2] < self.highs[-3]) and (
            self.lows[-2] > self.lows[-3]
        )
        inside_day_3 = (self.highs[-3] < self.highs[-4]) and (
            self.lows[-3] > self.lows[-4]
        )

        if inside_day_1:
            return f"Today's Range [{self.lows[-1]}-{self.highs[-1]}] inside Yesterday's Range [{self.lows[-2]}-{self.highs[-2]}]"
        if inside_day_2:
            return f"Yesterday's Range [{self.lows[-2]}-{self.highs[-2]}] inside Day Before Yesterday's Range [{self.lows[-3]}-{self.highs[-3]}]"
        if inside_day_3:
            return f"Day Before Yesterday's Range [{self.lows[-3]}-{self.highs[-3]}] inside Three Days Ago Range [{self.lows[-4]}-{self.highs[-4]}]"
        return "No inside day pattern detected"
