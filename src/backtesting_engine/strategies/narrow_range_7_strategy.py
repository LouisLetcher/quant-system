from __future__ import annotations

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class NarrowRange7Strategy(BaseStrategy):
    """
    Narrow Range 7 (NR7) Strategy.

    Enters long when:
    1. Today's range (high - low) is narrower than the lowest range of the previous 6 days

    Exits when:
    1. Today's close is higher than yesterday's high

    The NR7 pattern often precedes a breakout as volatility contracts before expanding.
    """

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= 7:  # Need at least 7 bars (today + 6 previous days)
            return

        # Calculate today's range
        today_range = self.data.High[-1] - self.data.Low[-1]

        # Find the lowest range among the last 6 trading days (excluding today)
        ranges_past_6 = [
            self.data.High[-i - 1] - self.data.Low[-i - 1] for i in range(1, 7)
        ]
        lowest_range_past_6 = min(ranges_past_6)

        # Check if today's range is the narrowest compared to the last 6 days
        nr7 = today_range < lowest_range_past_6

        # Entry condition: If today's range is the narrowest, enter long at the close
        if nr7 and not self.position:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Today's Range: {today_range}, Lowest Range Past 6 Days: {lowest_range_past_6}"
            )
            print(
                "📏 NR7 Pattern Detected: Today's range is narrower than the previous 6 days"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Exit condition: Exit at the close when today's close > yesterday's high
        if self.position and self.data.Close[-1] > self.data.High[-2]:
            print(f"🔴 EXIT TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📈 Today's Close: {self.data.Close[-1]}, Yesterday's High: {self.data.High[-2]}"
            )
            self.position.close()
