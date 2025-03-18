from __future__ import annotations

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class LowerHighsLowerLowsStrategy(BaseStrategy):
    """
    Lower Highs & Lower Lows Strategy.

    Enters long when:
    1. Today's high is lower than yesterday's high
    2. Today's low is lower than yesterday's low

    Exits after holding for a specified number of days.

    This strategy aims to capture reversals after a short-term downtrend.
    """

    # Define parameters that can be optimized
    holding_days = 1  # Exit after N days

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Track entry bar for holding period calculation
        self.entry_bar = None

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) < 2:
            return

        # Define condition: Lower High and Lower Low compared to yesterday
        lower_high_low = (self.data.High[-1] < self.data.High[-2]) and (
            self.data.Low[-1] < self.data.Low[-2]
        )

        # Entry logic: If today's bar meets the condition and we're not in a position, enter long
        if lower_high_low and not self.position:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Today's High: {self.data.High[-1]}, Yesterday's High: {self.data.High[-2]}"
            )
            print(
                f"📊 Today's Low: {self.data.Low[-1]}, Yesterday's Low: {self.data.Low[-2]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)

            # Store the entry bar index
            self.entry_bar = len(self.data) - 1

            self.buy(size=size)

        # Exit logic: Check how long we've been in the trade
        if self.position and self.entry_bar is not None:
            # Calculate bars in trade
            bars_in_trade = len(self.data) - 1 - self.entry_bar

            # Once we've held for `holding_days` bars, close the position
            if bars_in_trade >= self.holding_days:
                print(
                    f"🔴 EXIT TRIGGERED after {bars_in_trade} bars at price: {self.data.Close[-1]}"
                )
                print(f"📅 Holding period of {self.holding_days} days reached")

                self.position.close()
                self.entry_bar = None
