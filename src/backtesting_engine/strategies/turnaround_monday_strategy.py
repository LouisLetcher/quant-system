from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class TurnaroundMondayStrategy(BaseStrategy):
    """
    Turnaround Monday Strategy.

    A simple calendar-based strategy that:
    1. Buys at Monday's open if the previous Friday was a down day
    2. Exits at Monday's close

    This strategy is based on the tendency for markets to reverse direction
    after a down day on Friday, particularly on Mondays.
    """

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # We'll need to track the day of the week
        # Convert index to datetime if it's not already
        if not isinstance(self.data.index, pd.DatetimeIndex):
            # If your data doesn't have a datetime index, you'll need to adjust this
            # This is just a placeholder assuming there's a 'Date' column
            self.dates = pd.to_datetime(self.data.index)
        else:
            self.dates = self.data.index

        # Create a series for day of week (0=Monday, 6=Sunday)
        self.day_of_week = pd.Series([d.weekday() for d in self.dates])

        # Track if the previous day was a down day (close < open)
        self.is_down_day = self.data.Close < self.data.Open

    def next(self):
        """Trading logic for each bar."""
        # Only proceed if we have enough data
        if len(self.data) < 2:
            return

        # Check if today is Monday (weekday=0)
        is_monday = self.day_of_week.iloc[-1] == 0

        # Find the last Friday
        # We need to look back to find the most recent Friday (weekday=4)
        was_friday_down = False

        # Look back up to 10 bars to find the last Friday
        for i in range(1, min(10, len(self.data))):
            if self.day_of_week.iloc[-i - 1] == 4:  # 4 = Friday
                was_friday_down = self.is_down_day.iloc[-i - 1]
                break

        # Entry logic: Buy on Monday's open if Friday was a down day
        if is_monday and was_friday_down and not self.position:
            print(f"🟢 BUY SIGNAL TRIGGERED at Monday's open: {self.data.Open[-1]}")
            print("📊 Previous Friday was a down day")

            # Use the position sizing method from BaseStrategy
            price = self.data.Open[-1]  # Use open price for Monday
            size = self.position_size(price)
            self.buy(size=size, comment="Buy Monday Open")

        # Exit logic: Close at Monday's close
        if is_monday and self.position:
            print(f"🔴 SELL SIGNAL TRIGGERED at Monday's close: {self.data.Close[-1]}")
            self.position.close(comment="Exit Monday Close")

    def _get_last_friday_index(self, current_index):
        """
        Find the index of the last Friday before the current bar

        Args:
            current_index: Current bar index

        Returns:
            Index of the last Friday, or None if not found
        """
        for i in range(1, min(10, current_index + 1)):
            if self.day_of_week.iloc[current_index - i] == 4:  # 4 = Friday
                return current_index - i
        return None
