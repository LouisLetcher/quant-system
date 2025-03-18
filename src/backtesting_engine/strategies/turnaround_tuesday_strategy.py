from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class TurnaroundTuesdayStrategy(BaseStrategy):
    """
    Turnaround Tuesday Strategy.

    A simple calendar-based strategy that:
    1. Enters long at Monday's close if Monday's close is lower than Friday's close
    2. Exits at Tuesday's close

    This strategy is designed for daily charts, such as SPY (S&P 500 ETF).
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
        # Note: pandas uses 0 for Monday, while TradingView uses 2 for Monday
        self.day_of_week = pd.Series([d.weekday() for d in self.dates])

    def next(self):
        """Trading logic for each bar."""
        # Only proceed if we have enough data
        if len(self.data) < 2:
            return

        # Check if today is Monday (weekday=0) or Tuesday (weekday=1)
        is_monday = self.day_of_week.iloc[-1] == 0
        is_tuesday = self.day_of_week.iloc[-1] == 1

        # Entry condition: On Monday's bar, at the close, if today's close < yesterday's close
        enter_long = is_monday and self.data.Close[-1] < self.data.Close[-2]

        # Entry logic: If the entry condition is met on Monday, enter a long position
        if enter_long and not self.position:
            print(f"🟢 LONG ENTRY TRIGGERED at Monday's close: {self.data.Close[-1]}")
            print(
                f"📊 Monday's close: {self.data.Close[-1]}, Friday's close: {self.data.Close[-2]}"
            )
            print(f"📅 Date: {self.dates.iloc[-1].strftime('%Y-%m-%d')}")

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size, comment="TurnaroundTuesdayLong")

        # Exit logic: On Tuesday's close, if we have an open position, exit
        if is_tuesday and self.position:
            print(f"🔴 EXIT TRIGGERED at Tuesday's close: {self.data.Close[-1]}")
            print(f"📅 Date: {self.dates.iloc[-1].strftime('%Y-%m-%d')}")
            self.position.close(comment="TurnaroundTuesdayExit")
