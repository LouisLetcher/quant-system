from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class RussellRebalancingStrategy(BaseStrategy):
    """
    Russell Rebalancing Strategy.

    Enters long in June, on or after the 24th, if we haven't entered yet that year.
    Exits at the first trading day of July.

    This strategy aims to capture the price movements around the annual Russell indexes
    rebalancing, which typically occurs at the end of June.
    """

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Use a variable to ensure we enter only once per year
        self.trade_opened = False
        self.current_year = None

        # We'll need to track the date
        # Convert index to datetime if it's not already
        if not isinstance(self.data.index, pd.DatetimeIndex):
            # If your data doesn't have a datetime index, you'll need to adjust this
            # This is just a placeholder assuming there's a 'Date' column
            self.dates = pd.to_datetime(self.data.index)
        else:
            self.dates = self.data.index

    def next(self):
        """Trading logic for each bar."""
        # Get current date components
        current_date = self.dates.iloc[-1]
        yr = current_date.year
        mon = current_date.month
        dom = current_date.day

        # If we're in a new year, reset the trade_opened flag
        if self.current_year is None or yr != self.current_year:
            self.current_year = yr
            self.trade_opened = False

        # Entry condition: In June, on or after the 24th, if we haven't entered yet.
        enter_condition = mon == 6 and dom >= 24 and not self.trade_opened

        # Exit condition: Detect the first bar of July
        # (i.e. current month is July and previous bar's month was not July)
        if len(self.dates) > 1:
            prev_date = self.dates.iloc[-2]
            exit_condition = mon == 7 and prev_date.month != 7
        else:
            exit_condition = False

        # Entry logic
        if enter_condition and not self.position:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📅 Date: {current_date.strftime('%Y-%m-%d')}")
            print("📊 Russell Rebalancing: Entering for June rebalancing")

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

            # Mark that we've opened a trade for this year
            self.trade_opened = True

        # Exit logic
        if exit_condition and self.position:
            print(f"🔴 EXIT TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📅 Date: {current_date.strftime('%Y-%m-%d')}")
            print("📊 Russell Rebalancing: Exiting at first trading day of July")
            self.position.close()
