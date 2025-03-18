from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class KingsCountingStrategy(BaseStrategy):
    """
    King's Counting Strategy (Königszählung).

    Based on the DeMark 9-13 sequence, this strategy identifies momentum and trend phases
    in market movements and uses specific triggers to enter long and short positions.

    The strategy employs stop loss and take profit mechanisms to control risk and secure profits.

    Momentum Phase:
    - Up: Close > highest close of the last 9 bars (offset by 4)
    - Down: Close < lowest close of the last 9 bars (offset by 4)

    Trend Phase:
    - Up: Close > highest close of the last 13 bars (offset by 2)
    - Down: Close < lowest close of the last 13 bars (offset by 2)

    Entry Signals:
    - Long: Momentum phase down + Close > previous highest close + Trend phase up
    - Short: Momentum phase up + Close < previous lowest close + Trend phase down

    Exit Signals:
    - Stop Loss: Fixed points from entry
    - Take Profit: Fixed points from entry
    """

    # Define parameters that can be optimized
    momentum_length = 9
    trend_length = 13
    slippage = 20  # Stop loss points
    take_profit = 30  # Take profit points

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Variables to track momentum and trend phases
        self.momentum_phase_up = False
        self.momentum_phase_down = False
        self.trend_phase_up = False
        self.trend_phase_down = False

        # Calculate highest and lowest values for momentum and trend phases
        self.highest_momentum = self.I(
            lambda x: self._calculate_highest(x, self.momentum_length), self.data.Close
        )

        self.lowest_momentum = self.I(
            lambda x: self._calculate_lowest(x, self.momentum_length), self.data.Close
        )

        self.highest_trend = self.I(
            lambda x: self._calculate_highest(x, self.trend_length),
            self.data.Close.shift(2),  # Offset by 2 for trend calculation
        )

        self.lowest_trend = self.I(
            lambda x: self._calculate_lowest(x, self.trend_length),
            self.data.Close.shift(2),  # Offset by 2 for trend calculation
        )

        # Store stop loss and take profit levels
        self.long_stop = None
        self.long_profit = None
        self.short_stop = None
        self.short_profit = None

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(self.momentum_length, self.trend_length) + 4:
            return

        # Momentum Phase Calculation
        momentum_up = self.data.Close[-1] > self.highest_momentum[-5]  # Offset by 4
        momentum_down = self.data.Close[-1] < self.lowest_momentum[-5]  # Offset by 4

        # Trend Phase Calculation
        trend_up = self.data.Close[-1] > self.highest_trend[-1]
        trend_down = self.data.Close[-1] < self.lowest_trend[-1]

        # Setting flags for Momentum and Trend Phases
        if momentum_up:
            self.momentum_phase_up = True
            self.momentum_phase_down = False

        if momentum_down:
            self.momentum_phase_down = True
            self.momentum_phase_up = False

        if trend_up and self.momentum_phase_up:
            self.trend_phase_up = True
            self.trend_phase_down = False

        if trend_down and self.momentum_phase_down:
            self.trend_phase_down = True
            self.trend_phase_up = False

        # Entry signals: Trigger points for trades
        long_signal = (
            self.momentum_phase_down
            and self.data.Close[-1] > self.highest_momentum[-2]  # Previous highest
            and self.trend_phase_up
        )

        short_signal = (
            self.momentum_phase_up
            and self.data.Close[-1] < self.lowest_momentum[-2]  # Previous lowest
            and self.trend_phase_down
        )

        # Long entry logic
        if not self.position and long_signal:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print("📊 Momentum Phase: Down, Trend Phase: Up")
            print(
                f"📈 Close: {self.data.Close[-1]}, Previous Highest: {self.highest_momentum[-2]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)

            # Set stop loss and take profit levels
            self.long_stop = price - self.slippage
            self.long_profit = price + self.take_profit

            print(
                f"🛑 Stop Loss: {self.long_stop} ({self.slippage} points below entry)"
            )
            print(
                f"🎯 Take Profit: {self.long_profit} ({self.take_profit} points above entry)"
            )

            self.buy(size=size)

        # Short entry logic
        elif not self.position and short_signal:
            print(f"🔴 SHORT ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print("📊 Momentum Phase: Up, Trend Phase: Down")
            print(
                f"📉 Close: {self.data.Close[-1]}, Previous Lowest: {self.lowest_momentum[-2]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)

            # Set stop loss and take profit levels
            self.short_stop = price + self.slippage
            self.short_profit = price - self.take_profit

            print(
                f"🛑 Stop Loss: {self.short_stop} ({self.slippage} points above entry)"
            )
            print(
                f"🎯 Take Profit: {self.short_profit} ({self.take_profit} points below entry)"
            )

            self.sell(size=size)

        # Exit logic for long positions
        if self.position and self.position.is_long:
            # Check for stop loss
            if self.data.Close[-1] < self.long_stop:
                print(f"🛑 LONG STOP LOSS TRIGGERED at price: {self.data.Close[-1]}")
                print(f"📉 Close: {self.data.Close[-1]}, Stop Level: {self.long_stop}")
                self.position.close()
                self.long_stop = None
                self.long_profit = None

            # Check for take profit
            elif self.data.Close[-1] > self.long_profit:
                print(f"🎯 LONG TAKE PROFIT TRIGGERED at price: {self.data.Close[-1]}")
                print(
                    f"📈 Close: {self.data.Close[-1]}, Profit Level: {self.long_profit}"
                )
                self.position.close()
                self.long_stop = None
                self.long_profit = None

        # Exit logic for short positions
        if self.position and self.position.is_short:
            # Check for stop loss
            if self.data.Close[-1] > self.short_stop:
                print(f"🛑 SHORT STOP LOSS TRIGGERED at price: {self.data.Close[-1]}")
                print(f"📈 Close: {self.data.Close[-1]}, Stop Level: {self.short_stop}")
                self.position.close()
                self.short_stop = None
                self.short_profit = None

            # Check for take profit
            elif self.data.Close[-1] < self.short_profit:
                print(f"🎯 SHORT TAKE PROFIT TRIGGERED at price: {self.data.Close[-1]}")
                print(
                    f"📉 Close: {self.data.Close[-1]}, Profit Level: {self.short_profit}"
                )
                self.position.close()
                self.short_stop = None
                self.short_profit = None

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

    def _calculate_lowest(self, prices, period):
        """
        Calculate lowest value over a period

        Args:
            prices: Price series
            period: Lookback period

        Returns:
            Lowest values over the lookback period
        """
        return pd.Series(prices).rolling(window=period).min()
