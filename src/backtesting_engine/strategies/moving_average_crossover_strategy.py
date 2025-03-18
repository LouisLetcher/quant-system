from __future__ import annotations

import pandas as pd

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.

    Enters long when:
    1. Short-term EMA crosses above long-term EMA

    Enters short when:
    1. Short-term EMA crosses below long-term EMA

    Uses take profit and stop loss for risk management.
    """

    # Define parameters that can be optimized
    short_ma_length = 20
    long_ma_length = 50
    take_profit_pips = 50
    stop_loss_pips = 20
    pip_value = (
        0.0001  # For most forex pairs, 1 pip = 0.0001. For JPY pairs, 1 pip = 0.01
    )

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate Moving Averages
        self.short_ma = self.I(
            lambda x: self._calculate_ema(x, self.short_ma_length), self.data.Close
        )

        self.long_ma = self.I(
            lambda x: self._calculate_ema(x, self.long_ma_length), self.data.Close
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(self.short_ma_length, self.long_ma_length):
            return

        # Check for crossover (short MA crosses above long MA)
        long_condition = (
            self.short_ma[-2] <= self.long_ma[-2]
            and self.short_ma[-1] > self.long_ma[-1]
        )

        # Check for crossunder (short MA crosses below long MA)
        short_condition = (
            self.short_ma[-2] >= self.long_ma[-2]
            and self.short_ma[-1] < self.long_ma[-1]
        )

        # Calculate take profit and stop loss in price terms
        tp_long = self.take_profit_pips * self.pip_value
        sl_long = self.stop_loss_pips * self.pip_value
        tp_short = self.take_profit_pips * self.pip_value
        sl_short = self.stop_loss_pips * self.pip_value

        # Entry logic for long positions
        if not self.position and long_condition:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 Short MA: {self.short_ma[-1]}, Long MA: {self.long_ma[-1]}")

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)

            # Calculate take profit and stop loss levels
            take_profit_price = price + tp_long
            stop_loss_price = price - sl_long

            print(f"🎯 Take Profit: {take_profit_price} ({self.take_profit_pips} pips)")
            print(f"🛑 Stop Loss: {stop_loss_price} ({self.stop_loss_pips} pips)")

            # Enter position with take profit and stop loss
            self.buy(size=size, tp=take_profit_price, sl=stop_loss_price)

        # Entry logic for short positions
        elif not self.position and short_condition:
            print(f"🔴 SHORT ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(f"📊 Short MA: {self.short_ma[-1]}, Long MA: {self.long_ma[-1]}")

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)

            # Calculate take profit and stop loss levels
            take_profit_price = price - tp_short
            stop_loss_price = price + sl_short

            print(f"🎯 Take Profit: {take_profit_price} ({self.take_profit_pips} pips)")
            print(f"🛑 Stop Loss: {stop_loss_price} ({self.stop_loss_pips} pips)")

            # Enter position with take profit and stop loss
            self.sell(size=size, tp=take_profit_price, sl=stop_loss_price)

    def _calculate_ema(self, prices, period):
        """
        Calculate Exponential Moving Average (EMA)

        Args:
            prices: Price series
            period: EMA period

        Returns:
            EMA values
        """
        return pd.Series(prices).ewm(span=period, adjust=False).mean()
