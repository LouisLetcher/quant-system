from __future__ import annotations

import pandas as pd
import numpy as np

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class TurtleTradingStrategy(BaseStrategy):
    """
    Turtle Trading Strategy.

    Based on the famous trading system developed by Richard Dennis and Bill Eckhardt.

    Enters long when:
    1. Price breaks above the highest high of the last 'breakout_high_period' days

    Exits when:
    1. Price breaks below the lowest low of the last 'breakout_low_period' days
    2. Price falls below the ATR-based stop level (entry price - ATR_multiplier * entry_ATR)

    This strategy is designed for stocks, commodities, and indices.
    """

    # Define parameters that can be optimized
    breakout_high_period = 40
    breakout_low_period = 20
    atr_length = 14
    atr_multiplier = 2.0

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate highest high for breakout detection
        # Instead of using shift, we'll use a custom function that handles the offset
        self.highest_high = self.I(
            lambda x: self._calculate_highest_with_offset(x, self.breakout_high_period, 1),
            self.data.High
        )

        # Calculate lowest low for breakout detection
        self.lowest_low = self.I(
            lambda x: self._calculate_lowest_with_offset(x, self.breakout_low_period, 1),
            self.data.Low
        )

        # Calculate ATR
        self.atr = self.I(
            self._calculate_atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_length
        )

        # Track the ATR value at the time of entry
        self.entry_atr = None
        self.stop_level = None

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if (
            len(self.data)
            <= max(self.breakout_high_period, self.breakout_low_period, self.atr_length)
            + 1
        ):  # +1 for the offset
            return

        # Check for breakouts
        check_b = self.data.High[-1] > self.highest_high[-1]  # Upside breakout
        check_s = self.data.Low[-1] < self.lowest_low[-1]  # Downside breakout

        # Entry logic: Enter long on a breakout to the upside
        if not self.position and check_b:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 High: {self.data.High[-1]}, Highest High ({self.breakout_high_period} days): {self.highest_high[-1]}"
            )
            print("📈 Upside breakout detected")

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

            # Record the ATR at entry
            self.entry_atr = self.atr[-1]
            # Calculate the stop level
            self.stop_level = price - (self.atr_multiplier * self.entry_atr)

            print(f"📊 Entry ATR: {self.entry_atr}")
            print(f"🛑 Stop Level: {self.stop_level}")

        # Exit logic: Close position on downside breakout or stop level hit
        elif self.position:
            # Update stop level if needed
            if self.stop_level is not None:
                # Check if price fell below the stop level
                stop_hit = self.data.Close[-1] < self.stop_level

                if check_s or stop_hit:
                    reason = "Downside breakout" if check_s else "Stop level hit"
                    print(
                        f"🔴 EXIT TRIGGERED at price: {self.data.Close[-1]} - {reason}"
                    )

                    if check_s:
                        print(
                            f"📊 Low: {self.data.Low[-1]}, Lowest Low ({self.breakout_low_period} days): {self.lowest_low[-1]}"
                        )

                    if stop_hit:
                        print(
                            f"📊 Close: {self.data.Close[-1]}, Stop Level: {self.stop_level}"
                        )

                    self.position.close()
                    self.entry_atr = None
                    self.stop_level = None

    def _calculate_highest_with_offset(self, prices, period, offset=0):
        """
        Calculate highest value over a period with an offset
        
        This function calculates the highest value over a period, but excludes
        the most recent 'offset' number of bars from the calculation.

        Args:
            prices: Price series
            period: Lookback period
            offset: Number of recent bars to exclude (default: 0)

        Returns:
            Highest values over the lookback period
        """
        result = np.full_like(prices, np.nan)
        
        for i in range(period + offset - 1, len(prices)):
            # Calculate highest over the period, excluding the most recent 'offset' bars
            highest = np.max(prices[i-period-offset+1:i-offset+1])
            result[i] = highest
            
        return result

    def _calculate_lowest_with_offset(self, prices, period, offset=0):
        """
        Calculate lowest value over a period with an offset
        
        This function calculates the lowest value over a period, but excludes
        the most recent 'offset' number of bars from the calculation.

        Args:
            prices: Price series
            period: Lookback period
            offset: Number of recent bars to exclude (default: 0)

        Returns:
            Lowest values over the lookback period
        """
        result = np.full_like(prices, np.nan)
        
        for i in range(period + offset - 1, len(prices)):
            # Calculate lowest over the period, excluding the most recent 'offset' bars
            lowest = np.min(prices[i-period-offset+1:i-offset+1])
            result[i] = lowest
            
        return result

    def _calculate_atr(self, high, low, close, period=14):
        """
        Calculate Average True Range (ATR)

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR values
        """
        # Convert arrays to numpy arrays if they aren't already
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        
        # Calculate True Range
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        # Calculate ATR as the simple moving average of the True Range
        atr = np.zeros_like(tr)
        for i in range(period, len(tr)):
            atr[i] = np.mean(tr[i-period+1:i+1])
        
        return atr
