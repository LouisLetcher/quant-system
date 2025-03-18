from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.backtesting_engine.strategies.base_strategy import BaseStrategy


class LinearRegressionStrategy(BaseStrategy):
    """
    Linear Regression Strategy.

    Enters long when:
    - Current price is below linear regression value AND above long-term MA

    Enters short when:
    - Current price is above linear regression value AND below long-term MA

    Exits long when:
    - Last 3 bars' closes are above the linear regression value

    Exits short when:
    - Current price is below the linear regression value
    """

    # Define parameters that can be optimized
    lin_reg_length = 20
    long_term_ma_length = 200

    def init(self):
        """Initialize strategy indicators."""
        # Call parent init to set up common properties
        super().init()

        # Calculate Linear Regression
        self.lin_reg = self.I(
            lambda x: self._calculate_linear_regression(x, self.lin_reg_length),
            self.data.Close,
        )

        # Calculate long-term moving average
        self.long_term_ma = self.I(
            lambda x: self._calculate_sma(x, self.long_term_ma_length), self.data.Close
        )

    def next(self):
        """Trading logic for each bar."""
        # Only check for signals after we have enough bars
        if len(self.data) <= max(self.lin_reg_length, self.long_term_ma_length):
            return

        # Entry conditions
        long_entry_condition = (self.data.Close[-1] < self.lin_reg[-1]) and (
            self.data.Close[-1] > self.long_term_ma[-1]
        )
        short_entry_condition = (self.data.Close[-1] > self.lin_reg[-1]) and (
            self.data.Close[-1] < self.long_term_ma[-1]
        )

        # Exit conditions
        long_exit_condition = (
            (self.data.Close[-1] > self.lin_reg[-1])
            and (self.data.Close[-2] > self.lin_reg[-2])
            and (self.data.Close[-3] > self.lin_reg[-3])
        )
        short_exit_condition = self.data.Close[-1] < self.lin_reg[-1]

        # Long entry logic
        if not self.position and long_entry_condition:
            print(f"🟢 LONG ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Close: {self.data.Close[-1]}, Linear Regression: {self.lin_reg[-1]}"
            )
            print(
                f"📈 Close: {self.data.Close[-1]}, Long-Term MA({self.long_term_ma_length}): {self.long_term_ma[-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.buy(size=size)

        # Short entry logic
        elif not self.position and short_entry_condition:
            print(f"🔴 SHORT ENTRY TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Close: {self.data.Close[-1]}, Linear Regression: {self.lin_reg[-1]}"
            )
            print(
                f"📉 Close: {self.data.Close[-1]}, Long-Term MA({self.long_term_ma_length}): {self.long_term_ma[-1]}"
            )

            # Use the position sizing method from BaseStrategy
            price = self.data.Close[-1]
            size = self.position_size(price)
            self.sell(size=size)

        # Long exit logic
        elif self.position and self.position.is_long and long_exit_condition:
            print(f"🟡 LONG EXIT TRIGGERED at price: {self.data.Close[-1]}")
            print("📊 Last 3 closes above Linear Regression")
            self.position.close()

        # Short exit logic
        elif self.position and self.position.is_short and short_exit_condition:
            print(f"🟡 SHORT EXIT TRIGGERED at price: {self.data.Close[-1]}")
            print(
                f"📊 Close below Linear Regression: {self.data.Close[-1]} < {self.lin_reg[-1]}"
            )
            self.position.close()

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

    def _calculate_linear_regression(self, prices, length):
        """
        Calculate Linear Regression line

        Args:
            prices: Price series
            length: Lookback period for linear regression

        Returns:
            Linear regression values
        """
        # Convert to pandas Series if not already
        prices = pd.Series(prices)
        result = np.full_like(prices, np.nan)

        # Calculate linear regression for each point with enough history
        for i in range(length - 1, len(prices)):
            # Get the slice of data for regression
            y = prices.iloc[i - length + 1 : i + 1].values
            x = np.arange(length).reshape(-1, 1)

            # Fit linear regression model
            model = LinearRegression()
            model.fit(x, y)

            # Predict the current value
            result[i] = model.predict([[length - 1]])[0]

        return result
