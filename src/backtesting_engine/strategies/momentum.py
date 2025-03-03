from src.backtesting_engine.strategies.base_strategy import BaseStrategy
import backtrader as bt

class MomentumStrategy(BaseStrategy):
    """Momentum Trading Strategy."""

    params = (("sma_period", 50),)

    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(period=self.params.sma_period)

    def next(self):
        if self.data.close[0] > self.sma[0]:  # Buy when price is above SMA
            self.buy()
        elif self.data.close[0] < self.sma[0]:  # Sell when price is below SMA
            self.sell()