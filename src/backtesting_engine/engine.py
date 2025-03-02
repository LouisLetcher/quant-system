import backtrader as bt

class BacktestEngine:
    def __init__(self, strategy, data, cash=10000, commission=0.001):
        self.strategy = strategy
        self.data = data
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.set_cash(cash)
        self.cerebro.broker.setcommission(commission=commission)
    
    def run(self):
        """Runs backtest on the provided strategy and data"""
        data_feed = bt.feeds.PandasData(dataname=self.data)
        self.cerebro.adddata(data_feed)
        self.cerebro.addstrategy(self.strategy)
        results = self.cerebro.run()
        return results[0]