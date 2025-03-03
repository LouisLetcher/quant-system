from backtesting import Backtest

class BacktestEngine:
    def __init__(self, strategy_class, data, cash=10000, commission=0.001, ticker=None):
        """Initialize the backtesting engine with a strategy and data.
        
        Args:
            strategy_class: The strategy class to use for backtesting
            data: DataFrame containing OHLCV data
            cash: Initial cash amount
            commission: Commission rate
            ticker: Stock ticker symbol to identify the asset
        """
        if data.empty:
            raise ValueError("❌ Data must be a non-empty Pandas DataFrame")
        
        print(f"✅ Data successfully loaded: {len(data)} rows.")
        
        # Ensure DataFrame has required columns 
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col.lower() not in [c.lower() for c in data.columns]:
                raise ValueError(f"❌ Data must include {col} column")
        
        # Ensure column names are properly capitalized for backtesting.py
        data.columns = [c.capitalize() if c.lower() in [rc.lower() for rc in required_columns] 
                        else c for c in data.columns]
        
        # Store data and prepare backtest
        self.data = data
        self.strategy_class = strategy_class
        self.cash = cash
        self.commission = commission
        
        # Set the ticker name on the data
        if ticker:
            self.data.name = ticker
        
        self.backtest = Backtest(
            data=self.data,
            strategy=self.strategy_class,
            cash=self.cash,
            commission=self.commission,
            trade_on_close=True
        )

    def run(self):
        """Runs the backtest and returns results."""
        print("🚀 Running Backtesting.py Engine...")
        results = self.backtest.run()
        
        if results is None:
            raise RuntimeError("❌ Backtesting.py did not return any results.")
            
        print(f"📊 Backtest complete with {len(self.data)} data points")
        return results
