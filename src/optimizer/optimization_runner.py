from backtesting import Backtest
from src.backtesting_engine.data_loader import DataLoader

class OptimizationRunner:
    def __init__(self, strategy_name, ticker, start_date, end_date):
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        # Load the data and strategy class
        self.data = DataLoader.load_data(ticker, start_date, end_date)
        self.strategy_class = StrategyFactory.get_strategy(strategy_name)
        
        if self.strategy_class is None:
            raise ValueError(f"Strategy '{strategy_name}' not found")
    
    def run(self):
        """Run optimization and return best parameters."""
        print(f"🔍 Optimizing {self.strategy_name} on {self.ticker}...")
        
        # Define parameters to optimize - example for mean reversion
        params = {
            'sma_period': range(10, 50, 5)
        }
        
        # Create backtest
        bt = Backtest(
            data=self.data,
            strategy=self.strategy_class,
            cash=10000,
            commission=0.001
        )
        
        # Run optimization
        stats = bt.optimize(
            **params,
            maximize='Sharpe Ratio',
            method='grid',  # or 'skopt' for Bayesian optimization
            max_tries=100
        )
        
        print(f"✅ Optimization complete. Best parameters: {stats._strategy.params}")
        
        return [{
            "best_params": stats._strategy.params,
            "best_score": stats['Sharpe Ratio']
        }]
