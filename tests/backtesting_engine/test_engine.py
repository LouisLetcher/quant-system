import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.backtesting_engine.engine import BacktestEngine
from src.backtesting_engine.strategies.mean_reversion import MeanReversion

class TestBacktestEngine(unittest.TestCase):
    
    def setUp(self):
        # Create sample data for testing
        self.test_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'Close': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=pd.date_range('2023-01-01', periods=10))
        
        # Create strategy instance
        self.strategy = MeanReversion
        
        # Create backtest engine
        self.engine = BacktestEngine(
            strategy=self.strategy,
            data=self.test_data,
            cash=10000,
            commission=0.001,
            ticker='TEST'
        )
    
    def test_initialization(self):
        self.assertEqual(self.engine.cash, 10000)
        self.assertEqual(self.engine.commission, 0.001)
        self.assertEqual(self.engine.ticker, 'TEST')
        self.assertIs(self.engine.strategy, self.strategy)
        self.assertTrue(self.test_data.equals(self.engine.data))
    
    def test_run_backtest(self):
        # Run the backtest
        result = self.engine.run()
        
        # Check that result contains expected keys
        self.assertIn('equity_curve', result)
        self.assertIn('trades', result)
        self.assertIn('metrics', result)
    
    def test_calculate_metrics(self):
        # Run backtest first
        self.engine.run()
        
        # Calculate metrics
        metrics = self.engine._calculate_metrics()
        
        # Check that metrics contains expected keys
        self.assertIn('return_pct', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown_pct', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('profit_factor', metrics)
        self.assertIn('trades_count', metrics)

if __name__ == '__main__':
    unittest.main()
