import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.optimizer.parameter_tuner import ParameterTuner
from src.backtesting_engine.strategies.mean_reversion import MeanReversion

class TestParameterTuner(unittest.TestCase):
    
    def setUp(self):
        # Create sample data for testing
        self.test_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'Close': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=pd.date_range('2023-01-01', periods=10))
        
        # Create parameter tuner
        self.tuner = ParameterTuner(
            strategy_class=MeanReversion,
            data=self.test_data,
            initial_capital=10000,
            commission=0.001,
            ticker='TEST',
            metric='sharpe'
        )
    
    def test_initialization(self):
        self.assertEqual(self.tuner.initial_capital, 10000)
        self.assertEqual(self.tuner.commission, 0.001)
        self.assertEqual(self.tuner.ticker, 'TEST')
        self.assertEqual(self.tuner.metric, 'sharpe')
        self.assertIs(self.tuner.strategy_class, MeanReversion)
    
    @patch('src.optimizer.parameter_tuner.BacktestEngine')
    def test_evaluate_params(self, mock_backtest_engine):
        # Setup mock
        mock_instance = MagicMock()
        mock_backtest_engine.return_value = mock_instance
        mock_instance.run.return_value = {
            'metrics': {
                'sharpe_ratio': 1.5,
                'return_pct': 10.0,
                'max_drawdown_pct': 5.0,
                'win_rate': 60.0,
                'profit_factor': 2.0,
                'trades_count': 10
            }
        }
        
        # Test evaluate_params with sharpe metric
        params = {'sma_period': 20, 'std_dev': 2.0}
        result = self.tuner.evaluate_params(params)
        
        # Check result
        self.assertEqual(result, 1.5)  # Should return sharpe_ratio
        
        # Change metric and test again
        self.tuner.metric = 'profit_factor'
        result = self.tuner.evaluate_params(params)
        self.assertEqual(result, 2.0)  # Should return profit_factor
    
    @patch('src.optimizer.parameter_tuner.ParameterTuner.evaluate_params')
    def test_optimize_random(self, mock_evaluate):
        # Setup mock
        mock_evaluate.side_effect = [0.5, 1.0, 1.5, 1.2, 0.8]
        
        # Define parameter ranges
        param_ranges = {
            'sma_period': (10, 50),
            'std_dev': (1.0, 3.0)
        }
        
        # Run optimization
        best_params, best_value, results = self.tuner.optimize(
            param_ranges=param_ranges,
            max_tries=5,
            method='random'
        )
        
        # Check results
        self.assertEqual(best_value, 1.5)  # Should be the max value returned by mock
        self.assertEqual(len(results), 5)  # Should have 5 results

if __name__ == '__main__':
    unittest.main()
