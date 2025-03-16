import unittest
import os
import pandas as pd
from unittest.mock import patch
from src.backtesting_engine.data_loader import DataLoader
from src.backtesting_engine.engine import BacktestEngine
from src.backtesting_engine.strategies.mean_reversion import MeanReversion
from src.optimizer.parameter_tuner import ParameterTuner
from src.reports.report_generator import ReportGenerator

class TestWorkflow(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        # Create sample data
        cls.test_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'Close': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=pd.date_range('2023-01-01', periods=10))
    
    @patch('src.backtesting_engine.data_loader.yf.download')
    def test_complete_workflow(self, mock_download):
        """Test the complete workflow from data loading to report generation."""
        # Setup mock for data loading
        mock_download.return_value = self.test_data
        
        # 1. Load data
        data = DataLoader.load_data('TEST', period='1mo', interval='1d')
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 10)
        
        # 2. Run backtest
        engine = BacktestEngine(
            strategy=MeanReversion,
            data=data,
            cash=10000,
            commission=0.001,
            ticker='TEST'
        )
        result = engine.run()
        
        # Check backtest results
        self.assertIn('equity_curve', result)
        self.assertIn('trades', result)
        self.assertIn('metrics', result)
        
        # 3. Optimize parameters
        tuner = ParameterTuner(
            strategy_class=MeanReversion,
            data=data,
            initial_capital=10000,
            commission=0.001,
            ticker='TEST',
            metric='sharpe'
        )
        
        param_ranges = {
            'sma_period': (10, 30),
            'std_dev': (1.0, 3.0)
        }
        
        best_params, best_value, optimization_results = tuner.optimize(
            param_ranges=param_ranges,
            max_tries=5,
            method='random'
        )
        
        # Check optimization results
        self.assertIsInstance(best_params, dict)
        self.assertIn('sma_period', best_params)
        self.assertIn('std_dev', best_params)
        self.assertIsInstance(best_value, (int, float))
        self.assertEqual(len(optimization_results), 5)
        
        # 4. Generate report
        generator = ReportGenerator()
        
        # Prepare report data
        report_data = {
            'strategy': 'MeanReversion',
            'ticker': 'TEST',
            'results': optimization_results,
            'metric': 'sharpe'
        }
        
        # Generate report
        output_path = 'reports_output/test_integration.html'
        report_path = generator.generate_optimizer_report(report_data, output_path)
        
        # Check report was generated
        self.assertEqual(report_path, output_path)
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == '__main__':
    unittest.main()
