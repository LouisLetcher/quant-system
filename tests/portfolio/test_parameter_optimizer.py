import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from bs4 import BeautifulSoup
from src.portfolio.parameter_optimizer import (
    optimize_portfolio_parameters,
    extract_best_combinations_from_report,
    get_param_ranges,
    run_backtest_with_params
)
from src.backtesting_engine.strategies.mean_reversion import MeanReversion

class TestParameterOptimizer(unittest.TestCase):
    
    @patch('src.portfolio.parameter_optimizer.extract_best_combinations_from_report')
    @patch('src.portfolio.parameter_optimizer.get_portfolio_config')
    def test_optimize_portfolio_parameters(self, mock_get_config, mock_extract):
        # This is a complex function to test fully, so we'll just test the basic flow
        # Setup mocks
        mock_get_config.return_value = {
            'description': 'Test portfolio',
            'assets': [
                {'ticker': 'AAPL', 'commission': 0.001, 'initial_capital': 10000}
            ]
        }
        mock_extract.return_value = {
            'AAPL': {
                'strategy': 'mean_reversion',
                'interval': '1d',
                'return_pct': 15.5,
                'sharpe_ratio': 1.2,
                'max_drawdown_pct': 8.3,
                'win_rate': 62.5,
                'trades_count': 24,
                'profit_factor': 1.8,
                'score': 1.2
            }
        }
        
        # Create mock args
        class Args:
            name = 'test_portfolio'
            report_path = None
            metric = 'sharpe'
            max_tries = 10
            method = 'random'
            open_browser = False
        
        # We'll patch the rest of the function calls to avoid actual execution
        with patch('src.portfolio.parameter_optimizer.DataLoader'):
            with patch('src.portfolio.parameter_optimizer.ParameterTuner'):
                with patch('src.portfolio.parameter_optimizer.run_backtest_with_params'):
                    with patch('src.portfolio.parameter_optimizer.extract_detailed_metrics'):
                        with patch('src.portfolio.parameter_optimizer.generate_equity_chart'):
                            with patch('src.portfolio.parameter_optimizer.ReportGenerator'):
                                # Call function
                                result = optimize_portfolio_parameters(Args())
                                
                                # Basic assertions
                                mock_get_config.assert_called_with('test_portfolio')
                                mock_extract.assert_called_once()
    
    def test_extract_best_combinations_from_report(self):
        # Create a sample HTML report
        html_content = """
        <html>
            <h2>Assets Overview</h2>
            <table>
                <tr>
                    <th>Asset</th>
                    <th>Strategy</th>
                    <th>Interval</th>
                    <th>Return</th>
                    <th>Sharpe</th>
                    <th>Max DD</th>
                    <th>Win Rate</th>
                    <th>Trades</th>
                    <th>Profit Factor</th>
                </tr>
                <tr>
                    <td>AAPL</td>
                    <td>mean_reversion</td>
                    <td>1d</td>
                    <td>15.5%</td>
                    <td>1.2</td>
                    <td>8.3%</td>
                    <td>62.5%</td>
                    <td>24</td>
                    <td>1.8</td>
                </tr>
            </table>
        </html>
        """
        
        # Mock file open
        with patch('builtins.open', mock_open(read_data=html_content)):
            # Call function
            result = extract_best_combinations_from_report('dummy_path.html')
            
            # Assertions
            self.assertIn('AAPL', result)
            self.assertEqual(result['AAPL']['strategy'], 'mean_reversion')
            self.assertEqual(result['AAPL']['interval'], '1d')
            self.assertEqual(result['AAPL']['return_pct'], 15.5)
            self.assertEqual(result['AAPL']['sharpe_ratio'], 1.2)
            self.assertEqual(result['AAPL']['max_drawdown_pct'], 8.3)
            self.assertEqual(result['AAPL']['win_rate'], 62.5)
            self.assertEqual(result['AAPL']['trades_count'], 24)
            self.assertEqual(result['AAPL']['profit_factor'], 1.8)
            self.assertEqual(result['AAPL']['score'], 1.2)  # Default to sharpe
    
    def test_get_param_ranges(self):
        # Test with a strategy class that has param_ranges attribute
        class TestStrategy:
            param_ranges = {
                'sma_period': (10, 50),
                'std_dev': (1.0, 3.0)
            }
        
        # Call function
        result = get_param_ranges(TestStrategy)
        
        # Assertions
        self.assertEqual(result, TestStrategy.param_ranges)
        
        # Test with a strategy class that doesn't have param_ranges
        class TestStrategy2:
            sma_period = 20
            std_dev = 2.0
            threshold = 0.5
        
        # Call function
        result = get_param_ranges(TestStrategy2)
        
        # Assertions
        self.assertIn('sma_period', result)
        self.assertIn('std_dev', result)
        self.assertIn('threshold', result)
        self.assertEqual(result['sma_period'][0], 10)  # min = period/2
        self.assertEqual(result['sma_period'][1], 40)  # max = period*2
        self.assertTrue(0 < result['threshold'][0] < result['threshold'][1] < 1)  # threshold between 0-1
    
    @patch('src.portfolio.parameter_optimizer.BacktestEngine')
    def test_run_backtest_with_params(self, mock_backtest_engine):
        # Setup mock
        mock_instance = MagicMock()
        mock_backtest_engine.return_value = mock_instance
        mock_instance.run.return_value = {'test': 'result'}
        
        # Create test data
        test_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [98, 99, 100],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        # Call function
        result = run_backtest_with_params(
            strategy_class=MeanReversion,
            data=test_data,
            params={'sma_period': 20, 'std_dev': 2.0},
            initial_capital=10000,
            commission=0.001,
            ticker='AAPL'
        )
        
        # Assertions
        mock_backtest_engine.assert_called_once()
        self.assertEqual(result, {'test': 'result'})

if __name__ == '__main__':
    unittest.main()
