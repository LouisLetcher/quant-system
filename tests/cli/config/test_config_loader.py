import unittest
import os
import json
from unittest.mock import patch, mock_open
from src.cli.config.config_loader import (
    get_portfolio_config,
    get_default_parameters,
    load_config_file
)

class TestConfigLoader(unittest.TestCase):
    
    def setUp(self):
        # Sample config data
        self.sample_config = {
            "portfolios": {
                "tech_stocks": {
                    "description": "Technology sector stocks",
                    "assets": [
                        {
                            "ticker": "AAPL",
                            "commission": 0.001,
                            "initial_capital": 10000
                        },
                        {
                            "ticker": "MSFT",
                            "commission": 0.001,
                            "initial_capital": 10000
                        }
                    ]
                }
            }
        }
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_config_file(self, mock_json_load, mock_file_open):
        # Setup mock
        mock_json_load.return_value = self.sample_config
        
        # Call function
        result = load_config_file('config/assets_config.json')
        
        # Assertions
        mock_file_open.assert_called_with('config/assets_config.json', 'r')
        mock_json_load.assert_called_once()
        self.assertEqual(result, self.sample_config)
        
        # Test with file not found
        mock_file_open.side_effect = FileNotFoundError()
        result = load_config_file('config/assets_config.json')
        self.assertEqual(result, {})
    
    @patch('src.cli.config.config_loader.load_config_file')
    def test_get_portfolio_config(self, mock_load_config):
        # Setup mock
        mock_load_config.return_value = self.sample_config
        
        # Call function
        result = get_portfolio_config('tech_stocks')
        
        # Assertions
        mock_load_config.assert_called_once()
        self.assertEqual(result['description'], 'Technology sector stocks')
        self.assertEqual(len(result['assets']), 2)
        self.assertEqual(result['assets'][0]['ticker'], 'AAPL')
        
        # Test with non-existent portfolio
        result = get_portfolio_config('non_existent')
        self.assertIsNone(result)
    
    def test_get_default_parameters(self):
        # Call function
        defaults = get_default_parameters()
        
        # Assertions
        self.assertIsInstance(defaults, dict)
        self.assertIn('commission', defaults)
        self.assertIn('initial_capital', defaults)
        self.assertIn('period', defaults)
        self.assertIn('interval', defaults)

if __name__ == '__main__':
    unittest.main()
