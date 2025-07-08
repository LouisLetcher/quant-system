import unittest

import numpy as np
import pandas as pd

from src.backtesting_engine.strategies.mean_reversion import MeanReversion
from src.backtesting_engine.strategies.momentum import Momentum
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory


class TestStrategies(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.test_data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "High": [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
                "Low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "Close": [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
                "Volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            },
            index=pd.date_range("2023-01-01", periods=10),
        )

    def test_strategy_factory(self):
        # Test getting strategy classes
        mean_reversion = StrategyFactory.get_strategy("mean_reversion")
        momentum = StrategyFactory.get_strategy("momentum")

        self.assertEqual(mean_reversion, MeanReversion)
        self.assertEqual(momentum, Momentum)

        # Test getting non-existent strategy
        with self.assertRaises(ValueError):
            StrategyFactory.get_strategy("non_existent_strategy")

    def test_mean_reversion_indicators(self):
        # Create strategy instance
        strategy = MeanReversion

        # Add indicators to data
        data = strategy.add_indicators(self.test_data.copy())

        # Check that indicators were added
        self.assertIn("sma", data.columns)
        self.assertIn("upper_band", data.columns)
        self.assertIn("lower_band", data.columns)

    def test_momentum_indicators(self):
        # Create strategy instance
        strategy = Momentum

        # Add indicators to data
        data = strategy.add_indicators(self.test_data.copy())

        # Check that indicators were added
        self.assertIn("ema_fast", data.columns)
        self.assertIn("ema_slow", data.columns)


if __name__ == "__main__":
    unittest.main()
