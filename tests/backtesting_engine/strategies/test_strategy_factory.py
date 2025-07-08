import unittest

from src.backtesting_engine.strategies.mean_reversion import MeanReversion
from src.backtesting_engine.strategies.momentum import Momentum
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory


class TestStrategyFactory(unittest.TestCase):

    def test_get_strategy(self):
        # Test getting valid strategies
        mean_reversion = StrategyFactory.get_strategy("mean_reversion")
        momentum = StrategyFactory.get_strategy("momentum")

        self.assertEqual(mean_reversion, MeanReversion)
        self.assertEqual(momentum, Momentum)

        # Test case insensitivity
        mean_reversion_upper = StrategyFactory.get_strategy("MEAN_REVERSION")
        self.assertEqual(mean_reversion_upper, MeanReversion)

        # Test invalid strategy
        with self.assertRaises(ValueError):
            StrategyFactory.get_strategy("non_existent_strategy")

    def test_list_strategies(self):
        # Get list of strategies
        strategies = StrategyFactory.list_strategies()

        # Check that it's a list of strings
        self.assertIsInstance(strategies, list)
        self.assertTrue(all(isinstance(s, str) for s in strategies))

        # Check that common strategies are included
        self.assertIn("mean_reversion", strategies)
        self.assertIn("momentum", strategies)

    def test_get_strategy_info(self):
        # Get info for a strategy
        info = StrategyFactory.get_strategy_info("mean_reversion")

        # Check that it contains expected keys
        self.assertIsInstance(info, dict)
        self.assertIn("name", info)
        self.assertIn("description", info)

        # Check values
        self.assertEqual(info["name"], "mean_reversion")

        # Test invalid strategy
        with self.assertRaises(ValueError):
            StrategyFactory.get_strategy_info("non_existent_strategy")


if __name__ == "__main__":
    unittest.main()
