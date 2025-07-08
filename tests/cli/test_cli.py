import argparse
import unittest
from unittest.mock import MagicMock, patch

from src.cli.main import main, setup_parser


class TestCLI(unittest.TestCase):

    def test_setup_parser(self):
        # Test that the parser is set up correctly
        parser = setup_parser()

        # Check that it's an ArgumentParser
        self.assertIsInstance(parser, argparse.ArgumentParser)

        # Check that it has the expected commands
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        commands = subparsers.choices.keys()

        # Check for essential commands
        self.assertIn("backtest", commands)
        self.assertIn("all-strategies", commands)
        self.assertIn("portfolio", commands)
        self.assertIn("intervals", commands)
        self.assertIn("optimize", commands)
        self.assertIn("portfolio-optimal", commands)
        self.assertIn("portfolio-optimize-params", commands)
        self.assertIn("list-portfolios", commands)
        self.assertIn("list-strategies", commands)

    @patch("argparse.ArgumentParser.parse_args")
    @patch("src.cli.main.backtest_strategy")
    def test_backtest_command(self, mock_backtest, mock_parse_args):
        # Setup mock args
        args = MagicMock()
        args.func = mock_backtest
        args.strategy = "mean_reversion"
        args.ticker = "AAPL"
        args.period = "1mo"
        args.interval = "1d"
        args.commission = 0.001
        args.initial_capital = 10000
        args.open_browser = False
        mock_parse_args.return_value = args

        # Call main function
        main()

        # Assertions
        mock_backtest.assert_called_once_with(args)

    @patch("argparse.ArgumentParser.parse_args")
    @patch("src.cli.main.compare_all_strategies")
    def test_all_strategies_command(self, mock_compare, mock_parse_args):
        # Setup mock args
        args = MagicMock()
        args.func = mock_compare
        args.ticker = "AAPL"
        args.period = "1mo"
        args.interval = "1d"
        args.metric = "sharpe"
        args.open_browser = False
        mock_parse_args.return_value = args

        # Call main function
        main()

        # Assertions
        mock_compare.assert_called_once_with(args)

    @patch("argparse.ArgumentParser.parse_args")
    @patch("src.cli.main.analyze_portfolio")
    def test_portfolio_command(self, mock_analyze, mock_parse_args):
        # Setup mock args
        args = MagicMock()
        args.func = mock_analyze
        args.name = "test_portfolio"
        args.period = "1mo"
        args.metric = "sharpe"
        args.open_browser = False
        mock_parse_args.return_value = args

        # Call main function
        main()

        # Assertions
        mock_analyze.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()
