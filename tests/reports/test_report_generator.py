import json
import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

from src.reports.report_generator import ReportGenerator


class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = ReportGenerator()
        self.test_data = {
            "strategy": "mean_reversion",
            "asset": "AAPL",
            "metrics": {
                "return_pct": 15.5,
                "sharpe_ratio": 1.2,
                "max_drawdown_pct": 8.3,
                "win_rate": 62.5,
                "profit_factor": 1.8,
                "trades_count": 24,
            },
            "trades": [
                {
                    "entry_date": "2023-01-05",
                    "exit_date": "2023-01-10",
                    "entry_price": 150.0,
                    "exit_price": 155.0,
                    "size": 10,
                    "pnl": 50.0,
                    "return_pct": 3.33,
                    "type": "long",
                    "duration": "5 days",
                }
            ],
            "equity_curve": [
                {"date": "2023-01-01", "value": 10000},
                {"date": "2023-01-10", "value": 10500},
                {"date": "2023-01-20", "value": 11000},
                {"date": "2023-01-30", "value": 11550},
            ],
        }

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.reports.report_generator.Environment")
    def test_generate_backtest_report(self, mock_env, mock_file, mock_makedirs):
        # Setup mock template
        mock_template = MagicMock()
        mock_env_instance = MagicMock()
        mock_env.return_value = mock_env_instance
        mock_env_instance.get_template.return_value = mock_template
        mock_template.render.return_value = "<html>Test Report</html>"

        # Call method
        output_path = self.generator.generate_backtest_report(self.test_data)

        # Assertions (continued)
        mock_env_instance.get_template.assert_called_with("backtest_report.html")
        mock_template.render.assert_called()
        mock_file.assert_called()
        self.assertTrue(output_path.endswith(".html"))

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_generate_error_report(self, mock_file, mock_makedirs):
        # Call the method with an error
        error = ValueError("Test error")
        self.generator._generate_error_report(self.test_data, error, "test_output.html")

        # Assertions
        mock_makedirs.assert_called()
        mock_file.assert_called_with("test_output.html", "w")
        mock_file().write.assert_called()

        # Check that error message is in the written content
        written_content = mock_file().write.call_args[0][0]
        self.assertIn("Error Generating Report", written_content)
        self.assertIn("Test error", written_content)

    def test_prepare_template_variables(self):
        # Test with NaN values
        test_data_with_nan = self.test_data.copy()
        test_data_with_nan["metrics"]["sharpe_ratio"] = float("nan")

        # Call the method
        result = self.generator._prepare_template_variables(
            test_data_with_nan, self.generator.TEMPLATES["single_strategy"]
        )

        # Check that NaN was replaced
        self.assertEqual(result["data"]["metrics"]["sharpe_ratio"], 0.0)

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.reports.report_generator.Environment")
    def test_generate_multi_strategy_report(self, mock_env, mock_file, mock_makedirs):
        # Setup mock template
        mock_template = MagicMock()
        mock_env_instance = MagicMock()
        mock_env.return_value = mock_env_instance
        mock_env_instance.get_template.return_value = mock_template
        mock_template.render.return_value = "<html>Test Multi-Strategy Report</html>"

        # Create test data for multi-strategy report
        multi_strategy_data = {
            "asset": "AAPL",
            "strategies": {
                "mean_reversion": {
                    "return_pct": 15.5,
                    "sharpe_ratio": 1.2,
                    "trades_count": 24,
                },
                "momentum": {
                    "return_pct": 12.3,
                    "sharpe_ratio": 1.1,
                    "trades_count": 18,
                },
            },
            "best_strategy": "mean_reversion",
            "best_score": 1.2,
            "metric": "sharpe",
        }

        # Call method
        output_path = self.generator.generate_multi_strategy_report(multi_strategy_data)

        # Assertions
        mock_env_instance.get_template.assert_called_with("multi_strategy_report.html")
        mock_template.render.assert_called()
        mock_file.assert_called()
        self.assertTrue(output_path.endswith(".html"))

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.reports.report_generator.Environment")
    def test_generate_parameter_optimization_report(
        self, mock_env, mock_file, mock_makedirs
    ):
        # Setup mock template
        mock_template = MagicMock()
        mock_env_instance = MagicMock()
        mock_env.return_value = mock_env_instance
        mock_env_instance.get_template.return_value = mock_template
        mock_template.render.return_value = (
            "<html>Test Parameter Optimization Report</html>"
        )

        # Create test data for parameter optimization report
        optimization_data = {
            "portfolio": "test_portfolio",
            "description": "Test portfolio description",
            "metric": "sharpe",
            "best_combinations": {
                "AAPL": {
                    "strategy": "mean_reversion",
                    "interval": "1d",
                    "original_score": 1.2,
                    "optimized_score": 1.5,
                    "improvement": 0.3,
                    "improvement_pct": 25.0,
                    "best_params": {"sma_period": 20, "std_dev": 2.0},
                    "return_pct": 15.5,
                    "sharpe_ratio": 1.5,
                    "max_drawdown_pct": 7.5,
                    "win_rate": 65.0,
                    "trades_count": 22,
                    "profit_factor": 1.9,
                    "optimization_results": [
                        {"params": {"sma_period": 15, "std_dev": 1.5}, "score": 1.3},
                        {"params": {"sma_period": 20, "std_dev": 2.0}, "score": 1.5},
                        {"params": {"sma_period": 25, "std_dev": 2.5}, "score": 1.4},
                    ],
                }
            },
        }

        # Call method
        output_path = self.generator.generate_parameter_optimization_report(
            optimization_data
        )

        # Assertions
        mock_env_instance.get_template.assert_called_with(
            "parameter_optimization_report.html"
        )
        mock_template.render.assert_called()
        mock_file.assert_called()
        self.assertTrue(output_path.endswith(".html"))


if __name__ == "__main__":
    unittest.main()
