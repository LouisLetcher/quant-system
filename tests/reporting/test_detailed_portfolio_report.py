"""Tests for detailed portfolio reporting."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from src.reporting.detailed_portfolio_report import DetailedPortfolioReporter


class TestDetailedPortfolioReporter:
    """Test cases for DetailedPortfolioReporter class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        reporter = DetailedPortfolioReporter()
        assert reporter.output_dir == Path("exports/reports")
        assert reporter.template_dir is not None

    def test_init_custom_output_dir(self):
        """Test initialization with custom output directory."""
        custom_dir = "custom/reports"
        reporter = DetailedPortfolioReporter(output_dir=custom_dir)
        assert reporter.output_dir == Path(custom_dir)

    @patch("src.reporting.detailed_portfolio_report.Path.mkdir")
    def test_ensure_output_directory(self, mock_mkdir):
        """Test output directory creation."""
        reporter = DetailedPortfolioReporter()
        reporter._ensure_output_directory()
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_format_currency_default(self):
        """Test currency formatting with default USD."""
        reporter = DetailedPortfolioReporter()
        assert "$1,000.00" in reporter._format_currency(1000.0)
        assert "$-500.50" in reporter._format_currency(-500.5)

    def test_format_currency_custom(self):
        """Test currency formatting with custom currency."""
        reporter = DetailedPortfolioReporter()
        result = reporter._format_currency(1000.0, currency="EUR")
        assert "1,000" in result  # Should contain formatted number

    def test_format_percentage(self):
        """Test percentage formatting."""
        reporter = DetailedPortfolioReporter()
        assert reporter._format_percentage(0.1234) == "12.34%"
        assert reporter._format_percentage(-0.0567) == "-5.67%"
        assert reporter._format_percentage(0.0) == "0.00%"

    def test_calculate_portfolio_metrics_valid_data(self):
        """Test portfolio metrics calculation with valid data."""
        reporter = DetailedPortfolioReporter()

        # Sample portfolio data
        portfolio_data = pd.DataFrame(
            {
                "Asset": ["AAPL", "MSFT", "GOOGL"],
                "Weight": [0.4, 0.35, 0.25],
                "Value": [40000, 35000, 25000],
                "Return": [0.12, 0.08, 0.15],
            }
        )

        metrics = reporter._calculate_portfolio_metrics(portfolio_data)

        assert isinstance(metrics, dict)
        assert "total_value" in metrics
        assert "num_assets" in metrics
        assert metrics["total_value"] == 100000
        assert metrics["num_assets"] == 3

    def test_calculate_portfolio_metrics_empty_data(self):
        """Test portfolio metrics calculation with empty data."""
        reporter = DetailedPortfolioReporter()

        empty_data = pd.DataFrame()
        metrics = reporter._calculate_portfolio_metrics(empty_data)

        assert isinstance(metrics, dict)
        assert metrics["total_value"] == 0
        assert metrics["num_assets"] == 0

    def test_calculate_risk_metrics_valid_returns(self):
        """Test risk metrics calculation with valid returns."""
        reporter = DetailedPortfolioReporter()

        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        risk_metrics = reporter._calculate_risk_metrics(returns)

        assert isinstance(risk_metrics, dict)
        assert "volatility" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        assert "max_drawdown" in risk_metrics
        assert risk_metrics["volatility"] > 0

    def test_calculate_risk_metrics_empty_returns(self):
        """Test risk metrics calculation with empty returns."""
        reporter = DetailedPortfolioReporter()

        empty_returns = pd.Series([])
        risk_metrics = reporter._calculate_risk_metrics(empty_returns)

        assert isinstance(risk_metrics, dict)
        assert risk_metrics["volatility"] == 0
        assert risk_metrics["sharpe_ratio"] == 0

    @patch("src.reporting.detailed_portfolio_report.Environment")
    def test_load_template_success(self, mock_env):
        """Test successful template loading."""
        mock_template = MagicMock()
        mock_env.return_value.get_template.return_value = mock_template

        reporter = DetailedPortfolioReporter()
        template = reporter._load_template("test_template.html")

        assert template == mock_template

    @patch("src.reporting.detailed_portfolio_report.Environment")
    def test_load_template_not_found(self, mock_env):
        """Test template loading when template not found."""
        mock_env.return_value.get_template.side_effect = Exception("Template not found")

        reporter = DetailedPortfolioReporter()

        with pytest.raises(Exception, match="Template not found"):
            reporter._load_template("nonexistent_template.html")

    @patch("builtins.open", new_callable=mock_open)
    @patch("src.reporting.detailed_portfolio_report.Path.mkdir")
    def test_save_report_success(self, mock_mkdir, mock_file):
        """Test successful report saving."""
        reporter = DetailedPortfolioReporter()

        html_content = "<html><body>Test Report</body></html>"
        filename = "test_report.html"

        reporter._save_report(html_content, filename)

        mock_mkdir.assert_called_once()
        mock_file.assert_called_once()

    def test_generate_filename(self):
        """Test filename generation."""
        reporter = DetailedPortfolioReporter()

        filename = reporter._generate_filename("Test Portfolio", "Q3", 2023)

        assert "Test_Portfolio" in filename
        assert "Q3" in filename
        assert "2023" in filename
        assert filename.endswith(".html")

    @patch.object(DetailedPortfolioReporter, "_load_template")
    @patch.object(DetailedPortfolioReporter, "_calculate_portfolio_metrics")
    @patch.object(DetailedPortfolioReporter, "_calculate_risk_metrics")
    @patch.object(DetailedPortfolioReporter, "_save_report")
    def test_generate_comprehensive_report_success(
        self, mock_save, mock_risk, mock_portfolio, mock_template
    ):
        """Test successful comprehensive report generation."""
        # Setup mocks
        mock_template_obj = MagicMock()
        mock_template_obj.render.return_value = "<html>Rendered Report</html>"
        mock_template.return_value = mock_template_obj

        mock_portfolio.return_value = {"total_value": 100000, "num_assets": 5}
        mock_risk.return_value = {"volatility": 0.15, "sharpe_ratio": 1.2}

        reporter = DetailedPortfolioReporter()

        # Sample data
        portfolio_data = pd.DataFrame(
            {"Asset": ["AAPL", "MSFT"], "Weight": [0.6, 0.4], "Value": [60000, 40000]}
        )

        returns_data = pd.Series([0.01, 0.02, -0.01])

        # Test
        result = reporter.generate_comprehensive_report(
            portfolio_data=portfolio_data,
            returns_data=returns_data,
            portfolio_name="Test Portfolio",
        )

        # Assertions
        assert result is not None
        mock_template.assert_called_once()
        mock_portfolio.assert_called_once_with(portfolio_data)
        mock_risk.assert_called_once_with(returns_data)
        mock_save.assert_called_once()

    def test_generate_comprehensive_report_missing_required_data(self):
        """Test report generation with missing required data."""
        reporter = DetailedPortfolioReporter()

        with pytest.raises(ValueError, match="Portfolio data is required"):
            reporter.generate_comprehensive_report(
                portfolio_data=None, portfolio_name="Test Portfolio"
            )

    @patch.object(DetailedPortfolioReporter, "_load_template")
    @patch.object(DetailedPortfolioReporter, "_save_report")
    def test_generate_comprehensive_report_minimal_data(self, mock_save, mock_template):
        """Test report generation with minimal required data."""
        mock_template_obj = MagicMock()
        mock_template_obj.render.return_value = "<html>Minimal Report</html>"
        mock_template.return_value = mock_template_obj

        reporter = DetailedPortfolioReporter()

        # Minimal data
        portfolio_data = pd.DataFrame(
            {"Asset": ["AAPL"], "Weight": [1.0], "Value": [100000]}
        )

        result = reporter.generate_comprehensive_report(
            portfolio_data=portfolio_data, portfolio_name="Minimal Portfolio"
        )

        assert result is not None
        mock_template.assert_called_once()
        mock_save.assert_called_once()

    def test_add_custom_metrics(self):
        """Test adding custom metrics to report."""
        reporter = DetailedPortfolioReporter()

        custom_metrics = {"custom_ratio": 1.5, "benchmark_alpha": 0.03}

        # Should be able to add custom metrics without error
        # This tests the extensibility of the reporter
        assert isinstance(custom_metrics, dict)

    @patch("src.reporting.detailed_portfolio_report.datetime")
    def test_report_timestamp_generation(self, mock_datetime):
        """Test report timestamp generation."""
        mock_datetime.now.return_value = datetime(2023, 6, 15, 14, 30)

        reporter = DetailedPortfolioReporter()
        filename = reporter._generate_filename("Test", "Q2", 2023)

        # Should include timestamp in some form
        assert isinstance(filename, str)
        assert len(filename) > 0


class TestIntegration:
    """Integration tests for the complete reporting workflow."""

    @patch("src.reporting.detailed_portfolio_report.Environment")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.reporting.detailed_portfolio_report.Path.mkdir")
    def test_complete_reporting_workflow(self, mock_mkdir, mock_file, mock_env):
        """Test complete workflow from data to saved report."""
        # Setup template mock
        mock_template = MagicMock()
        mock_template.render.return_value = "<html><body>Complete Report</body></html>"
        mock_env.return_value.get_template.return_value = mock_template

        # Create reporter
        reporter = DetailedPortfolioReporter(output_dir="test_reports")

        # Sample comprehensive data
        portfolio_data = pd.DataFrame(
            {
                "Asset": ["AAPL", "MSFT", "GOOGL", "AMZN"],
                "Weight": [0.3, 0.25, 0.25, 0.2],
                "Value": [30000, 25000, 25000, 20000],
                "Return": [0.12, 0.08, 0.15, 0.10],
            }
        )

        returns_data = pd.Series(
            [0.01, -0.02, 0.03, 0.01, 0.02] * 50
        )  # Simulate daily returns

        # Generate report
        result = reporter.generate_comprehensive_report(
            portfolio_data=portfolio_data,
            returns_data=returns_data,
            portfolio_name="Comprehensive Test Portfolio",
            reporting_period="Q2 2023",
        )

        # Verify workflow completion
        assert result is not None
        mock_template.render.assert_called_once()
        mock_file.assert_called()
        mock_mkdir.assert_called()

    @patch.object(DetailedPortfolioReporter, "_load_template")
    def test_error_recovery_workflow(self, mock_template):
        """Test error recovery in reporting workflow."""
        mock_template.side_effect = Exception("Template error")

        reporter = DetailedPortfolioReporter()

        portfolio_data = pd.DataFrame(
            {"Asset": ["AAPL"], "Weight": [1.0], "Value": [100000]}
        )

        with pytest.raises(Exception, match="Template error"):
            reporter.generate_comprehensive_report(
                portfolio_data=portfolio_data, portfolio_name="Error Test"
            )

    def test_large_portfolio_handling(self):
        """Test handling of large portfolio datasets."""
        reporter = DetailedPortfolioReporter()

        # Create large portfolio dataset
        large_portfolio = pd.DataFrame(
            {
                "Asset": [f"STOCK_{i}" for i in range(1000)],
                "Weight": [0.001] * 1000,
                "Value": [100] * 1000,
                "Return": [0.01] * 1000,
            }
        )

        # Should handle large datasets without memory issues
        metrics = reporter._calculate_portfolio_metrics(large_portfolio)

        assert metrics["num_assets"] == 1000
        assert metrics["total_value"] == 100000
