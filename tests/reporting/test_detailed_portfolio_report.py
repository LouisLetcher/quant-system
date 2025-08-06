"""Simple tests for detailed portfolio reporting."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.reporting.detailed_portfolio_report import DetailedPortfolioReporter


class TestDetailedPortfolioReporter:
    """Test cases for DetailedPortfolioReporter class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        reporter = DetailedPortfolioReporter()
        assert reporter.report_data == {}
        assert reporter.report_organizer is not None
        assert reporter.rng is not None

    @patch.object(DetailedPortfolioReporter, "_analyze_asset_with_timeframes")
    @patch.object(DetailedPortfolioReporter, "_generate_html_report")
    def test_generate_comprehensive_report_basic(
        self, mock_generate_html, mock_analyze
    ):
        """Test basic comprehensive report generation."""
        # Setup mocks
        mock_analyze.return_value = (
            {"strategy": "BuyAndHold", "timeframe": "1d"},
            {"metrics": {"sharpe": 1.2}},
        )
        mock_generate_html.return_value = "<html>Test Report</html>"

        reporter = DetailedPortfolioReporter()

        portfolio_config = {
            "symbols": ["AAPL", "MSFT"],
            "portfolio_name": "Test Portfolio",
        }

        result = reporter.generate_comprehensive_report(
            portfolio_config=portfolio_config,
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategies=["BuyAndHold"],
        )

        assert isinstance(result, str)
        mock_analyze.assert_called()
        mock_generate_html.assert_called_once()

    def test_generate_comprehensive_report_with_timeframes(self):
        """Test report generation with custom timeframes."""
        reporter = DetailedPortfolioReporter()

        portfolio_config = {"symbols": ["AAPL"], "portfolio_name": "Test Portfolio"}

        with patch.object(reporter, "_analyze_asset_with_timeframes") as mock_analyze:
            with patch.object(reporter, "_generate_html_report") as mock_generate:
                mock_analyze.return_value = (
                    {"strategy": "BuyAndHold", "timeframe": "1h"},
                    {"metrics": {}},
                )
                mock_generate.return_value = "<html>Report</html>"

                result = reporter.generate_comprehensive_report(
                    portfolio_config=portfolio_config,
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                    strategies=["BuyAndHold"],
                    timeframes=["1h", "4h", "1d"],
                )

                assert isinstance(result, str)
                mock_analyze.assert_called()

    def test_generate_comprehensive_report_missing_symbols(self):
        """Test report generation with missing symbols in config."""
        reporter = DetailedPortfolioReporter()

        portfolio_config = {
            "portfolio_name": "Test Portfolio"
            # Missing symbols key
        }

        with pytest.raises(KeyError):
            reporter.generate_comprehensive_report(
                portfolio_config=portfolio_config,
                start_date="2023-01-01",
                end_date="2023-12-31",
                strategies=["BuyAndHold"],
            )

    @patch("src.reporting.detailed_portfolio_report.np.random.default_rng")
    def test_random_number_generator_initialization(self, mock_rng):
        """Test that random number generator is properly initialized."""
        mock_rng.return_value = MagicMock()

        reporter = DetailedPortfolioReporter()

        assert reporter.rng is not None
        mock_rng.assert_called_once()

    def test_report_data_initialization(self):
        """Test that report data is properly initialized."""
        reporter = DetailedPortfolioReporter()

        assert isinstance(reporter.report_data, dict)
        assert len(reporter.report_data) == 0

    def test_analyze_asset_method_exists(self):
        """Test that the analyze asset method exists."""
        reporter = DetailedPortfolioReporter()

        # Check that the method exists (even if we don't call it)
        assert hasattr(reporter, "_analyze_asset_with_timeframes")

    def test_generate_html_method_exists(self):
        """Test that the HTML generation method exists."""
        reporter = DetailedPortfolioReporter()

        # Check that the method exists (even if we don't call it)
        assert hasattr(reporter, "_generate_html_report")


class TestReportOrganizer:
    """Test cases for report organizer integration."""

    @patch("src.reporting.detailed_portfolio_report.ReportOrganizer")
    def test_report_organizer_initialization(self, mock_report_organizer):
        """Test that report organizer is properly initialized."""
        mock_organizer_instance = MagicMock()
        mock_report_organizer.return_value = mock_organizer_instance

        reporter = DetailedPortfolioReporter()

        assert reporter.report_organizer is not None
        mock_report_organizer.assert_called_once()


class TestIntegration:
    """Integration tests for the complete reporting workflow."""

    @patch.object(DetailedPortfolioReporter, "_analyze_asset_with_timeframes")
    @patch.object(DetailedPortfolioReporter, "_generate_html_report")
    def test_complete_workflow_single_asset(self, mock_generate_html, mock_analyze):
        """Test complete workflow with single asset."""
        # Setup detailed mocks
        mock_analyze.return_value = (
            {
                "strategy": "BuyAndHold",
                "timeframe": "1d",
                "sharpe_ratio": 1.5,
                "total_return": 0.15,
            },
            {
                "metrics": {
                    "sharpe_ratio": 1.5,
                    "total_return": 0.15,
                    "max_drawdown": 0.08,
                },
                "trades": [],
                "equity_curve": [],
            },
        )
        mock_generate_html.return_value = "<html><body>Complete Report</body></html>"

        reporter = DetailedPortfolioReporter()

        portfolio_config = {
            "symbols": ["AAPL"],
            "portfolio_name": "Single Asset Portfolio",
            "allocation": {"AAPL": 1.0},
        }

        result = reporter.generate_comprehensive_report(
            portfolio_config=portfolio_config,
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategies=["BuyAndHold"],
            timeframes=["1d"],
        )

        # Verify workflow completion
        assert isinstance(result, str)
        assert len(result) > 0
        mock_analyze.assert_called_once()
        mock_generate_html.assert_called_once()

    @patch.object(DetailedPortfolioReporter, "_analyze_asset_with_timeframes")
    @patch.object(DetailedPortfolioReporter, "_generate_html_report")
    def test_complete_workflow_multiple_assets(self, mock_generate_html, mock_analyze):
        """Test complete workflow with multiple assets."""
        # Setup mocks for multiple assets
        mock_analyze.side_effect = [
            ({"strategy": "BuyAndHold", "timeframe": "1d"}, {"metrics": {}}),
            ({"strategy": "MeanReversion", "timeframe": "4h"}, {"metrics": {}}),
            ({"strategy": "Momentum", "timeframe": "1h"}, {"metrics": {}}),
        ]
        mock_generate_html.return_value = "<html>Multi-Asset Report</html>"

        reporter = DetailedPortfolioReporter()

        portfolio_config = {
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "portfolio_name": "Diversified Portfolio",
        }

        result = reporter.generate_comprehensive_report(
            portfolio_config=portfolio_config,
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategies=["BuyAndHold", "MeanReversion", "Momentum"],
        )

        assert isinstance(result, str)
        assert mock_analyze.call_count == 3  # Called once per asset
        mock_generate_html.assert_called_once()

    def test_error_handling_workflow(self):
        """Test error handling in reporting workflow."""
        reporter = DetailedPortfolioReporter()

        # Test with invalid portfolio config
        invalid_config = {}  # Missing required keys

        with pytest.raises(KeyError):
            reporter.generate_comprehensive_report(
                portfolio_config=invalid_config,
                start_date="2023-01-01",
                end_date="2023-12-31",
                strategies=["BuyAndHold"],
            )
