"""Tests for detailed portfolio reporting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.reporting.detailed_portfolio_report import DetailedPortfolioReporter


class TestDetailedPortfolioReporter:
    """Test cases for DetailedPortfolioReporter class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        reporter = DetailedPortfolioReporter()
        assert reporter.report_organizer is not None

    @patch("src.reporting.detailed_portfolio_report.get_sync_engine")
    @patch("src.reporting.detailed_portfolio_report.sessionmaker")
    def test_generate_comprehensive_report_basic(
        self, mock_sessionmaker, mock_get_sync_engine
    ):
        """Test basic comprehensive report generation."""
        # Setup mock database session
        mock_session = MagicMock()
        mock_sessionmaker.return_value = mock_session
        mock_get_sync_engine.return_value = MagicMock()

        # Mock database query results
        mock_session.return_value.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = []

        reporter = DetailedPortfolioReporter()

        portfolio_config = {"symbols": ["AAPL", "MSFT"], "name": "Test Portfolio"}

        result = reporter.generate_comprehensive_report(
            portfolio_config=portfolio_config,
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategies=["BuyAndHold"],
        )

        assert isinstance(result, str)
        assert "Test Portfolio" in result

    def test_generate_comprehensive_report_missing_symbols(self):
        """Test report generation with missing symbols in config."""
        reporter = DetailedPortfolioReporter()

        portfolio_config = {
            "name": "Test Portfolio"
            # Missing symbols key
        }

        with pytest.raises(KeyError):
            reporter.generate_comprehensive_report(
                portfolio_config=portfolio_config,
                start_date="2023-01-01",
                end_date="2023-12-31",
                strategies=["BuyAndHold"],
            )

    def test_create_html_method_exists(self):
        """Test that the HTML creation method exists."""
        reporter = DetailedPortfolioReporter()

        # Check that the method exists
        assert hasattr(reporter, "_create_html_report")

    def test_generate_equity_curve_empty(self):
        """Test equity curve generation with empty orders."""
        reporter = DetailedPortfolioReporter()
        result = reporter._generate_simple_equity_curve([])
        assert result == []

    def test_generate_equity_curve_with_orders(self):
        """Test equity curve generation with orders."""
        reporter = DetailedPortfolioReporter()
        orders = [
            {"date": "2023-01-01", "equity": 10000},
            {"date": "2023-01-02", "equity": 10100},
        ]
        result = reporter._generate_simple_equity_curve(orders)
        assert len(result) == 2
        assert result[0]["date"] == "2023-01-01"
        assert result[0]["equity"] == 10000


class TestIntegration:
    """Integration tests for the complete reporting workflow."""

    @patch("src.reporting.detailed_portfolio_report.get_sync_engine")
    @patch("src.reporting.detailed_portfolio_report.sessionmaker")
    def test_complete_workflow_single_asset(
        self, mock_sessionmaker, mock_get_sync_engine
    ):
        """Test complete workflow with single asset."""
        # Setup mock database session
        mock_session = MagicMock()
        mock_sessionmaker.return_value = mock_session
        mock_get_sync_engine.return_value = MagicMock()

        # Mock database query results
        mock_session.return_value.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = []

        reporter = DetailedPortfolioReporter()

        portfolio_config = {
            "symbols": ["AAPL"],
            "name": "Single Asset Portfolio",
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
