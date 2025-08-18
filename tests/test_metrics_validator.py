"""
Tests for Metrics Validator functionality.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.metrics_validator import MetricsValidator


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    return MagicMock()


@pytest.fixture
def validator(mock_db_session):
    """Create metrics validator with mocked dependencies."""
    with patch(
        "src.utils.metrics_validator.get_db_session", return_value=mock_db_session
    ):
        return MetricsValidator()


class TestMetricsValidator:
    """Test cases for MetricsValidator."""

    def test_initialization(self, validator):
        """Test proper initialization of the validator."""
        assert validator is not None

    def test_validate_best_strategy_metrics_success(self, validator):
        """Test successful validation of strategy metrics."""
        # Mock database data
        mock_strategy = MagicMock()
        mock_strategy.sortino_ratio = 1.5
        mock_strategy.sharpe_ratio = 1.2
        mock_strategy.total_return = 25.0
        mock_strategy.max_drawdown = -15.0
        mock_strategy.volatility = 20.0

        mock_result = MagicMock()
        mock_result.sortino_ratio = 1.48
        mock_result.sharpe_ratio = 1.18
        mock_result.total_return = 24.5
        mock_result.max_drawdown = -14.8
        mock_result.volatility = 19.8

        with (
            patch.object(validator, "_get_best_strategy", return_value=mock_strategy),
            patch.object(validator, "_get_backtest_result", return_value=mock_result),
        ):
            result = validator.validate_best_strategy_metrics(
                "BTCUSDT", "BuyAndHold", "1d", tolerance=0.05
            )

            assert result is not None
            assert result["symbol"] == "BTCUSDT"
            assert result["strategy"] == "BuyAndHold"
            assert result["is_valid"]
            assert len(result["discrepancies"]) == 0

    def test_validate_best_strategy_metrics_with_discrepancies(self, validator):
        """Test validation with metrics discrepancies."""
        # Mock database data with significant differences
        mock_strategy = MagicMock()
        mock_strategy.sortino_ratio = 1.5
        mock_strategy.sharpe_ratio = 1.2
        mock_strategy.total_return = 25.0
        mock_strategy.max_drawdown = -15.0
        mock_strategy.volatility = 20.0

        mock_result = MagicMock()
        mock_result.sortino_ratio = 1.0  # Significant difference
        mock_result.sharpe_ratio = 0.8  # Significant difference
        mock_result.total_return = 15.0  # Significant difference
        mock_result.max_drawdown = -25.0  # Significant difference
        mock_result.volatility = 30.0  # Significant difference

        with (
            patch.object(validator, "_get_best_strategy", return_value=mock_strategy),
            patch.object(validator, "_get_backtest_result", return_value=mock_result),
        ):
            result = validator.validate_best_strategy_metrics(
                "BTCUSDT", "BuyAndHold", "1d", tolerance=0.05
            )

            assert result is not None
            assert result["symbol"] == "BTCUSDT"
            assert result["strategy"] == "BuyAndHold"
            assert not result["is_valid"]
            assert len(result["discrepancies"]) > 0

    def test_validate_best_strategy_metrics_missing_data(self, validator):
        """Test validation when data is missing."""
        with (
            patch.object(validator, "_get_best_strategy", return_value=None),
            patch.object(validator, "_get_backtest_result", return_value=None),
        ):
            result = validator.validate_best_strategy_metrics(
                "NONEXISTENT", "BuyAndHold", "1d"
            )

            assert result is not None
            assert result["symbol"] == "NONEXISTENT"
            assert result["strategy"] == "BuyAndHold"
            assert not result["is_valid"]
            assert "error" in result

    def test_validate_multiple_strategies(self, validator):
        """Test validation of multiple strategies."""
        # Mock multiple strategies
        mock_strategies = [
            MagicMock(symbol="BTCUSDT", best_strategy="BuyAndHold", timeframe="1d"),
            MagicMock(symbol="ETHUSDT", best_strategy="RSI", timeframe="1d"),
            MagicMock(symbol="ADAUSDT", best_strategy="MACD", timeframe="1d"),
        ]

        with (
            patch.object(
                validator, "_get_top_strategies", return_value=mock_strategies
            ),
            patch.object(validator, "validate_best_strategy_metrics") as mock_validate,
        ):
            # Mock individual validation results
            mock_validate.side_effect = [
                {"symbol": "BTCUSDT", "is_valid": True, "discrepancies": []},
                {
                    "symbol": "ETHUSDT",
                    "is_valid": False,
                    "discrepancies": ["sortino_ratio"],
                },
                {"symbol": "ADAUSDT", "is_valid": True, "discrepancies": []},
            ]

            results = validator.validate_multiple_strategies(limit=3)

            assert len(results) == 3
            assert results[0]["symbol"] == "BTCUSDT"
            assert results[1]["symbol"] == "ETHUSDT"
            assert results[2]["symbol"] == "ADAUSDT"

            # Check that validation was called for each strategy
            assert mock_validate.call_count == 3

    def test_generate_validation_report_single_result(self, validator):
        """Test generation of validation report for single result."""
        result = {
            "symbol": "BTCUSDT",
            "strategy": "BuyAndHold",
            "timeframe": "1d",
            "is_valid": True,
            "discrepancies": [],
            "tolerance": 0.05,
        }

        report = validator.generate_validation_report(result)

        assert "BTCUSDT" in report
        assert "BuyAndHold" in report
        assert "VALID" in report
        assert "No discrepancies found" in report

    def test_generate_validation_report_with_discrepancies(self, validator):
        """Test generation of validation report with discrepancies."""
        result = {
            "symbol": "BTCUSDT",
            "strategy": "BuyAndHold",
            "timeframe": "1d",
            "is_valid": False,
            "discrepancies": [
                {
                    "metric": "sortino_ratio",
                    "best_strategy_value": 1.5,
                    "backtest_result_value": 1.0,
                    "difference": 0.5,
                    "percentage_diff": 33.33,
                }
            ],
            "tolerance": 0.05,
        }

        report = validator.generate_validation_report(result)

        assert "BTCUSDT" in report
        assert "BuyAndHold" in report
        assert "INVALID" in report
        assert "sortino_ratio" in report
        assert "33.33%" in report

    def test_generate_validation_report_multiple_results(self, validator):
        """Test generation of validation report for multiple results."""
        results = [
            {
                "symbol": "BTCUSDT",
                "strategy": "BuyAndHold",
                "timeframe": "1d",
                "is_valid": True,
                "discrepancies": [],
            },
            {
                "symbol": "ETHUSDT",
                "strategy": "RSI",
                "timeframe": "1d",
                "is_valid": False,
                "discrepancies": [{"metric": "sharpe_ratio"}],
            },
        ]

        report = validator.generate_validation_report(results)

        assert "BTCUSDT" in report
        assert "ETHUSDT" in report
        assert "Summary" in report
        assert "2 strategies" in report
        assert "1 valid" in report
        assert "1 invalid" in report

    def test_tolerance_calculation(self, validator):
        """Test tolerance calculation for metrics comparison."""
        # Test within tolerance
        assert validator._is_within_tolerance(
            1.0, 1.05, 0.1
        )  # 5% difference, 10% tolerance
        assert validator._is_within_tolerance(
            100.0, 95.0, 0.1
        )  # 5% difference, 10% tolerance

        # Test outside tolerance
        assert not validator._is_within_tolerance(
            1.0, 1.15, 0.1
        )  # 15% difference, 10% tolerance
        assert not validator._is_within_tolerance(
            100.0, 85.0, 0.1
        )  # 15% difference, 10% tolerance

    def test_edge_cases(self, validator):
        """Test edge cases in validation."""
        # Test with zero values
        assert validator._is_within_tolerance(0.0, 0.0, 0.1)
        assert not validator._is_within_tolerance(0.0, 1.0, 0.1)

        # Test with negative values
        assert validator._is_within_tolerance(-1.0, -1.05, 0.1)
        assert not validator._is_within_tolerance(-1.0, -1.15, 0.1)

    def test_metrics_comparison(self, validator):
        """Test metrics comparison functionality."""
        best_strategy_metrics = {
            "sortino_ratio": 1.5,
            "sharpe_ratio": 1.2,
            "total_return": 25.0,
            "max_drawdown": -15.0,
            "volatility": 20.0,
        }

        backtest_metrics = {
            "sortino_ratio": 1.48,  # Within tolerance
            "sharpe_ratio": 1.0,  # Outside tolerance
            "total_return": 24.5,  # Within tolerance
            "max_drawdown": -14.8,  # Within tolerance
            "volatility": 19.8,  # Within tolerance
        }

        discrepancies = validator._compare_metrics(
            best_strategy_metrics, backtest_metrics, tolerance=0.05
        )

        assert len(discrepancies) == 1
        assert discrepancies[0]["metric"] == "sharpe_ratio"
        assert discrepancies[0]["percentage_diff"] > 15  # Should be around 16.67%
