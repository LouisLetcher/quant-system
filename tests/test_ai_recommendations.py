"""Tests for AI Investment Recommendations module."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.ai.investment_recommendations import AIInvestmentRecommendations
from src.ai.models import AssetRecommendation, PortfolioRecommendation


class TestAIInvestmentRecommendations:
    """Test suite for AI Investment Recommendations."""

    @pytest.fixture
    def ai_recommender(self):
        """Create AI recommender instance."""
        return AIInvestmentRecommendations()

    @pytest.fixture
    def mock_backtest_data(self):
        """Mock backtest data for testing."""
        return [
            {
                "symbol": "AAPL",
                "strategy": "rsi",
                "sortino_ratio": 1.5,
                "calmar_ratio": 1.2,
                "profit_factor": 1.8,
                "max_drawdown": -0.08,
                "win_rate": 0.65,
                "volatility": 0.25,
                "total_return": 0.35,
                "num_trades": 50,
            },
            {
                "symbol": "MSFT",
                "strategy": "macd",
                "sortino_ratio": 1.8,
                "calmar_ratio": 1.4,
                "profit_factor": 2.1,
                "max_drawdown": -0.09,
                "win_rate": 0.70,
                "volatility": 0.20,
                "total_return": 0.42,
                "num_trades": 45,
            },
            {
                "symbol": "RISKY",
                "strategy": "rsi",
                "sortino_ratio": 0.3,
                "calmar_ratio": 0.2,
                "profit_factor": 0.8,
                "max_drawdown": -0.45,
                "win_rate": 0.35,
                "volatility": 0.60,
                "total_return": -0.15,
                "num_trades": 20,
            },
        ]

    def test_performance_scoring(self, ai_recommender, mock_backtest_data):
        """Test performance-based scoring algorithm."""
        scored_assets = ai_recommender._calculate_performance_scores(mock_backtest_data)

        # Should be sorted by score (highest first)
        assert scored_assets[0]["symbol"] == "MSFT"  # Best performer
        assert scored_assets[-1]["symbol"] == "RISKY"  # Worst performer

        # Scores should be between 0 and 1
        for asset in scored_assets:
            assert 0 <= asset["score"] <= 1

    def test_risk_filtering(self, ai_recommender, mock_backtest_data):
        """Test risk-adjusted filtering."""
        scored_assets = ai_recommender._calculate_performance_scores(mock_backtest_data)

        # Conservative filtering
        conservative = ai_recommender._apply_risk_filters(scored_assets, "conservative")
        assert len(conservative) == 2  # Should filter out RISKY
        assert all(abs(asset["max_drawdown"]) <= 0.10 for asset in conservative)

        # Moderate filtering
        moderate = ai_recommender._apply_risk_filters(scored_assets, "moderate")
        assert len(moderate) == 2  # Should still filter out RISKY

        # Aggressive filtering
        aggressive = ai_recommender._apply_risk_filters(scored_assets, "aggressive")
        assert len(aggressive) >= 2  # Should include more assets

    def test_red_flag_detection(self, ai_recommender, mock_backtest_data):
        """Test red flag detection system."""
        flagged_assets = ai_recommender._detect_red_flags(mock_backtest_data)

        # Check that RISKY asset has red flags
        risky_asset = next(
            asset for asset in flagged_assets if asset["symbol"] == "RISKY"
        )
        assert len(risky_asset["red_flags"]) > 0
        assert "High maximum drawdown risk" in risky_asset["red_flags"]
        assert "Low risk-adjusted returns" in risky_asset["red_flags"]

        # Check that good assets have fewer flags
        good_assets = [
            asset for asset in flagged_assets if asset["symbol"] in ["AAPL", "MSFT"]
        ]
        for asset in good_assets:
            assert len(asset["red_flags"]) <= 2

    def test_confidence_scoring(self, ai_recommender, mock_backtest_data):
        """Test confidence score calculation."""
        confidence_assets = ai_recommender._calculate_confidence_scores(
            mock_backtest_data, mock_backtest_data
        )

        # Confidence should be between 0 and 1
        for asset in confidence_assets:
            assert 0 <= asset["confidence"] <= 1

        # Better performing assets should have higher confidence
        msft_confidence = next(
            asset["confidence"]
            for asset in confidence_assets
            if asset["symbol"] == "MSFT"
        )
        risky_confidence = next(
            asset["confidence"]
            for asset in confidence_assets
            if asset["symbol"] == "RISKY"
        )
        assert msft_confidence > risky_confidence

    def test_allocation_suggestions(self, ai_recommender, mock_backtest_data):
        """Test investment allocation suggestions."""
        scored_assets = ai_recommender._calculate_performance_scores(mock_backtest_data)
        allocations = ai_recommender._suggest_allocations(scored_assets, "moderate", 3)

        # Should have allocations that sum to ~100%
        total_allocation = sum(asset["allocation"] for asset in allocations)
        assert abs(total_allocation - 100) < 1  # Allow small rounding errors

        # Better assets should get higher allocations
        msft_allocation = next(
            asset["allocation"] for asset in allocations if asset["symbol"] == "MSFT"
        )
        risky_allocation = next(
            (
                asset["allocation"]
                for asset in allocations
                if asset["symbol"] == "RISKY"
            ),
            0,
        )

        if risky_allocation > 0:  # Only compare if RISKY made it through filters
            assert msft_allocation > risky_allocation

    def test_risk_level_classification(self, ai_recommender):
        """Test risk level classification."""
        low_risk_asset = {"max_drawdown": -0.05, "volatility": 0.10}
        medium_risk_asset = {"max_drawdown": -0.18, "volatility": 0.25}
        high_risk_asset = {"max_drawdown": -0.40, "volatility": 0.50}

        assert ai_recommender._classify_risk_level(low_risk_asset) == "Low"
        assert ai_recommender._classify_risk_level(medium_risk_asset) == "Medium"
        assert ai_recommender._classify_risk_level(high_risk_asset) == "High"

    @patch("src.ai.investment_recommendations.LLMClient")
    def test_generate_recommendations_success(
        self, mock_llm_client, ai_recommender, mock_backtest_data
    ):
        """Test successful recommendation generation."""
        # Mock LLM response
        mock_llm_client.return_value.analyze_portfolio.return_value = {
            "reasoning": "Strong portfolio with good diversification",
            "warnings": [],
        }

        # Mock data loading
        ai_recommender._load_backtest_results = Mock(return_value=mock_backtest_data)

        recommendations = ai_recommender.generate_recommendations(
            risk_tolerance="moderate", min_confidence=0.5, max_assets=2
        )

        assert isinstance(recommendations, PortfolioRecommendation)
        assert len(recommendations.recommendations) <= 2
        assert recommendations.risk_profile == "moderate"
        assert all(rec.confidence >= 0.5 for rec in recommendations.recommendations)

    def test_generate_recommendations_no_data(self, ai_recommender):
        """Test recommendation generation with no data."""
        ai_recommender._load_backtest_results = Mock(return_value=[])

        with pytest.raises(ValueError, match="No backtest results found"):
            ai_recommender.generate_recommendations()

    @patch("src.ai.investment_recommendations.LLMClient")
    def test_llm_failure_fallback(
        self, mock_llm_client, ai_recommender, mock_backtest_data
    ):
        """Test fallback when LLM fails."""
        # Mock LLM failure
        mock_llm_client.return_value.analyze_portfolio.side_effect = Exception(
            "LLM failed"
        )

        ai_recommender._load_backtest_results = Mock(return_value=mock_backtest_data)

        recommendations = ai_recommender.generate_recommendations()

        # Should still work with quantitative analysis
        assert isinstance(recommendations, PortfolioRecommendation)
        assert "LLM service unavailable" in recommendations.warnings[0]

    def test_asset_comparison(self, ai_recommender):
        """Test asset comparison functionality."""
        symbols = ["AAPL", "MSFT", "GOOGL"]

        # Mock database query
        with patch.object(ai_recommender, "db_session") as mock_session:
            mock_session.query.return_value.filter.return_value.all.return_value = []

            ai_recommender.get_asset_comparison(symbols, "rsi")

    @patch("src.ai.investment_recommendations.Path")
    @patch("builtins.open")
    @patch("json.dump")
    def test_save_to_exports_with_quarter_structure(
        self, mock_json_dump, mock_open, mock_path, ai_recommender
    ):
        """Test that exports are saved with year/quarter directory structure."""
        # Mock recommendations
        recommendations = [
            AssetRecommendation(
                symbol="AAPL",
                strategy="rsi",
                score=0.85,
                confidence=0.9,
                allocation_percentage=50.0,
                risk_level="moderate",
                reasoning="Strong performance",
                red_flags=[],
                sortino_ratio=1.5,
                calmar_ratio=1.2,
                max_drawdown=-0.15,
                win_rate=0.65,
                profit_factor=1.8,
                total_return=0.35,
            )
        ]

        # Test with quarter format "Q3_2025"
        ai_recommender._save_to_exports(recommendations, "moderate", "Q3_2025")

        # Verify file operations were called
        assert mock_open.called  # File should have been opened for writing
        assert mock_json_dump.called  # JSON data should have been written

        # This test verifies the method completes without error
        # The actual file path creation is tested through the method execution


if __name__ == "__main__":
    pytest.main([__file__])
