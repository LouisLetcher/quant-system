"""Unit tests for PortfolioManager."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from src.core.backtest_engine import BacktestResult
from src.core.portfolio_manager import PortfolioManager


class TestPortfolioManager:
    """Test cases for PortfolioManager."""

    @pytest.fixture
    def mock_backtest_engine(self):
        """Mock backtest engine."""
        engine = Mock()
        engine.batch_backtest.return_value = []
        return engine

    @pytest.fixture
    def portfolio_manager(self, mock_backtest_engine):
        """Create PortfolioManager instance."""
        return PortfolioManager(backtest_engine=mock_backtest_engine)

    @pytest.fixture
    def sample_backtest_results(self):
        """Sample backtest results."""
        return [
            BacktestResult(
                symbol="AAPL",
                strategy="rsi",
                config={},
                total_return=0.15,
                annualized_return=0.12,
                sharpe_ratio=1.2,
                sortino_ratio=1.5,
                max_drawdown=-0.08,
                volatility=0.18,
                beta=1.1,
                alpha=0.02,
                var_95=-0.05,
                cvar_95=-0.07,
                calmar_ratio=1.5,
                omega_ratio=1.3,
                trades_count=25,
                win_rate=0.64,
                avg_win=0.05,
                avg_loss=-0.03,
                profit_factor=2.1,
                kelly_criterion=0.15,
                start_date="2023-01-01",
                end_date="2023-12-31",
                duration_days=365,
                equity_curve=pd.Series([10000, 10500, 11000, 11500]),
                trades=pd.DataFrame(),
                drawdown_curve=pd.Series([0, -0.02, -0.05, -0.01]),
            ),
            BacktestResult(
                symbol="MSFT",
                strategy="rsi",
                config={},
                total_return=0.18,
                annualized_return=0.16,
                sharpe_ratio=1.4,
                sortino_ratio=1.7,
                max_drawdown=-0.06,
                volatility=0.16,
                beta=0.9,
                alpha=0.04,
                var_95=-0.04,
                cvar_95=-0.06,
                calmar_ratio=2.67,
                omega_ratio=1.5,
                trades_count=28,
                win_rate=0.68,
                avg_win=0.06,
                avg_loss=-0.025,
                profit_factor=2.4,
                kelly_criterion=0.18,
                start_date="2023-01-01",
                end_date="2023-12-31",
                duration_days=365,
                equity_curve=pd.Series([10000, 10600, 11200, 11800]),
                trades=pd.DataFrame(),
                drawdown_curve=pd.Series([0, -0.01, -0.03, -0.02]),
            ),
        ]

    @pytest.fixture
    def sample_portfolios(self):
        """Sample portfolio configurations."""
        return {
            "tech_growth": {
                "name": "Tech Growth",
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "strategies": ["rsi", "macd"],
                "risk_profile": "aggressive",
                "target_return": 0.15,
            },
            "conservative": {
                "name": "Conservative Mix",
                "symbols": ["SPY", "BND", "VTI"],
                "strategies": ["sma_crossover"],
                "risk_profile": "conservative",
                "target_return": 0.08,
            },
        }

    def test_init(self, portfolio_manager, mock_backtest_engine):
        """Test initialization."""
        assert portfolio_manager.backtest_engine == mock_backtest_engine
        assert isinstance(portfolio_manager.portfolios, dict)
        assert len(portfolio_manager.portfolios) == 0

    def test_add_portfolio(self, portfolio_manager):
        """Test adding portfolios."""
        portfolio_config = {
            "name": "Test Portfolio",
            "symbols": ["AAPL", "MSFT"],
            "strategies": ["rsi"],
            "risk_profile": "moderate",
        }

        portfolio_manager.add_portfolio("test_portfolio", portfolio_config)

        assert "test_portfolio" in portfolio_manager.portfolios
        assert (
            portfolio_manager.portfolios["test_portfolio"]["name"] == "Test Portfolio"
        )

    def test_remove_portfolio(self, portfolio_manager):
        """Test removing portfolios."""
        portfolio_config = {
            "name": "Test Portfolio",
            "symbols": ["AAPL", "MSFT"],
            "strategies": ["rsi"],
        }

        portfolio_manager.add_portfolio("test_portfolio", portfolio_config)
        portfolio_manager.remove_portfolio("test_portfolio")

        assert "test_portfolio" not in portfolio_manager.portfolios

    def test_backtest_portfolio(self, portfolio_manager, sample_backtest_results):
        """Test backtesting a portfolio."""
        portfolio_config = {"symbols": ["AAPL", "MSFT"], "strategies": ["rsi"]}

        # Mock the backtest engine to return sample results
        portfolio_manager.backtest_engine.batch_backtest.return_value = (
            sample_backtest_results
        )

        results = portfolio_manager.backtest_portfolio(
            portfolio_config, start_date="2023-01-01", end_date="2023-12-31"
        )

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, BacktestResult) for r in results)

    def test_compare_portfolios(
        self, portfolio_manager, sample_portfolios, sample_backtest_results
    ):
        """Test comparing multiple portfolios."""
        # Add portfolios
        for portfolio_id, config in sample_portfolios.items():
            portfolio_manager.add_portfolio(portfolio_id, config)

        # Mock backtest results
        portfolio_manager.backtest_engine.batch_backtest.return_value = (
            sample_backtest_results
        )

        comparison = portfolio_manager.compare_portfolios(
            start_date="2023-01-01", end_date="2023-12-31"
        )

        assert isinstance(comparison, dict)
        assert "portfolio_results" in comparison
        assert "rankings" in comparison
        assert "summary" in comparison

    def test_calculate_portfolio_metrics(
        self, portfolio_manager, sample_backtest_results
    ):
        """Test calculating portfolio-level metrics."""
        metrics = portfolio_manager._calculate_portfolio_metrics(
            sample_backtest_results
        )

        assert isinstance(metrics, dict)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "volatility" in metrics
        assert "win_rate" in metrics

    def test_calculate_risk_score(self, portfolio_manager, sample_backtest_results):
        """Test risk score calculation."""
        risk_score = portfolio_manager._calculate_risk_score(sample_backtest_results)

        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 100

    def test_generate_investment_recommendations(
        self, portfolio_manager, sample_portfolios, sample_backtest_results
    ):
        """Test generating investment recommendations."""
        # Add portfolios and mock results
        for portfolio_id, config in sample_portfolios.items():
            portfolio_manager.add_portfolio(portfolio_id, config)

        portfolio_manager.backtest_engine.batch_backtest.return_value = (
            sample_backtest_results
        )

        recommendations = portfolio_manager.generate_investment_recommendations(
            capital=100000,
            risk_tolerance="moderate",
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        assert isinstance(recommendations, dict)
        assert "recommended_allocations" in recommendations
        assert "expected_return" in recommendations
        assert "expected_risk" in recommendations
        assert "investment_plan" in recommendations

    def test_optimize_portfolio_allocation(
        self, portfolio_manager, sample_backtest_results
    ):
        """Test portfolio allocation optimization."""
        allocations = portfolio_manager._optimize_allocation(
            sample_backtest_results, risk_tolerance="moderate"
        )

        assert isinstance(allocations, dict)
        assert sum(allocations.values()) == pytest.approx(1.0, rel=1e-2)

        for allocation in allocations.values():
            assert 0 <= allocation <= 1

    def test_generate_investment_plan(self, portfolio_manager):
        """Test investment plan generation."""
        allocations = {"AAPL": 0.6, "MSFT": 0.4}

        plan = portfolio_manager._generate_investment_plan(
            allocations=allocations,
            capital=100000,
            expected_return=0.15,
            expected_risk=0.18,
        )

        assert isinstance(plan, dict)
        assert "total_investment" in plan
        assert "allocations" in plan
        assert "expected_annual_return" in plan
        assert "estimated_risk" in plan

    def test_rank_portfolios(self, portfolio_manager):
        """Test portfolio ranking."""
        portfolio_metrics = {
            "portfolio_1": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.08,
                "risk_score": 65,
            },
            "portfolio_2": {
                "total_return": 0.18,
                "sharpe_ratio": 1.4,
                "max_drawdown": -0.06,
                "risk_score": 55,
            },
        }

        rankings = portfolio_manager._rank_portfolios(portfolio_metrics)

        assert isinstance(rankings, list)
        assert len(rankings) == 2
        assert rankings[0][0] == "portfolio_2"  # Should be ranked higher

    @pytest.mark.parametrize(
        ("risk_tolerance", "expected_weights"),
        [
            ("conservative", {"return": 0.2, "sharpe": 0.3, "drawdown": 0.5}),
            ("moderate", {"return": 0.4, "sharpe": 0.4, "drawdown": 0.2}),
            ("aggressive", {"return": 0.6, "sharpe": 0.3, "drawdown": 0.1}),
        ],
    )
    def test_get_risk_weights(
        self, portfolio_manager, risk_tolerance, expected_weights
    ):
        """Test risk tolerance weight mapping."""
        weights = portfolio_manager._get_risk_weights(risk_tolerance)

        assert isinstance(weights, dict)
        for key, expected_value in expected_weights.items():
            assert weights[key] == expected_value

    def test_validate_portfolio_config(self, portfolio_manager):
        """Test portfolio configuration validation."""
        # Valid config
        valid_config = {
            "symbols": ["AAPL", "MSFT"],
            "strategies": ["rsi"],
            "name": "Test Portfolio",
        }

        is_valid = portfolio_manager._validate_portfolio_config(valid_config)
        assert is_valid

        # Invalid config - missing symbols
        invalid_config = {"strategies": ["rsi"], "name": "Test Portfolio"}

        is_valid = portfolio_manager._validate_portfolio_config(invalid_config)
        assert not is_valid

    def test_calculate_correlation_matrix(
        self, portfolio_manager, sample_backtest_results
    ):
        """Test correlation matrix calculation."""
        correlation_matrix = portfolio_manager._calculate_correlation_matrix(
            sample_backtest_results
        )

        assert isinstance(correlation_matrix, pd.DataFrame)
        assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
        assert len(correlation_matrix) == len(sample_backtest_results)

    def test_diversification_score(self, portfolio_manager, sample_backtest_results):
        """Test diversification score calculation."""
        score = portfolio_manager._calculate_diversification_score(
            sample_backtest_results
        )

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_rebalancing_recommendations(self, portfolio_manager):
        """Test rebalancing recommendations."""
        current_allocations = {"AAPL": 0.7, "MSFT": 0.3}
        target_allocations = {"AAPL": 0.6, "MSFT": 0.4}

        rebalance_actions = portfolio_manager._generate_rebalancing_actions(
            current_allocations, target_allocations, capital=100000
        )

        assert isinstance(rebalance_actions, list)
        assert len(rebalance_actions) > 0

        for action in rebalance_actions:
            assert "symbol" in action
            assert "action" in action  # 'buy' or 'sell'
            assert "amount" in action

    def test_portfolio_performance_attribution(
        self, portfolio_manager, sample_backtest_results
    ):
        """Test performance attribution analysis."""
        attribution = portfolio_manager._calculate_performance_attribution(
            sample_backtest_results
        )

        assert isinstance(attribution, dict)
        assert "individual_contributions" in attribution
        assert "interaction_effects" in attribution
        assert "total_attribution" in attribution

    def test_error_handling(self, portfolio_manager):
        """Test error handling in portfolio operations."""
        # Test with empty portfolio config
        with pytest.raises(ValueError):
            portfolio_manager.backtest_portfolio({}, "2023-01-01", "2023-12-31")

        # Test with invalid risk tolerance
        with pytest.raises(ValueError):
            portfolio_manager.generate_investment_recommendations(
                capital=100000,
                risk_tolerance="invalid",
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

    def test_portfolio_stress_testing(self, portfolio_manager, sample_backtest_results):
        """Test portfolio stress testing."""
        stress_results = portfolio_manager._perform_stress_test(
            sample_backtest_results,
            scenarios={
                "market_crash": {"return_shock": -0.2, "volatility_shock": 0.5},
                "interest_rate_rise": {"return_shock": -0.1, "volatility_shock": 0.2},
            },
        )

        assert isinstance(stress_results, dict)
        assert "market_crash" in stress_results
        assert "interest_rate_rise" in stress_results

    def test_portfolio_summary_statistics(
        self, portfolio_manager, sample_backtest_results
    ):
        """Test portfolio summary statistics generation."""
        summary = portfolio_manager._generate_portfolio_summary(sample_backtest_results)

        assert isinstance(summary, dict)
        assert "asset_count" in summary
        assert "total_trades" in summary
        assert "avg_holding_period" in summary
        assert "sector_allocation" in summary

    def test_concurrent_portfolio_analysis(
        self, portfolio_manager, sample_portfolios, sample_backtest_results
    ):
        """Test concurrent analysis of multiple portfolios."""
        # Add multiple portfolios
        for portfolio_id, config in sample_portfolios.items():
            portfolio_manager.add_portfolio(portfolio_id, config)

        # Mock backtest results
        portfolio_manager.backtest_engine.batch_backtest.return_value = (
            sample_backtest_results
        )

        # Run concurrent analysis
        results = portfolio_manager.analyze_portfolios_concurrent(
            start_date="2023-01-01", end_date="2023-12-31", max_workers=2
        )

        assert isinstance(results, dict)
        assert len(results) == len(sample_portfolios)
