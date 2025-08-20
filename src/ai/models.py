"""AI Models for Investment Recommendations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AssetRecommendation:
    """AI recommendation for a single asset."""

    symbol: str
    strategy: str
    score: float
    confidence: float
    allocation_percentage: float
    risk_level: str
    reasoning: str
    red_flags: list[str]
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    trading_style: str
    timeframe: str
    position_size: float
    risk_per_trade: float
    stop_loss: float
    take_profit: float

    @property
    def confidence_score(self) -> float:
        return self.confidence

    @property
    def sharpe_ratio(self) -> float:
        # Calculate approximate Sharpe from available data
        return self.sortino_ratio * 0.8  # Rough approximation

    @property
    def recommendation_type(self) -> str:
        if self.confidence > 0.7 and self.sortino_ratio > 1.0:
            return "BUY"
        if self.confidence > 0.5:
            return "HOLD"
        return "SELL"

    @property
    def risk_score(self) -> float:
        return abs(self.max_drawdown) / 100


@dataclass
class PortfolioRecommendation:
    """AI recommendation for a portfolio."""

    recommendations: list[AssetRecommendation]
    total_score: float
    risk_profile: str
    diversification_score: float
    correlation_analysis: dict
    overall_reasoning: str
    warnings: list[str]
    confidence: float

    @property
    def total_assets(self) -> int:
        return len(self.recommendations)

    @property
    def expected_return(self) -> float:
        if not self.recommendations:
            return 0.0
        return sum(r.total_return for r in self.recommendations) / len(
            self.recommendations
        )

    @property
    def asset_recommendations(self) -> list[AssetRecommendation]:
        return self.recommendations

    @property
    def reasoning(self) -> str:
        return self.overall_reasoning

    @property
    def confidence_score(self) -> float:
        return self.confidence
