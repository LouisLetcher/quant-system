"""Data models for AI Investment Recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AssetRecommendation:
    """Individual asset recommendation."""

    symbol: str
    strategy: str
    score: float
    confidence: float
    allocation_percentage: float
    risk_level: str
    reasoning: str
    red_flags: list[str]

    # Performance metrics
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float

    # Trading parameters
    trading_style: str = "swing"  # "swing" or "scalp"
    timeframe: str = "1h"
    risk_per_trade: float = 2.0  # Percentage risk per trade
    stop_loss_points: float = 50.0  # Stop loss in points
    take_profit_points: float = 150.0  # Take profit in points
    position_size_percent: float = 10.0  # Percentage of portfolio for position sizing

    def __post_init__(self):
        """Convert numpy types to Python native types."""
        # Convert all numeric fields to Python native types
        numeric_fields = [
            "score",
            "confidence",
            "allocation_percentage",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "total_return",
            "risk_per_trade",
            "stop_loss_points",
            "take_profit_points",
            "position_size_percent",
        ]

        for field in numeric_fields:
            value = getattr(self, field)
            if (
                isinstance(value, (np.floating, np.integer, np.bool_))
                or hasattr(value, "item")
                or (isinstance(value, np.ndarray) and value.size == 1)
            ):
                setattr(self, field, value.item())


@dataclass
class PortfolioRecommendation:
    """Complete portfolio recommendation."""

    recommendations: list[AssetRecommendation]
    total_score: float
    risk_profile: str
    diversification_score: float
    correlation_analysis: dict[str, float]
    overall_reasoning: str
    warnings: list[str]
    confidence: float

    # Additional properties for HTML template
    @property
    def confidence_level(self) -> float:
        """Alias for confidence property."""
        return self.confidence

    @property
    def analysis(self) -> str:
        """Alias for overall_reasoning property."""
        return self.overall_reasoning

    @property
    def market_correlation(self) -> float:
        """Get average market correlation."""
        if (
            self.correlation_analysis
            and "market_correlation" in self.correlation_analysis
        ):
            return self.correlation_analysis["market_correlation"]
        return 0.0
