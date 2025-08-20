"""LLM Client for AI Investment Recommendations."""

from __future__ import annotations

import logging
from typing import Any


class LLMClient:
    """Simple LLM client for generating investment recommendations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_portfolio_analysis(
        self, backtest_data: list[dict], risk_profile: str
    ) -> dict[str, Any]:
        """Generate portfolio analysis using backtest data."""

        # Simple rule-based analysis since we don't have actual LLM
        # Filter by risk profile
        if risk_profile == "conservative":
            filtered_data = [d for d in backtest_data if d.get("max_drawdown", 0) > -20]
        elif risk_profile == "moderate":
            filtered_data = [d for d in backtest_data if d.get("max_drawdown", 0) > -40]
        else:  # aggressive
            filtered_data = backtest_data

        # Select top performers by Sortino ratio
        top_performers = sorted(
            filtered_data, key=lambda x: x.get("sortino_ratio", 0), reverse=True
        )[:10]

        return {
            "reasoning": f"Based on {risk_profile} risk profile, selected top {len(top_performers)} strategies with appropriate risk levels.",
            "confidence_score": 0.85,
            "expected_return": sum(d.get("total_return", 0) for d in top_performers)
            / len(top_performers)
            if top_performers
            else 0,
            "expected_risk": sum(abs(d.get("max_drawdown", 0)) for d in top_performers)
            / len(top_performers)
            if top_performers
            else 0,
            "recommendations": top_performers,
        }

    def explain_asset_recommendation(self, asset_data: dict) -> dict[str, Any]:
        """Explain a specific asset recommendation."""

        symbol = asset_data.get("symbol", "Unknown")
        strategy = asset_data.get("strategy", "Unknown")
        sortino = asset_data.get("sortino_ratio", 0)

        reasoning = f"Asset {symbol} with {strategy} strategy shows strong performance with Sortino ratio of {sortino:.3f}."

        if sortino > 1.0:
            reasoning += " This indicates excellent risk-adjusted returns."
        elif sortino > 0.5:
            reasoning += " This shows good risk-adjusted performance."
        else:
            reasoning += " Performance may need improvement."

        return {
            "reasoning": reasoning,
            "confidence_score": min(0.95, max(0.3, sortino / 2.0)),
            "recommendation": "BUY" if sortino > 0.5 else "HOLD",
        }
