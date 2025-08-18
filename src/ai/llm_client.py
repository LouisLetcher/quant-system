"""
LLM Client for AI-powered investment analysis.
Supports OpenAI and Anthropic models.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import openai
from anthropic import Anthropic


class LLMClientError(Exception):
    """Custom exception for LLM client errors."""


class LLMClient:
    """Client for interacting with LLM providers for investment analysis."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize clients based on available API keys
        self.openai_client = None
        self.anthropic_client = None

        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = openai
            self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")

        if os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.anthropic_model = os.getenv(
                "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
            )

        if not self.openai_client and not self.anthropic_client:
            self.logger.warning(
                "No LLM API keys configured - AI features will be limited"
            )

    def analyze_portfolio(
        self, analysis_data: dict, recommendations: list
    ) -> dict[str, Any]:
        """Analyze portfolio recommendations using AI."""

        # Prepare prompt
        prompt = self._create_portfolio_analysis_prompt(analysis_data, recommendations)

        try:
            if self.openai_client:
                return self._query_openai(prompt)
            if self.anthropic_client:
                return self._query_anthropic(prompt)
            raise LLMClientError("No LLM client available")

        except Exception as e:
            self.logger.error("LLM analysis failed: %s", e)
            return {
                "reasoning": "Quantitative analysis complete - AI reasoning unavailable",
                "warnings": ["LLM service unavailable"],
            }

    def explain_asset_recommendation(self, asset_data: dict) -> dict[str, Any]:
        """Generate detailed explanation for an asset recommendation."""

        prompt = self._create_asset_explanation_prompt(asset_data)

        try:
            if self.openai_client:
                response = self._query_openai(prompt)
            elif self.anthropic_client:
                response = self._query_anthropic(prompt)
            else:
                raise LLMClientError("No LLM client available")

            return {
                "summary": response.get("summary", "Asset analysis complete"),
                "strengths": response.get("strengths", []),
                "concerns": response.get("concerns", []),
                "recommendation": response.get("recommendation", "Review metrics"),
            }

        except Exception as e:
            self.logger.error("Asset explanation failed: %s", e)
            return {
                "summary": f"Quantitative analysis for {asset_data.get('symbol', 'asset')}",
                "strengths": ["Metrics available for analysis"],
                "concerns": ["AI explanation unavailable"],
                "recommendation": "Review quantitative metrics manually",
            }

    def _create_portfolio_analysis_prompt(
        self, data: dict, recommendations: list
    ) -> str:
        """Create prompt for portfolio analysis."""

        return f"""
Analyze this quantitative trading portfolio and provide investment recommendations.

Portfolio Overview:
- Risk Tolerance: {data["risk_tolerance"]}
- Number of Assets: {data["num_recommendations"]}
- Average Sortino Ratio: {data["avg_sortino"]:.2f}
- Average Calmar Ratio: {data["avg_calmar"]:.2f}
- Diversification Score: {data["diversification_score"]:.2f}
- Total Red Flags: {data["red_flags_count"]}

Asset Details:
{self._format_recommendations_for_prompt(recommendations)}

Please provide:
1. Overall portfolio assessment (2-3 sentences)
2. Key strengths and concerns
3. Risk level appropriateness for {data["risk_tolerance"]} investor
4. Any warnings or recommendations

Return JSON format:
{{
    "reasoning": "Overall analysis...",
    "warnings": ["warning1", "warning2"]
}}
"""

    def _create_asset_explanation_prompt(self, asset_data: dict) -> str:
        """Create prompt for individual asset explanation."""

        return f"""
Explain this trading asset recommendation based on quantitative backtest results:

Asset: {asset_data.get("symbol", "Unknown")}
Strategy: {asset_data.get("strategy", "Unknown")}

Performance Metrics:
- Sortino Ratio: {asset_data.get("sortino_ratio", 0):.2f}
- Calmar Ratio: {asset_data.get("calmar_ratio", 0):.2f}
- Max Drawdown: {asset_data.get("max_drawdown", 0):.1%}
- Total Return: {asset_data.get("total_return", 0):.1%}
- Win Rate: {asset_data.get("win_rate", 0):.1%}
- Profit Factor: {asset_data.get("profit_factor", 0):.2f}

Provide:
1. Brief summary of performance
2. Main strengths
3. Key concerns or risks
4. Investment recommendation

Return JSON format:
{{
    "summary": "Brief performance summary...",
    "strengths": ["strength1", "strength2"],
    "concerns": ["concern1", "concern2"],
    "recommendation": "Overall recommendation..."
}}
"""

    def _format_recommendations_for_prompt(self, recommendations: list) -> str:
        """Format recommendations for LLM prompt."""
        formatted = []

        for rec in recommendations[:5]:  # Limit to top 5 for prompt length
            formatted.append(
                f"- {rec.symbol} ({rec.strategy}): "
                f"Score={rec.score:.2f}, "
                f"Sortino={rec.sortino_ratio:.2f}, "
                f"Allocation={rec.allocation_percentage:.1f}%, "
                f"Red Flags={len(rec.red_flags)}"
            )

        return "\n".join(formatted)

    def _query_openai(self, prompt: str) -> dict[str, Any]:
        """Query OpenAI model."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quantitative investment analyst. Provide concise, data-driven investment analysis based on backtest metrics.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            content = response.choices[0].message.content

            # Try to parse as JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"reasoning": content, "warnings": []}

        except Exception as e:
            self.logger.error("OpenAI query failed: %s", e)
            raise

    def _query_anthropic(self, prompt: str) -> dict[str, Any]:
        """Query Anthropic Claude model."""
        try:
            response = self.anthropic_client.messages.create(
                model=self.anthropic_model,
                max_tokens=1000,
                temperature=0.3,
                system="You are a quantitative investment analyst. Provide concise, data-driven investment analysis based on backtest metrics.",
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text

            # Try to parse as JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"reasoning": content, "warnings": []}

        except Exception as e:
            self.logger.error("Anthropic query failed: %s", e)
            raise
