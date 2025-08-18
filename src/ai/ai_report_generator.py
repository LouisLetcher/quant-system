"""
AI Investment Recommendations HTML Report Generator.
Creates professional HTML reports for AI investment recommendations organized by portfolio.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from jinja2 import BaseLoader, Environment

from .models import PortfolioRecommendation


class AIReportGenerator:
    """Generate HTML reports for AI investment recommendations."""

    def __init__(self, output_dir: str = "exports/recommendations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Setup Jinja2 environment with string templates
        self.template_env = Environment(loader=BaseLoader())

    def generate_portfolio_html_report(
        self,
        portfolio_name: str,
        recommendations: PortfolioRecommendation,
        quarter: str = None,
        year: str = None,
    ) -> str:
        """Generate HTML report for portfolio AI recommendations."""

        # Parse quarter and year or use current
        if quarter and "_" in quarter:
            # quarter might be like "Q3_2025"
            quarter_part, year_part = quarter.split("_")
        else:
            current_date = datetime.now()
            quarter_part = quarter or f"Q{(current_date.month - 1) // 3 + 1}"
            year_part = year or str(current_date.year)

        # Create organized output directory
        output_dir = self.output_dir / year_part / quarter_part
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{portfolio_name}_AI_Recommendations_{quarter_part}_{year_part}.html"
        )

        # Generate HTML content
        html_content = self._render_html_template(
            portfolio_name, recommendations, quarter_part, year_part
        )

        # Save HTML file
        output_path = output_dir / filename
        output_path.write_text(html_content, encoding="utf-8")

        self.logger.info(f"AI recommendations HTML report saved to {output_path}")
        return str(output_path)

    def _render_html_template(
        self,
        portfolio_name: str,
        recommendations: PortfolioRecommendation,
        quarter: str,
        year: str,
    ) -> str:
        """Render the HTML template with AI recommendations data."""

        template_content = self._get_html_template()
        template = self.template_env.from_string(template_content)

        # Prepare template data
        template_data = {
            "portfolio_name": portfolio_name,
            "quarter": quarter,
            "year": year,
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recommendations": recommendations,
            "has_recommendations": len(recommendations.recommendations) > 0,
            "total_recommendations": len(recommendations.recommendations),
            "risk_profile": recommendations.risk_profile,
            "overall_score": recommendations.total_score,
            "confidence_level": recommendations.confidence_level,
            "analysis": recommendations.analysis,
            "warnings": recommendations.warnings,
            "diversification_score": recommendations.diversification_score,
            "market_correlation": recommendations.market_correlation,
        }

        return template.render(template_data)

    def _get_html_template(self) -> str:
        """Get the HTML template for AI recommendations reports."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ portfolio_name }} - AI Investment Recommendations {{ quarter }} {{ year }}</title>
    <style>
        /* Modern CSS styling similar to quarterly reports */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 600;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .summary-card {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
            transition: transform 0.2s ease;
        }

        .summary-card:hover {
            transform: translateY(-2px);
        }

        .summary-card h3 {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .summary-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: #333;
        }

        .section {
            background: white;
            margin-bottom: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .section-header {
            background: #f8f9fa;
            padding: 20px 30px;
            border-bottom: 1px solid #e9ecef;
        }

        .section-header h2 {
            font-size: 1.5rem;
            font-weight: 600;
            color: #333;
        }

        .section-content {
            padding: 30px;
        }

        .recommendation-item {
            background: #f8f9fa;
            margin: 20px 0;
            padding: 25px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }

        .recommendation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .asset-symbol {
            font-size: 1.3rem;
            font-weight: 700;
            color: #333;
        }

        .strategy-badge {
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 500;
        }

        .recommendation-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .metric {
            text-align: center;
        }

        .metric-value {
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .reasoning {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 3px solid #17a2b8;
        }

        .red-flags {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }

        .red-flags h4 {
            margin-bottom: 10px;
            color: #721c24;
        }

        .red-flags ul {
            margin-left: 20px;
        }

        .analysis-text {
            font-size: 1.1rem;
            line-height: 1.8;
            color: #555;
        }

        .no-recommendations {
            text-align: center;
            padding: 60px;
            color: #666;
        }

        .no-recommendations h3 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #999;
        }

        .warning-banner {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .warning-banner h4 {
            margin-bottom: 10px;
            color: #856404;
        }

        .score-high { color: #28a745; }
        .score-medium { color: #ffc107; }
        .score-low { color: #dc3545; }

        .confidence-high { border-left-color: #28a745; }
        .confidence-medium { border-left-color: #ffc107; }
        .confidence-low { border-left-color: #dc3545; }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            .header {
                padding: 20px;
                text-align: center;
            }

            .header h1 {
                font-size: 2rem;
            }

            .summary-cards {
                grid-template-columns: 1fr;
            }

            .recommendation-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🤖 AI Investment Recommendations</h1>
            <p>{{ portfolio_name }} Portfolio • {{ quarter }} {{ year }} • Generated: {{ generation_time }}</p>
        </div>

        <!-- Summary Cards -->
        <div class="summary-cards">
            <div class="summary-card">
                <h3>Risk Profile</h3>
                <div class="value">{{ risk_profile|title }}</div>
            </div>

            <div class="summary-card">
                <h3>Total Recommendations</h3>
                <div class="value">{{ total_recommendations }}</div>
            </div>

            <div class="summary-card">
                <h3>Overall Score</h3>
                <div class="value score-{% if overall_score >= 0.7 %}high{% elif overall_score >= 0.4 %}medium{% else %}low{% endif %}">
                    {{ "%.1f"|format(overall_score * 100) }}%
                </div>
            </div>

            <div class="summary-card">
                <h3>Confidence Level</h3>
                <div class="value score-{% if confidence_level >= 0.7 %}high{% elif confidence_level >= 0.4 %}medium{% else %}low{% endif %}">
                    {{ "%.1f"|format(confidence_level * 100) }}%
                </div>
            </div>
        </div>

        <!-- Warnings Banner -->
        {% if warnings %}
        <div class="warning-banner">
            <h4>⚠️ Important Warnings</h4>
            <ul>
                {% for warning in warnings %}
                <li>{{ warning }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        <!-- Analysis Section -->
        <div class="section">
            <div class="section-header">
                <h2>📊 AI Analysis</h2>
            </div>
            <div class="section-content">
                <div class="analysis-text">
                    {{ analysis if analysis else "No detailed analysis available for this portfolio." }}
                </div>

                {% if diversification_score or market_correlation %}
                <div class="recommendation-metrics" style="margin-top: 30px;">
                    {% if diversification_score %}
                    <div class="metric">
                        <div class="metric-value score-{% if diversification_score >= 0.7 %}high{% elif diversification_score >= 0.4 %}medium{% else %}low{% endif %}">
                            {{ "%.1f"|format(diversification_score * 100) }}%
                        </div>
                        <div class="metric-label">Diversification</div>
                    </div>
                    {% endif %}

                    {% if market_correlation %}
                    <div class="metric">
                        <div class="metric-value">{{ "%.2f"|format(market_correlation) }}</div>
                        <div class="metric-label">Market Correlation</div>
                    </div>
                    {% endif %}
                </div>
                {% endif %}
            </div>
        </div>

        <!-- Recommendations Section -->
        {% if has_recommendations %}
        <div class="section">
            <div class="section-header">
                <h2>💡 Investment Recommendations</h2>
            </div>
            <div class="section-content">
                {% for recommendation in recommendations.recommendations %}
                <div class="recommendation-item confidence-{% if recommendation.confidence >= 0.7 %}high{% elif recommendation.confidence >= 0.4 %}medium{% else %}low{% endif %}">
                    <div class="recommendation-header">
                        <div class="asset-symbol">{{ recommendation.symbol }}</div>
                        <div class="strategy-badge">{{ recommendation.strategy }}</div>
                    </div>

                    <div class="recommendation-metrics">
                        <div class="metric">
                            <div class="metric-value score-{% if recommendation.score >= 0.7 %}high{% elif recommendation.score >= 0.4 %}medium{% else %}low{% endif %}">
                                {{ "%.1f"|format(recommendation.score * 100) }}%
                            </div>
                            <div class="metric-label">Score</div>
                        </div>

                        <div class="metric">
                            <div class="metric-value">{{ "%.1f"|format(recommendation.confidence * 100) }}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>

                        <div class="metric">
                            <div class="metric-value">{{ "%.1f"|format(recommendation.allocation_percentage) }}%</div>
                            <div class="metric-label">Allocation</div>
                        </div>

                        <div class="metric">
                            <div class="metric-value">{{ recommendation.risk_level|title }}</div>
                            <div class="metric-label">Risk Level</div>
                        </div>

                        <div class="metric">
                            <div class="metric-value">{{ "%.2f"|format(recommendation.sortino_ratio) }}</div>
                            <div class="metric-label">Sortino Ratio</div>
                        </div>

                        <div class="metric">
                            <div class="metric-value">{{ "%.2f"|format(recommendation.calmar_ratio) }}</div>
                            <div class="metric-label">Calmar Ratio</div>
                        </div>
                    </div>

                    {% if recommendation.reasoning %}
                    <div class="reasoning">
                        <strong>💭 AI Reasoning:</strong><br>
                        {{ recommendation.reasoning }}
                    </div>
                    {% endif %}

                    {% if recommendation.red_flags %}
                    <div class="red-flags">
                        <h4>🚨 Red Flags</h4>
                        <ul>
                            {% for flag in recommendation.red_flags %}
                            <li>{{ flag }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </div>
        {% else %}
        <div class="section">
            <div class="section-content">
                <div class="no-recommendations">
                    <h3>🔍 No Suitable Investments Found</h3>
                    <p>Based on the current market conditions and risk criteria, no assets met the investment standards for this {{ risk_profile }} risk profile.</p>
                    <p style="margin-top: 20px; color: #999;">Consider adjusting risk tolerance or exploring different time periods.</p>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Footer -->
        <div class="section">
            <div class="section-content" style="text-align: center; color: #666;">
                <p>Generated by AI Investment Recommendations System</p>
                <p style="font-size: 0.9rem; margin-top: 10px;">
                    This analysis is for informational purposes only and should not be considered as financial advice.
                </p>
            </div>
        </div>
    </div>
</body>
</html>
        """
