"""AI Report Generator for Investment Recommendations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .models import PortfolioRecommendation


class AIReportGenerator:
    """Generates HTML reports for AI investment recommendations."""

    def __init__(self):
        self.output_dir = Path("exports/ai_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(
        self, recommendation: PortfolioRecommendation, portfolio_name: str
    ) -> str:
        """Generate HTML report for AI recommendations."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_recommendations_{portfolio_name}_{timestamp}.html"
        output_path = self.output_dir / filename

        html_content = self._create_html_content(recommendation, portfolio_name)

        output_path.write_text(html_content, encoding="utf-8")
        return str(output_path)

    def _create_html_content(
        self, recommendation: PortfolioRecommendation, portfolio_name: str
    ) -> str:
        """Create HTML content for AI recommendations."""

        asset_rows = ""
        for asset in recommendation.asset_recommendations:
            confidence_color = (
                "#28a745"
                if asset.confidence_score > 0.7
                else "#ffc107"
                if asset.confidence_score > 0.5
                else "#dc3545"
            )

            asset_rows += f"""
            <tr>
                <td>{asset.symbol}</td>
                <td>{asset.strategy}</td>
                <td>{asset.timeframe}</td>
                <td style="color: {confidence_color}; font-weight: bold;">{asset.confidence_score:.3f}</td>
                <td>{asset.sortino_ratio:.3f}</td>
                <td>{asset.sharpe_ratio:.3f}</td>
                <td>{asset.total_return:.2f}%</td>
                <td style="background: {"#ff6b6b" if asset.recommendation_type == "BUY" else "#4ecdc4"}; color: white; font-weight: bold; text-align: center;">
                    {asset.recommendation_type}
                </td>
            </tr>"""

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI Investment Recommendations: {portfolio_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 2.5em; font-weight: 300; }}
        .content {{ padding: 30px; }}
        .summary {{ background: #f8f9fa; border-radius: 8px; padding: 20px; margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #343a40; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        .metric {{ display: inline-block; margin-right: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Investment Recommendations</h1>
            <p>Portfolio: {portfolio_name} • Risk Profile: {recommendation.risk_profile.title()}</p>
        </div>

        <div class="content">
            <div class="summary">
                <h3>Portfolio Overview</h3>
                <div class="metric"><strong>Total Assets:</strong> {recommendation.total_assets}</div>
                <div class="metric"><strong>Expected Return:</strong> {recommendation.expected_return:.2f}%</div>
                <div class="metric"><strong>Confidence:</strong> {recommendation.confidence_score:.3f}</div>
                <p><strong>AI Analysis:</strong> {recommendation.reasoning}</p>
            </div>

            <h3>Asset Recommendations</h3>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Strategy</th>
                        <th>Timeframe</th>
                        <th>Confidence</th>
                        <th>Sortino</th>
                        <th>Sharpe</th>
                        <th>Return</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {asset_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""

        return html_template
