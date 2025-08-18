#!/usr/bin/env python3
"""
TradingView Alert Exporter

Extracts asset strategies and timeframes from PostgreSQL database and generates
TradingView alert messages organized by portfolio results.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from ..database.models import BacktestResult


class TradingViewAlertExporter:
    def __init__(
        self, db_session: Session = None, reports_dir: str = "exports/reports"
    ):
        self.db_session = db_session
        self.reports_dir = Path(reports_dir)

        # Check if old location exists and new location is empty for migration
        old_dir = Path("reports_output")
        if old_dir.exists() and not self.reports_dir.exists():
            print(f"⚠️  Found reports in old location: {old_dir}")
            print(
                f"💡 Consider running report organizer to migrate to: {self.reports_dir}"
            )
            self.reports_dir = old_dir

    def get_quarter_from_date(self, date: datetime) -> tuple[int, int]:
        """Get quarter and year from date."""
        quarter = (date.month - 1) // 3 + 1
        return date.year, quarter

    def organize_output_path(self, base_dir: str) -> Path:
        """Create organized output path based on current quarter/year."""
        now = datetime.now(timezone.utc)
        year, quarter = self.get_quarter_from_date(now)

        output_dir = Path(base_dir) / str(year) / f"Q{quarter}"
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    def export_from_database(
        self,
        quarter: str = None,
        year: str = None,
        output_dir: str = "exports/tradingview_alerts",
    ) -> List[str]:
        """
        Export TradingView alerts from PostgreSQL database.
        Creates markdown files organized by year/quarter matching reports structure.

        Args:
            quarter: Quarter filter (Q1, Q2, Q3, Q4)
            year: Year filter (YYYY)
            output_dir: Output directory for alert files

        Returns:
            List of generated file paths
        """
        if not self.db_session:
            print("No database session - falling back to HTML parsing")
            return self.export_alerts_from_reports(output_dir)

        # Use new query helper for consistent data access
        from ..database.query_helpers import DatabaseQueryHelper

        query_helper = DatabaseQueryHelper(self.db_session)

        # Get best strategies data with minimum performance threshold
        strategies_data = query_helper.get_best_strategies(
            quarter=quarter,
            year=year,
            min_sortino=0.001,  # Only positive strategies, no matter how small
        )

        if not strategies_data:
            print("No positive strategies found for TradingView alerts")
            return []

        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"Found {len(strategies_data)} positive strategies for alerts")
        for strategy in strategies_data:
            logger.debug(
                f"  {strategy['symbol']}/{strategy['strategy']} - Sortino: {strategy['sortino_ratio']:.4f}"
            )

        # Create year/quarter directory structure like reports
        now = datetime.now()
        current_year = year or str(now.year)
        current_quarter = quarter or f"Q{(now.month - 1) // 3 + 1}"

        output_path = Path(output_dir) / current_year / current_quarter
        output_path.mkdir(parents=True, exist_ok=True)

        # Group results by portfolio type (like reports)
        portfolio_groups = self._group_by_portfolio_type(strategies_data)
        generated_files = []

        for portfolio_type, portfolio_results in portfolio_groups.items():
            # Create markdown file matching report naming convention
            filename = (
                f"{portfolio_type}_Portfolio_{current_quarter}_{current_year}_Alerts.md"
            )
            file_path = output_path / filename

            with open(file_path, "w") as f:
                f.write(f"# TradingView Alerts - {portfolio_type} Portfolio\n\n")
                f.write(f"**Generated:** {datetime.now().isoformat()}\n")
                f.write(f"**Quarter:** {current_quarter} {current_year}\n")
                f.write(f"**Total Strategies:** {len(portfolio_results)}\n\n")
                f.write("---\n\n")

                for strategy in portfolio_results:
                    symbol = strategy["symbol"]
                    tv_symbol = symbol.replace("=X", "")

                    # Extract metrics
                    sortino = strategy["sortino_ratio"]
                    total_return = strategy["total_return"]
                    max_drawdown = strategy["max_drawdown"]
                    strategy_name = strategy["strategy"]

                    f.write(f"## {symbol} - {strategy_name}\n\n")
                    f.write(
                        f"**Performance:** Sortino: {sortino:.2f} | Return: {total_return:.2f}% | MaxDD: {max_drawdown:.2f}%\n\n"
                    )
                    f.write("### Alert Setup\n\n")
                    f.write(f"**Condition:** `{tv_symbol}` crosses above SMA(20)\n\n")
                    f.write("**Alert Message:**\n```\n")
                    f.write(f"BUY {tv_symbol} - {strategy_name}\n")
                    f.write("Entry: {{close}}\n")
                    f.write("Target: {{close * 1.05}}\n")
                    f.write("Stop: {{close * 0.95}}\n")
                    f.write(f"Sortino: {sortino:.2f}\n")
                    f.write(f"Strategy: {strategy_name}\n")
                    f.write("```\n\n")
                    f.write("---\n\n")

            generated_files.append(str(file_path))

        return generated_files

    def _group_by_portfolio_type(self, strategies_data: list[dict]) -> dict[str, list]:
        """Group strategies by portfolio type based on symbol patterns."""
        groups = {"Forex": [], "Crypto": [], "Stocks": [], "Bonds": [], "Other": []}

        for strategy in strategies_data:
            symbol = strategy["symbol"]

            if "=X" in symbol or "USD" in symbol:
                groups["Forex"].append(strategy)
            elif "USDT" in symbol or "BTC" in symbol or "ETH" in symbol:
                groups["Crypto"].append(strategy)
            elif any(
                bond in symbol
                for bond in ["TLT", "IEF", "SHY", "LQD", "HYG", "EMB", "AGG", "BND"]
            ):
                groups["Bonds"].append(strategy)
            elif len(symbol) <= 5 and symbol.isalpha():
                groups["Stocks"].append(strategy)
            else:
                groups["Other"].append(strategy)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _create_alert_file(
        self, results: List[BacktestResult], output_path: Path, risk_level: str
    ) -> str:
        """Create TradingView alert file for specific risk level."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tradingview_alerts_{risk_level}_{timestamp}.txt"
        file_path = output_path / filename

        with open(file_path, "w") as f:
            f.write(f"# TradingView Alerts - {risk_level.title()} Risk Level\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total Strategies: {len(results)}\n\n")

            for strategy in results:
                symbol = strategy["symbol"]

                # Clean symbol for TradingView (remove =X suffix for forex)
                tv_symbol = symbol.replace("=X", "")

                # Extract metrics
                sortino = strategy["sortino_ratio"]
                total_return = strategy["total_return"]
                max_drawdown = strategy["max_drawdown"]
                strategy_name = strategy["strategy"]

                alert_message = f"""
# {symbol} - {strategy_name}
# Sortino: {sortino:.2f} | Return: {total_return:.2f}% | MaxDD: {max_drawdown:.2f}%

Alert Condition: {tv_symbol} crosses above SMA(20)
Message:
BUY {tv_symbol} - {strategy_name}
Entry: {{{{close}}}}
Target: {{{{close * 1.05}}}}
Stop: {{{{close * 0.95}}}}
Sortino: {sortino:.2f}
Strategy: {strategy_name}

---
"""
                f.write(alert_message)

        return str(file_path)

    def _get_quarter_dates(self, quarter: str, year: str) -> tuple[datetime, datetime]:
        """Get start and end dates for a quarter."""
        if not quarter or len(quarter) < 2 or not quarter.startswith("Q"):
            raise ValueError(
                f"Invalid quarter format: '{quarter}'. Expected format: Q1, Q2, Q3, or Q4"
            )
        quarter_num = int(quarter[1])  # Extract number from Q1, Q2, etc.
        start_month = (quarter_num - 1) * 3 + 1

        start_date = datetime(int(year), start_month, 1)

        if quarter_num == 4:
            end_date = datetime(int(year) + 1, 1, 1)
        else:
            end_month = quarter_num * 3
            end_date = datetime(int(year), end_month + 1, 1)

        return start_date, end_date

    def extract_asset_data(self, html_content: str) -> List[Dict]:
        """Extract asset information from HTML report"""
        soup = BeautifulSoup(html_content, "html.parser")
        assets = []

        # Find all asset sections
        asset_sections = soup.find_all("div", class_="asset-section")

        for section in asset_sections:
            # Extract asset symbol from title
            asset_title = section.find("h2", class_="asset-title")
            if not asset_title:
                continue

            symbol = asset_title.text.strip()

            # Extract best strategy from strategy badge
            strategy_badges = section.find_all("span", class_="strategy-badge")
            best_strategy = None
            timeframe = None

            for badge in strategy_badges:
                text = badge.text.strip()
                if text.startswith("Best:"):
                    best_strategy = text.replace("Best:", "").strip()
                elif "⏰" in text:
                    timeframe = text.replace("⏰", "").strip()

            # Extract metrics for additional context
            metrics = {}
            metric_cards = section.find_all("div", class_="metric-card")
            for card in metric_cards:
                label_elem = card.find("div", class_="metric-label")
                value_elem = card.find("div", class_="metric-value")
                if label_elem and value_elem:
                    label = label_elem.text.strip()
                    value = value_elem.text.strip()
                    metrics[label] = value

            if symbol and best_strategy and timeframe:
                assets.append(
                    {
                        "symbol": symbol,
                        "strategy": best_strategy,
                        "timeframe": timeframe,
                        "metrics": metrics,
                    }
                )

        return assets

    def generate_tradingview_alert(self, asset_data: Dict) -> str:
        """Generate TradingView alert message for asset"""
        symbol = asset_data["symbol"]
        strategy = asset_data["strategy"]
        timeframe = asset_data["timeframe"]
        metrics = asset_data.get("metrics", {})

        # Get key metrics for context
        sharpe_ratio = metrics.get("Sharpe Ratio", "N/A")
        net_profit = metrics.get("Net Profit", "N/A")
        win_rate = metrics.get("Win Rate", "N/A")

        alert_message = f"""🚨 QUANT SIGNAL: {symbol} 📊
Strategy: {strategy}
Timeframe: {timeframe}
📈 Sharpe: {sharpe_ratio}
💰 Profit: {net_profit}
🎯 Win Rate: {win_rate}

Price: {{{{close}}}}
Time: {{{{timenow}}}}
Action: {{{{strategy.order.action}}}}
Qty: {{{{strategy.order.contracts}}}}

#QuantTrading #{symbol} #{strategy.replace(" ", "")}"""

        return alert_message

    def process_html_file(self, file_path: Path) -> List[Dict]:
        """Process single HTML file and extract asset data"""
        try:
            with file_path.open(encoding="utf-8") as f:
                content = f.read()

            assets = self.extract_asset_data(content)

            # Add file metadata
            for asset in assets:
                asset["source_file"] = str(file_path)
                asset["report_name"] = file_path.stem

            return assets
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def find_html_reports(self) -> List[Path]:
        """Find all HTML report files"""
        html_files = []
        for root, dirs, files in os.walk(self.reports_dir):
            for file in files:
                if file.endswith(".html"):
                    html_files.append(Path(root) / file)
        return html_files

    def export_alerts_from_reports(self, output_file: str | None = None) -> List[str]:
        """Export all TradingView alerts"""
        html_files = self.find_html_reports()
        all_alerts = {}

        for html_file in html_files:
            print(f"Processing: {html_file}")
            assets = self.process_html_file(html_file)

            for asset in assets:
                symbol = asset["symbol"]
                alert = self.generate_tradingview_alert(asset)

                if symbol not in all_alerts:
                    all_alerts[symbol] = []

                all_alerts[symbol].append({"alert_message": alert, "asset_data": asset})

        # Write to file if specified
        if output_file:
            # Check if output_file is just a filename or has path
            output_path = Path(output_file)
            if output_path.parent == Path():
                # Just filename provided, organize by quarter/year
                organized_dir = self.organize_output_path("exports/tradingview_alerts")
                output_path = organized_dir / output_file
            else:
                # Full path provided, create parent directories
                output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as f:
                f.write("# TradingView Alert Messages\n\n")

                for symbol, alerts in all_alerts.items():
                    f.write(f"## {symbol}\n\n")

                    for i, alert_data in enumerate(alerts):
                        asset = alert_data["asset_data"]
                        f.write(
                            f"### Alert {i + 1} - {asset['strategy']} ({asset['timeframe']})\n"
                        )
                        f.write(f"**Source:** {asset['source_file']}\n\n")
                        f.write("```\n")
                        f.write(alert_data["alert_message"])
                        f.write("\n```\n\n")
                        f.write("---\n\n")

        return [str(output_path)] if "output_path" in locals() and output_path else []


def main():
    parser = argparse.ArgumentParser(
        description="Export TradingView alerts from HTML reports"
    )
    parser.add_argument(
        "--reports-dir",
        default="exports/reports",
        help="Directory containing HTML reports",
    )
    parser.add_argument(
        "--output",
        default="tradingview_alerts.md",
        help="Output file for alerts (auto-organized by quarter/year if just filename)",
    )
    parser.add_argument("--symbol", help="Export alerts for specific symbol only")

    args = parser.parse_args()

    exporter = TradingViewAlertExporter(args.reports_dir)
    alerts = exporter.export_alerts(args.output)

    print("\n📊 Export Summary:")
    print(f"Found {len(alerts)} assets with alerts")

    if args.symbol:
        if args.symbol in alerts:
            print(f"\n🎯 Alerts for {args.symbol}:")
            for alert_data in alerts[args.symbol]:
                print("\n" + "=" * 60)
                print(alert_data["alert_message"])
        else:
            print(f"❌ No alerts found for {args.symbol}")
    else:
        for symbol, symbol_alerts in alerts.items():
            print(f"  {symbol}: {len(symbol_alerts)} alert(s)")

    if args.output:
        print(f"\n✅ Alerts exported to: {args.output}")


if __name__ == "__main__":
    main()
