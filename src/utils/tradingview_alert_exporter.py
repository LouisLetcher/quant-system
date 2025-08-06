#!/usr/bin/env python3
"""
TradingView Alert Exporter

Extracts asset strategies and timeframes from HTML reports and generates
TradingView alert messages with appropriate placeholders.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup


class TradingViewAlertExporter:
    def __init__(self, reports_dir: str = "exports/reports"):
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

#QuantTrading #{symbol} #{strategy.replace(' ', '')}"""

        return alert_message

    def process_html_file(self, file_path: Path) -> List[Dict]:
        """Process single HTML file and extract asset data"""
        try:
            with open(file_path, encoding="utf-8") as f:
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

    def export_alerts(self, output_file: str = None) -> Dict:
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

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# TradingView Alert Messages\n\n")

                for symbol, alerts in all_alerts.items():
                    f.write(f"## {symbol}\n\n")

                    for i, alert_data in enumerate(alerts):
                        asset = alert_data["asset_data"]
                        f.write(
                            f"### Alert {i+1} - {asset['strategy']} ({asset['timeframe']})\n"
                        )
                        f.write(f"**Source:** {asset['source_file']}\n\n")
                        f.write("```\n")
                        f.write(alert_data["alert_message"])
                        f.write("\n```\n\n")
                        f.write("---\n\n")

        return all_alerts


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
