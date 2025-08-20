"""Clean Portfolio Report Generator.

Uses only real data from database and backtesting library.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy.orm import sessionmaker

from src.database.db_connection import get_sync_engine
from src.database.models import BestStrategy, Trade
from utils.report_organizer import ReportOrganizer


class DetailedPortfolioReporter:
    """Generates detailed visual reports using only real database data."""

    def __init__(self):
        self.report_organizer = ReportOrganizer()

    def generate_comprehensive_report(
        self,
        portfolio_config: dict,
        start_date: str,
        end_date: str,
        strategies: list[str],
        timeframes: list[str] | None = None,
    ) -> str:
        """Generate a comprehensive HTML report using real database data."""

        if timeframes is None:
            timeframes = ["1d"]

        # Get real data for each asset from database
        assets_data = {}
        for symbol in portfolio_config["symbols"]:
            asset_data = self._get_real_asset_data(symbol)
            if asset_data:
                assets_data[symbol] = asset_data

        # Generate HTML report with real data
        html_content = self._create_html_report(
            portfolio_config, assets_data, start_date, end_date
        )

        # Save report
        return self._save_compressed_report(html_content, portfolio_config["name"])

    def _get_real_asset_data(self, symbol: str) -> dict | None:
        """Get real asset data from database."""
        engine = get_sync_engine()
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Get best strategy for this symbol
            best_strategy = (
                session.query(BestStrategy)
                .filter_by(symbol=symbol)
                .order_by(BestStrategy.sortino_ratio.desc())
                .first()
            )

            if not best_strategy:
                return None

            # Get trades for this strategy
            trades = (
                session.query(Trade)
                .filter_by(
                    symbol=symbol,
                    strategy=best_strategy.strategy,
                    timeframe=best_strategy.timeframe,
                )
                .order_by(Trade.trade_datetime)
                .all()
            )

            # Convert trades to orders format
            orders = []
            for trade in trades:
                orders.append(
                    {
                        "date": trade.trade_datetime.strftime("%Y-%m-%d"),
                        "type": trade.side,
                        "price": float(trade.price),
                        "size": float(trade.size),
                        "equity": float(trade.equity_after or 0),
                    }
                )

            return {
                "best_strategy": best_strategy.strategy,
                "best_timeframe": best_strategy.timeframe,
                "best_score": float(best_strategy.sortino_ratio or 0),
                "data": {
                    "overview": {
                        "PSR": float(best_strategy.sortino_ratio or 0),
                        "sharpe_ratio": float(best_strategy.sharpe_ratio or 0),
                        "total_orders": len(orders),
                        "net_profit": float(best_strategy.total_return or 0),
                        "max_drawdown": abs(float(best_strategy.max_drawdown or 0)),
                        "calmar_ratio": float(best_strategy.calmar_ratio or 0),
                        "best_timeframe": best_strategy.timeframe,
                        # Calculated values
                        "average_win": 0,
                        "average_loss": 0,
                        "win_rate": 0,
                    },
                    "orders": orders,
                    "equity_curve": self._generate_simple_equity_curve(orders),
                    "benchmark_curve": [],
                },
            }

        finally:
            session.close()

    def _generate_simple_equity_curve(self, orders: list) -> list:
        """Generate simple equity curve from real trades."""
        if not orders:
            return []

        equity_curve = []

        for order in orders:
            equity_curve.append({"date": order["date"], "equity": order["equity"]})

        return equity_curve

    def _generate_backtest_plot(self, symbol: str, strategy: str, timeframe: str, start_date: str, end_date: str) -> str | None:
        """Generate interactive plot from backtesting library."""
        try:
            from src.core.direct_backtest import run_direct_backtest
            from bokeh.io import save
            from bokeh.resources import CDN
            from bokeh.embed import file_html
            import tempfile
            
            # Run fresh backtest to get plot
            result = run_direct_backtest(
                symbol=symbol,
                strategy_name=strategy,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            if hasattr(result, 'backtest_instance'):
                # Generate the plot
                plot = result.backtest_instance.plot()
                
                # Convert to HTML string
                html = file_html(plot, CDN, f"{symbol} - {strategy} ({timeframe})")
                return html
                
        except Exception as e:
            print(f"Warning: Could not generate plot for {symbol}/{strategy}: {e}")
            
        return None

    def _create_html_report(
        self, portfolio_config: dict, assets_data: dict, start_date: str, end_date: str
    ) -> str:
        """Create HTML report using real data."""

        html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Portfolio Analysis: {portfolio_name}</title>
    <style>
        body {{font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
               margin: 0; padding: 20px; background: #f5f5f5;}}
        .container {{max-width: 1400px; margin: 0 auto; background: white;
                    border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    overflow: hidden;}}
        .header {{background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 color: white; padding: 30px; text-align: center;}}
        .header h1 {{margin: 0; font-size: 2.5em; font-weight: 300;}}
        .header p {{margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em;}}
        .asset-section {{border-bottom: 1px solid #eee; padding: 30px;}}
        .asset-section:last-child {{border-bottom: none;}}
        .asset-header {{display: flex; justify-content: space-between;
                       align-items: center; margin-bottom: 30px;}}
        .asset-title {{font-size: 1.8em; color: #333; margin: 0;}}
        .strategy-badge {{background: #4CAF50; color: white; padding: 8px 16px;
                         border-radius: 20px; font-size: 0.9em;}}
        .metrics-grid {{display: grid;
                       grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                       gap: 15px; margin-bottom: 30px;}}
        .metric-card {{background: #f8f9fa; border-radius: 8px; padding: 15px;
                      text-align: center; border-left: 4px solid #007bff;}}
        .metric-label {{font-size: 0.85em; color: #666; margin-bottom: 5px;
                       text-transform: uppercase; letter-spacing: 1px;}}
        .metric-value {{font-size: 1.4em; font-weight: bold; color: #333;}}
        .positive {{color: #28a745;}}
        .negative {{color: #dc3545;}}
        .orders-table {{width: 100%; border-collapse: collapse; margin-top: 20px;}}
        .orders-table th {{background: #343a40; color: white; padding: 12px; text-align: left;}}
        .orders-table td {{padding: 10px; border-bottom: 1px solid #eee;}}
        .buy {{color: #28a745; font-weight: bold;}}
        .sell {{color: #dc3545; font-weight: bold;}}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{portfolio_name}</h1>
            <p>Real Backtesting Data • {start_date} to {end_date}</p>
        </div>
        {asset_sections}
    </div>
</body>
</html>"""

        # Generate asset sections using real data
        asset_sections = ""
        for symbol, data in assets_data.items():
            overview = data["data"]["overview"]
            orders = data["data"]["orders"]

            # Create orders table
            orders_html = ""
            if orders:
                orders_html = """
                <table class="orders-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Type</th>
                            <th>Price</th>
                            <th>Size</th>
                            <th>Equity After</th>
                        </tr>
                    </thead>
                    <tbody>"""

                for order in orders:
                    orders_html += f'''
                        <tr>
                            <td>{order["date"]}</td>
                            <td class="{order["type"].lower()}">{order["type"]}</td>
                            <td>${order["price"]:.2f}</td>
                            <td>{order["size"]:.0f}</td>
                            <td>${order["equity"]:,.2f}</td>
                        </tr>'''

                orders_html += """
                    </tbody>
                </table>"""
            else:
                orders_html = "<p>No trades recorded for this strategy.</p>"

            asset_sections += f"""
        <div class="asset-section">
            <div class="asset-header">
                <h2 class="asset-title">{symbol}</h2>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <span class="strategy-badge">Best: {data["best_strategy"].title()}</span>
                    <span class="strategy-badge" style="background: #FF6B6B;">⏰ {data["best_timeframe"]}</span>
                </div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Sortino Ratio</div>
                    <div class="metric-value positive">{overview["PSR"]:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{overview["sharpe_ratio"]:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Orders</div>
                    <div class="metric-value">{overview["total_orders"]}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Net Profit</div>
                    <div class="metric-value {"positive" if overview["net_profit"] > 0 else "negative"}">{overview["net_profit"]:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">-{overview["max_drawdown"]:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Calmar Ratio</div>
                    <div class="metric-value">{overview.get("calmar_ratio", 0):.3f}</div>
                </div>
            </div>
            
            <h3>Interactive Chart</h3>
            <div class="plot-container">
                {self._generate_backtest_plot(symbol, data["best_strategy"], data["best_timeframe"], start_date, end_date) or "<p>Plot generation failed</p>"}
            </div>

            <h3>Trading Orders</h3>
            {orders_html}
        </div>"""

        return html_template.format(
            portfolio_name=portfolio_config["name"],
            start_date=start_date,
            end_date=end_date,
            asset_sections=asset_sections,
        )

    def _save_compressed_report(self, html_content: str, portfolio_name: str) -> str:
        """Save compressed HTML report."""
        # Use the existing report organizer
        temp_path = Path("temp_report.html")
        temp_path.write_text(html_content, encoding="utf-8")

        try:
            organized_path = self.report_organizer.organize_report(
                str(temp_path), portfolio_name
            )
            return organized_path
        finally:
            if temp_path.exists():
                temp_path.unlink()
