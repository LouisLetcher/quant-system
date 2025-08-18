"""Detailed Portfolio Report Generator.

Creates comprehensive visual reports for portfolio analysis with KPIs, orders, and charts.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.report_organizer import ReportOrganizer


class DetailedPortfolioReporter:
    """Generates detailed visual reports for portfolio analysis."""

    def __init__(self):
        self.report_data = {}
        self.report_organizer = ReportOrganizer()
        self.rng = np.random.default_rng()

        import logging

        self.logger = logging.getLogger(__name__)

    def generate_comprehensive_report(
        self,
        portfolio_config: dict,
        start_date: str,
        end_date: str,
        strategies: list[str],
        timeframes: list[str] | None = None,
        metric: str = "sortino_ratio",
    ) -> str:
        """Generate a comprehensive HTML report for the portfolio."""

        if timeframes is None:
            timeframes = ["1d"]

        # Generate data for each asset
        assets_data = {}
        for symbol in portfolio_config["symbols"]:
            best_combo, asset_data = self._analyze_asset_with_timeframes(
                symbol, strategies, timeframes, start_date, end_date, metric
            )
            assets_data[symbol] = {
                "best_strategy": best_combo["strategy"],
                "best_timeframe": best_combo["timeframe"],
                "best_score": best_combo["score"],
                "data": asset_data,
            }

        # Generate HTML report
        html_content = self._create_html_report(
            portfolio_config, assets_data, start_date, end_date, metric
        )

        # Compress and save
        return self._save_compressed_report(html_content, portfolio_config["name"])

    def _analyze_asset_with_timeframes(
        self,
        symbol: str,
        strategies: list[str],
        timeframes: list[str],
        start_date: str,
        end_date: str,
        metric: str = "sortino_ratio",
    ) -> tuple[dict, dict]:
        """Analyze an asset using best strategy from database."""
        from ..database import get_db_session
        from ..database.models import BestStrategy

        # Get best strategy from database for this symbol
        session = get_db_session()
        try:
            best_strategy_record = (
                session.query(BestStrategy)
                .filter_by(
                    symbol=symbol,
                    timeframe="1d",  # Default timeframe used in backtests
                )
                .first()
            )

            if not best_strategy_record:
                print(
                    f"WARNING: No BestStrategy record found for {symbol} - using fallback"
                )
                # Fallback to original logic if not found in database
                return self._analyze_asset_with_timeframes_fallback(
                    symbol, strategies, timeframes, start_date, end_date
                )
            self.logger.debug(
                f"Found BestStrategy for {symbol}: {best_strategy_record.best_strategy} with sortino {best_strategy_record.sortino_ratio:.4f}"
            )
            self.logger.debug(
                f"Advanced metrics - avg_win: {best_strategy_record.average_win}, beta: {best_strategy_record.beta}, total_fees: {best_strategy_record.total_fees}"
            )

            # Check if this is a meaningful strategy (has actual trades)
            num_trades = best_strategy_record.num_trades or 0
            self.logger.debug(
                f"{symbol} num_trades check: {num_trades} (type: {type(num_trades)})"
            )
            if num_trades == 0:
                print(
                    f"WARNING: Best strategy for {symbol} has 0 trades - no viable strategy found"
                )

            # Use best strategy from database
            self.logger.debug(
                f"Creating best_combination for {symbol} with strategy {best_strategy_record.best_strategy}"
            )
            self.logger.debug(
                f"Advanced metrics from DB - avg_win: {best_strategy_record.average_win}, alpha: {best_strategy_record.alpha}, beta: {best_strategy_record.beta}"
            )
            best_combination = {
                "strategy": best_strategy_record.best_strategy,
                "timeframe": best_strategy_record.timeframe,
                "score": best_strategy_record.sortino_ratio or 0,
                "metrics": {
                    "sharpe_ratio": best_strategy_record.sharpe_ratio or 0,
                    "sortino_ratio": best_strategy_record.sortino_ratio or 0,
                    "total_return": best_strategy_record.total_return or 0,
                    "max_drawdown": best_strategy_record.max_drawdown or 0,
                    "volatility": best_strategy_record.volatility or 0,
                    "win_rate": best_strategy_record.win_rate or 0,
                    "profit_factor": best_strategy_record.profit_factor or 0,
                    "calmar_ratio": best_strategy_record.calmar_ratio or 0,
                    "num_trades": best_strategy_record.num_trades or 0,
                    "alpha": best_strategy_record.alpha or 0,
                    "beta": best_strategy_record.beta or 0,
                    "expectancy": best_strategy_record.expectancy or 0,
                    "average_win": best_strategy_record.average_win or 0,
                    "average_loss": best_strategy_record.average_loss or 0,
                    "total_fees": best_strategy_record.total_fees or 0,
                    "portfolio_turnover": best_strategy_record.portfolio_turnover or 0,
                    "strategy_capacity": best_strategy_record.strategy_capacity
                    or 10000,
                },
            }

            # Get all strategy results from database for timeframe analysis
            all_combinations = self._get_all_strategy_results_from_db(symbol, metric)

            # Generate detailed data for best combination
            asset_data = self._generate_detailed_metrics_with_timeframe(
                symbol,
                best_combination["strategy"],
                best_combination["timeframe"],
                start_date,
                end_date,
                all_combinations,  # Include all combinations for timeframe analysis
                best_combination["metrics"],  # Pass real metrics
            )
        finally:
            session.close()

        return best_combination, asset_data

    def _get_all_strategy_results_from_db(
        self, symbol: str, metric: str = "sortino_ratio"
    ) -> list[dict]:
        """Get the best strategy for this symbol from BestStrategy table."""
        from ..database import get_db_session
        from ..database.models import BestStrategy

        session = get_db_session()
        try:
            # Get the best strategy for this symbol
            best_strategy = (
                session.query(BestStrategy)
                .filter(BestStrategy.symbol == symbol)
                .first()
            )

            if not best_strategy:
                return []

            # Return only the best strategy result
            combination = {
                "strategy": best_strategy.best_strategy,
                "timeframe": best_strategy.timeframe or "1d",
                "score": float(getattr(best_strategy, metric, 0) or 0),
                "metrics": {
                    "sortino_ratio": float(best_strategy.sortino_ratio or 0),
                    "sharpe_ratio": float(best_strategy.sharpe_ratio or 0),
                    "total_return": float(best_strategy.total_return or 0),
                    "max_drawdown": float(best_strategy.max_drawdown or 0),
                    "volatility": float(best_strategy.volatility or 0),
                    "win_rate": float(best_strategy.win_rate or 0),
                    "profit_factor": float(best_strategy.profit_factor or 0),
                    "calmar_ratio": float(best_strategy.calmar_ratio or 0),
                    "num_trades": int(best_strategy.num_trades or 0),
                    "alpha": float(best_strategy.alpha or 0),
                    "beta": float(best_strategy.beta or 1),
                    "expectancy": float(best_strategy.expectancy or 0),
                    "average_win": float(best_strategy.average_win or 0),
                    "average_loss": float(best_strategy.average_loss or 0),
                    "total_fees": float(best_strategy.total_fees or 0),
                    "portfolio_turnover": float(best_strategy.portfolio_turnover or 0),
                    "strategy_capacity": float(
                        best_strategy.strategy_capacity or 1000000
                    ),
                },
            }

            return [combination]  # Return as list with single best strategy

        finally:
            session.close()

    def _analyze_asset_with_timeframes_fallback(
        self,
        symbol: str,
        strategies: list[str],
        timeframes: list[str],
        start_date: str,
        end_date: str,
    ) -> tuple[dict, dict]:
        """Fallback to original logic when database doesn't have best strategy."""
        best_combination = None
        best_score = -999999
        all_combinations = []

        # Test all strategy + timeframe combinations
        for strategy in strategies:
            for timeframe in timeframes:
                combo_score = self._simulate_strategy_timeframe_performance(
                    symbol, strategy, timeframe
                )

                combination = {
                    "strategy": strategy,
                    "timeframe": timeframe,
                    "score": combo_score["sharpe_ratio"],
                    "metrics": combo_score,
                }
                all_combinations.append(combination)

                # Track best combination
                if combo_score["sharpe_ratio"] > best_score:
                    best_score = combo_score["sharpe_ratio"]
                    best_combination = combination

        # Generate detailed data for best combination
        asset_data = self._generate_detailed_metrics_with_timeframe(
            symbol,
            best_combination["strategy"],
            best_combination["timeframe"],
            start_date,
            end_date,
            all_combinations,
        )
        return best_combination, asset_data

    def _analyze_asset(
        self, symbol: str, strategies: list[str], start_date: str, end_date: str
    ) -> tuple[str, dict]:
        """Analyze an asset and return the best strategy with detailed metrics."""

        # Simulate strategy comparison (replace with actual backtesting when fixed)
        strategy_scores = {}
        for strategy in strategies:
            score = self._simulate_strategy_performance(symbol, strategy)
            strategy_scores[strategy] = score

        # Get best strategy
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1]["sharpe_ratio"])

        # Generate detailed data for best strategy
        asset_data = self._generate_detailed_metrics(
            symbol, best_strategy[0], start_date, end_date
        )

        return best_strategy[0], asset_data

    def _simulate_strategy_timeframe_performance(
        self, symbol: str, strategy: str, timeframe: str
    ) -> dict:
        """Get real performance metrics from database for symbol/strategy/timeframe."""
        # Use database metrics only - no simulation
        return self._get_real_metrics_from_db(symbol, strategy, timeframe)

    def _simulate_strategy_performance(self, symbol: str, strategy: str) -> dict:
        """Get real performance metrics from database for symbol/strategy."""
        # Use database metrics only - no simulation
        return self._get_real_metrics_from_db(symbol, strategy, "1d")

    def _generate_detailed_metrics(
        self,
        symbol: str,
        strategy: str,
        start_date: str,
        end_date: str,
        real_metrics: dict = None,
    ) -> dict:
        """Generate detailed metrics for an asset/strategy combination using real data."""
        self.logger.debug(
            f"_generate_detailed_metrics called for {symbol}/{strategy}, real_metrics={'provided' if real_metrics else 'None'}"
        )

        # Use real metrics if provided, otherwise fall back to database query
        if real_metrics is None:
            # Query database for this specific symbol/strategy combination
            from src.database.db_connection import DatabaseManager
            from src.database.models import BacktestResult

            db_manager = DatabaseManager()
            session = db_manager.get_sync_session()

            try:
                result = (
                    session.query(BacktestResult)
                    .filter(
                        BacktestResult.name.like(f"%{symbol}%"),
                        BacktestResult.strategy.ilike(strategy.lower()),
                    )
                    .first()
                )

                self.logger.debug(
                    f"Database query for {symbol}/{strategy}: {'Found' if result else 'Not found'}"
                )
                if result:
                    self.logger.debug(
                        f"Result sortino={result.sortino_ratio}, return={result.total_return}"
                    )
                    real_metrics = {
                        "sortino_ratio": float(result.sortino_ratio or 0),
                        "sharpe_ratio": float(result.sharpe_ratio or 0),
                        "total_return": float(result.total_return or 0),
                        "max_drawdown": float(result.max_drawdown or 0),
                        "volatility": float(result.volatility or 0),
                        "win_rate": float(result.win_rate or 0),
                        "profit_factor": float(result.profit_factor or 0),
                        "calmar_ratio": float(result.calmar_ratio or 0),
                        "num_trades": result.num_trades or 0,
                        "alpha": float(result.alpha or 0),
                        "beta": float(result.beta or 0),
                        "expectancy": float(result.expectancy or 0),
                        "average_win": float(result.average_win or 0),
                        "average_loss": float(result.average_loss or 0),
                        "total_fees": float(result.total_fees or 0),
                        "portfolio_turnover": float(result.portfolio_turnover or 0),
                        "strategy_capacity": float(result.strategy_capacity or 10000),
                    }
                else:
                    # Fallback to zero metrics if no data found
                    real_metrics = {
                        "sortino_ratio": 0,
                        "sharpe_ratio": 0,
                        "total_return": 0,
                        "max_drawdown": 0,
                        "volatility": 0,
                        "win_rate": 0,
                        "profit_factor": 0,
                        "calmar_ratio": 0,
                        "num_trades": 0,
                        "alpha": 0,
                        "beta": 0,
                        "expectancy": 0,
                        "average_win": 0,
                        "average_loss": 0,
                        "total_fees": 0,
                        "portfolio_turnover": 0,
                        "strategy_capacity": 10000,
                    }
            finally:
                session.close()

        # Generate realistic trading data for display
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Get real orders from database and calculate actual final equity from trades
        orders = self._get_real_orders_from_db(symbol, strategy)
        num_orders = len(orders)

        # Calculate actual final equity from order history
        initial_equity = 10000
        actual_final_equity = orders[-1]["equity"] if orders else initial_equity
        actual_total_return = (
            (actual_final_equity - initial_equity) / initial_equity
        ) * 100

        # Always use actual trade data for profit/equity calculations
        self.logger.debug(
            f"{symbol}/{strategy} - DB total_return: {real_metrics['total_return']:.2f}%, Actual from trades: {actual_total_return:.2f}%, Orders: {num_orders}"
        )
        self.logger.debug(
            f"{symbol}/{strategy} - Overview metrics: avg_win={real_metrics.get('average_win', 0)}, beta={real_metrics.get('beta', 0)}, total_fees={real_metrics.get('total_fees', 0)}"
        )

        # Use real metrics in overview
        return {
            "overview": {
                "PSR": real_metrics["sortino_ratio"],
                "sharpe_ratio": real_metrics["sharpe_ratio"],
                "total_orders": num_orders,
                "average_win": real_metrics.get("average_win", 0),
                "average_loss": real_metrics.get("average_loss", 0),
                "compounding_annual_return": actual_total_return,  # Use actual from trades
                "drawdown": real_metrics.get("max_drawdown", 0),
                "expectancy": real_metrics.get("expectancy", 0),
                "start_equity": initial_equity,
                "end_equity": actual_final_equity,  # Use actual from trades
                "net_profit": actual_total_return,  # Use actual from trades
                "sortino_ratio": real_metrics["sortino_ratio"],
                "loss_rate": max(1 - real_metrics.get("win_rate", 50) / 100, 0.0),
                "win_rate": real_metrics.get("win_rate", 0) / 100,
                "profit_loss_ratio": real_metrics.get("profit_factor", 0),
                "alpha": real_metrics.get("alpha", 0),
                "beta": real_metrics.get("beta", 0),
                "annual_std": real_metrics.get("volatility", 0.0),
                "annual_variance": (real_metrics.get("volatility", 0.0) ** 2),
                "information_ratio": real_metrics.get("sharpe_ratio", 0.0),
                "tracking_error": real_metrics.get("volatility", 0.0),
                "treynor_ratio": real_metrics.get("sharpe_ratio", 0.0),
                "total_fees": real_metrics.get("total_fees", 0),
                "strategy_capacity": real_metrics.get("strategy_capacity", 10000),
                "lowest_capacity_asset": f"{symbol}",
                "portfolio_turnover": real_metrics.get("portfolio_turnover", 0),
            },
            "orders": orders,
            "equity_curve": self._generate_equity_curve(
                start,
                end,
                initial_equity,
                actual_final_equity,  # Use actual final equity
            ),
            "benchmark_curve": [],  # No separate benchmark - use BuyAndHold from database
            "symbol": symbol,
            "strategy": strategy,
        }

    def _generate_detailed_metrics_with_timeframe(
        self,
        symbol: str,
        strategy: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        all_combinations: list,
        real_metrics: dict = None,
    ) -> dict:
        """Generate detailed metrics including timeframe analysis."""
        self.logger.debug(
            f"_generate_detailed_metrics_with_timeframe called for {symbol}/{strategy}"
        )
        base_metrics = self._generate_detailed_metrics(
            symbol, strategy, start_date, end_date, real_metrics
        )

        # Add timeframe-specific information
        base_metrics["best_timeframe"] = timeframe
        base_metrics["timeframe_analysis"] = all_combinations

        # Update overview with timeframe info
        base_metrics["overview"]["best_timeframe"] = timeframe
        base_metrics["overview"]["total_combinations_tested"] = len(all_combinations)

        # Calculate timeframe performance ranking
        sorted_combos = sorted(all_combinations, key=lambda x: x["score"], reverse=True)
        best_combo_rank = next(
            (
                i + 1
                for i, combo in enumerate(sorted_combos)
                if combo["strategy"] == strategy and combo["timeframe"] == timeframe
            ),
            1,
        )
        base_metrics["overview"]["combination_rank"] = (
            f"{best_combo_rank}/{len(all_combinations)}"
        )

        return base_metrics

    def _get_real_orders_from_db(self, symbol: str, strategy: str) -> list[dict]:
        """Get real order history from database for symbol/strategy combination."""
        from src.database.db_connection import get_db_session
        from src.database.models import BacktestResult as DBBacktestResult
        from src.database.models import Trade

        session = get_db_session()
        try:
            # Find the backtest result for this symbol/strategy
            backtest_result = (
                session.query(DBBacktestResult)
                .filter(
                    DBBacktestResult.symbols.any(symbol),
                    DBBacktestResult.strategy == strategy,
                )
                .order_by(DBBacktestResult.created_at.desc())
                .first()
            )

            if not backtest_result:
                return []

            # Get trades for this backtest result
            trades = (
                session.query(Trade)
                .filter(
                    Trade.backtest_result_id == backtest_result.id,
                    Trade.symbol == symbol,
                )
                .order_by(Trade.trade_datetime)
                .all()
            )

            # Convert to order format expected by HTML template
            orders = []
            for trade in trades:
                orders.append(
                    {
                        "datetime": trade.trade_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                        "type": trade.trade_type,
                        "price": float(trade.price),
                        "quantity": float(trade.quantity),
                        "equity": float(trade.equity_after_trade),
                        "fees": float(trade.fees),
                        "holdings": float(trade.holdings_after_trade),
                        "net_profit": float(trade.net_profit),
                        "unrealized": float(trade.unrealized_pnl),
                    }
                )
            return orders

        except Exception as e:
            print(f"Error getting real orders from database: {e}")
            import traceback

            traceback.print_exc()
            return []
        finally:
            session.close()

    def _generate_orders(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        num_orders: int,
        initial_equity: float,
    ) -> list[dict]:
        """Generate realistic order data."""
        orders = []
        current_equity = initial_equity
        current_holdings = 0

        for i in range(num_orders):
            # Random date within range
            total_days = (end_date - start_date).days
            random_days = self.rng.integers(0, int(total_days))
            order_date = start_date + timedelta(days=int(random_days))

            # Order details
            order_type = self.rng.choice(
                ["buy", "sell"], p=[0.6, 0.4] if current_holdings == 0 else [0.3, 0.7]
            )
            price = self.rng.uniform(50, 500)

            if order_type == "buy":
                max_quantity = int(current_equity * 0.3 / price)  # Max 30% of equity
                quantity = self.rng.integers(
                    1, max(2, max_quantity + 1)
                )  # Ensure high > low
                cost = quantity * price
                fees = cost * 0.001  # 0.1% fees
                current_equity -= cost + fees
                current_holdings += quantity
            else:
                if current_holdings > 0:
                    quantity = self.rng.integers(1, current_holdings + 1)
                    revenue = quantity * price
                    fees = revenue * 0.001
                    current_equity += revenue - fees
                    current_holdings -= quantity
                else:
                    continue

            orders.append(
                {
                    "datetime": order_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "type": order_type.upper(),
                    "price": round(price, 2),
                    "quantity": quantity,
                    "status": "FILLED",
                    "tag": f"Strategy_{i % 5}",
                    "equity": round(current_equity, 2),
                    "fees": round(fees, 2),
                    "holdings": current_holdings,
                    "net_profit": round(current_equity - initial_equity, 2),
                    "unrealized": (
                        round(
                            (
                                current_holdings * price
                                - sum(
                                    [
                                        o["quantity"] * o["price"]
                                        for o in orders
                                        if o["type"] == "BUY"
                                    ]
                                )
                            ),
                            2,
                        )
                        if current_holdings > 0
                        else 0
                    ),
                    "volume": quantity * price,
                }
            )

        return sorted(orders, key=lambda x: x["datetime"])

    def _generate_equity_curve(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_equity: float,
        final_equity: float,
    ) -> list[dict]:
        """Generate equity curve data."""
        days = (end_date - start_date).days
        curve = []

        # Generate smooth curve with some volatility
        for i in range(days):
            date = start_date + timedelta(days=i)
            progress = i / days

            # Base growth with some random walk
            base_value = (
                initial_equity + (float(final_equity) - initial_equity) * progress
            )
            noise = self.rng.normal(0, base_value * 0.02)  # 2% daily volatility
            value = max(
                base_value + noise, initial_equity * 0.7
            )  # Don't go below 30% loss

            curve.append({"date": date.strftime("%Y-%m-%d"), "equity": round(value, 2)})

        return curve

    def _generate_benchmark_curve(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_value: float,
    ) -> list[dict]:
        """Get BuyAndHold strategy equity curve from database results."""
        from ..database import get_db_session
        from ..database.models import BacktestResult as DBBacktestResult

        try:
            # Get BuyAndHold results from database
            session = get_db_session()
            try:
                buyandhold_result = (
                    session.query(DBBacktestResult)
                    .filter(
                        DBBacktestResult.symbols.any(symbol),
                        DBBacktestResult.strategy == "BuyAndHold",
                    )
                    .first()
                )

                if (
                    buyandhold_result
                    and hasattr(buyandhold_result, "equity_curve")
                    and buyandhold_result.equity_curve
                ):
                    curve = []
                    for date_str, value in buyandhold_result.equity_curve.items():
                        curve.append(
                            {"date": date_str, "benchmark": round(float(value), 2)}
                        )
                    return curve
            finally:
                session.close()

        except Exception as e:
            self.logger.warning(
                "Failed to get BuyAndHold benchmark from database for %s: %s", symbol, e
            )

        # Fallback to simple simulation if backtest fails
        days = (end_date - start_date).days
        curve = []

        # Use conservative market returns as fallback
        annual_return = 0.08  # 8% annual return (market average)
        daily_return = annual_return / 365

        for i in range(days):
            date = start_date + timedelta(days=i)
            value = initial_value * (1 + daily_return) ** i
            curve.append(
                {"date": date.strftime("%Y-%m-%d"), "benchmark": round(value, 2)}
            )

        return curve

    def _create_html_report(
        self,
        portfolio_config: dict,
        assets_data: dict,
        start_date: str,
        end_date: str,
        metric: str = "sortino_ratio",
    ) -> str:
        """Create comprehensive HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Portfolio Analysis: {portfolio_config["name"]}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"
            onerror="console.error('Failed to load Plotly from CDN');
            document.querySelectorAll('.chart-container').forEach(el =>
            el.innerHTML = '<div style=\'text-align:center;padding:50px;color:#666;\'>' +
            'Chart loading failed. Please check your internet connection.</div>')"></script>
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
        .chart-container {{margin: 30px 0; height: 400px; border-radius: 8px; border: 1px solid #ddd;}}
        .orders-container {{margin-top: 30px;}}
        .orders-table {{width: 100%; border-collapse: collapse; font-size: 0.9em;
                       background: white; border-radius: 8px; overflow: hidden;
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
        .orders-table th {{background: #343a40; color: white; padding: 12px 8px;
                          text-align: left; font-weight: 600;}}
        .orders-table td {{padding: 10px 8px; border-bottom: 1px solid #eee;}}
        .orders-table tr:hover {{background: #f8f9fa;}}
        .buy {{color: #28a745; font-weight: bold;}}
        .sell {{color: #dc3545; font-weight: bold;}}
        .summary-row {{background: #f8f9fa; font-weight: bold;}}
        .tab-container {{margin: 20px 0;}}
        .tab-buttons {{display: flex; background: #f8f9fa; border-radius: 8px 8px 0 0;
                      overflow: hidden;}}
        .tab-button {{flex: 1; padding: 15px; text-align: center; border: none;
                     background: transparent; cursor: pointer; transition: all 0.3s;}}
        .tab-button.active {{background: #007bff; color: white;}}
        .tab-content {{display: none; padding: 20px; border: 1px solid #ddd;
                      border-top: none; border-radius: 0 0 8px 8px;}}
        .tab-content.active {{display: block;}}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{portfolio_config["name"]}</h1>
            <p>Comprehensive Strategy Analysis • {start_date} to {end_date}</p>
        </div>
"""

        # Generate content for each asset
        for symbol, asset_info in assets_data.items():
            data = asset_info["data"]
            strategy = asset_info["best_strategy"]
            timeframe = asset_info.get("best_timeframe", "1d")
            overview = data["overview"]

            # Helper function to format metric values
            def format_metric_display(
                value, decimals=2, suffix="", prefix="", allow_zero=False
            ):
                if value is None:
                    return "N/A"
                # Convert Decimal to float for comparison
                if hasattr(value, "__float__"):
                    value = float(value)
                if isinstance(value, (int, float)) and value == 0 and not allow_zero:
                    return "N/A"
                return f"{prefix}{value:.{decimals}f}{suffix}"

            def format_integer_display(value):
                if value is None or value == 0:
                    return "N/A"
                return f"{value:,}"

            # Build CSS classes for metrics
            psr_class = "positive" if overview["PSR"] > 0.5 else ""
            sharpe_class = "positive" if overview["sharpe_ratio"] > 1 else ""
            profit_class = "positive" if overview["net_profit"] > 0 else "negative"
            alpha_class = "positive" if overview["alpha"] > 0 else "negative"
            sortino_class = "positive" if overview["sortino_ratio"] > 1 else ""

            # Extract long values
            annual_return = overview["compounding_annual_return"]
            best_timeframe = overview.get("best_timeframe", "1d")

            # Check if strategy has meaningful performance (trades > 0)
            has_trades = overview["total_orders"] > 0
            strategy_display = (
                strategy.replace("_", " ").title()
                if has_trades
                else "No viable strategy"
            )
            strategy_badge_color = "#4CAF50" if has_trades else "#FF6B6B"

            html += f"""
        <div class="asset-section">
            <div class="asset-header">
                <h2 class="asset-title">{symbol}</h2>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <span class="strategy-badge" style="background: {strategy_badge_color};">Best: {strategy_display}</span>
                    <span class="strategy-badge" style="background: #FF6B6B;">⏰ {timeframe}</span>
                </div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">PSR</div>
                    <div class="metric-value {psr_class}">{format_metric_display(overview["PSR"], 3)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value {sharpe_class}">{format_metric_display(overview["sharpe_ratio"], 3)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Orders</div>
                    <div class="metric-value">{format_integer_display(overview["total_orders"])}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Net Profit</div>
                    <div class="metric-value {profit_class}">{format_metric_display(overview["net_profit"], 2, "%")}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Average Win</div>
                    <div class="metric-value positive">{format_metric_display(overview["average_win"], 2, "%", "", True)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Average Loss</div>
                    <div class="metric-value negative">{format_metric_display(overview["average_loss"], 2, "%", "", True)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Annual Return</div>
                    <div class="metric-value {profit_class}">{format_metric_display(annual_return, 2, "%")}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">{format_metric_display(overview["drawdown"], 2, "%")}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{format_metric_display(overview["win_rate"], 1, "%")}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Profit/Loss Ratio</div>
                    <div class="metric-value">{format_metric_display(overview["profit_loss_ratio"], 2)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Alpha</div>
                    <div class="metric-value {alpha_class}">{format_metric_display(overview["alpha"], 3, "", "", True)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Beta</div>
                    <div class="metric-value">{format_metric_display(overview["beta"], 3, "", "", True)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sortino Ratio</div>
                    <div class="metric-value {sortino_class}">{format_metric_display(overview["sortino_ratio"], 3)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Fees</div>
                    <div class="metric-value">{format_metric_display(overview["total_fees"], 2, "", "$", True)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Strategy Capacity</div>
                    <div class="metric-value">{format_metric_display(overview["strategy_capacity"], 0, "", "$", True)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Portfolio Turnover</div>
                    <div class="metric-value">{format_metric_display(overview["portfolio_turnover"], 2, "%", "", True)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Best Timeframe</div>
                    <div class="metric-value" style="color: #FF6B6B;">{best_timeframe}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Combination Rank</div>
                    <div class="metric-value">{overview.get("combination_rank", "1/1")}</div>
                </div>
            </div>

            <div class="tab-container">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="showTab('{symbol}', 'equity')">
                        Equity Curve
                    </button>
                    <button class="tab-button" onclick="showTab('{symbol}', 'timeframes')">
                        Timeframe Analysis
                    </button>
                    <button class="tab-button" onclick="showTab('{symbol}', 'orders')">
                        Order History
                    </button>
                </div>

                <div id="{symbol}-equity" class="tab-content active">
                    <div id="chart-{symbol}" class="chart-container"></div>
                </div>

                <div id="{symbol}-timeframes" class="tab-content">
                    <div class="timeframe-analysis">
                        <h3>Strategy + Timeframe Combinations Analysis</h3>
                        <table class="orders-table" style="margin-top: 20px;">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Strategy</th>
                                    <th>Timeframe</th>
                                    <th>{metric.replace("_", " ").title()}</th>
                                    <th>Total Return</th>
                                    <th>Max Drawdown</th>
                                    <th>Win Rate</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
"""

            # Add timeframe analysis rows if available
            if "timeframe_analysis" in data:
                sorted_combos = sorted(
                    data["timeframe_analysis"], key=lambda x: x["score"], reverse=True
                )

                # Remove duplicates based on strategy name
                unique_combos = []
                seen_strategies = set()
                for combo in sorted_combos:
                    if combo["strategy"] not in seen_strategies:
                        seen_strategies.add(combo["strategy"])
                        unique_combos.append(combo)

                for i, combo in enumerate(unique_combos, 1):  # Show all unique results
                    is_best = (
                        combo["strategy"] == strategy
                        and combo["timeframe"] == timeframe
                    )
                    status_badge = "🏆 BEST" if is_best else ""
                    row_class = "summary-row" if is_best else ""

                    # Helper function to format metric values
                    def format_metric(value, suffix="", decimals=3):
                        if value is None or (
                            isinstance(value, (int, float)) and value == 0
                        ):
                            return "N/A"
                        return f"{value:.{decimals}f}{suffix}"

                    html += f"""
                                <tr class="{row_class}">
                                    <td>{i}</td>
                                    <td>{combo["strategy"].replace("_", " ").title()}</td>
                                    <td><strong>{combo["timeframe"]}</strong></td>
                                    <td>{format_metric(combo["score"])}</td>
                                    <td>{format_metric(combo["metrics"]["total_return"], "%", 1)}</td>
                                    <td>{format_metric(combo["metrics"]["max_drawdown"], "%", 1)}</td>
                                    <td>{format_metric(combo["metrics"]["win_rate"], "%", 1)}</td>
                                    <td>{status_badge}</td>
                                </tr>
"""

            html += f"""
                            </tbody>
                        </table>
                    </div>
                </div>

                <div id="{symbol}-orders" class="tab-content">
                    <div class="orders-container">
                        <table class="orders-table">
                            <thead>
                                <tr>
                                    <th>Date/Time</th>
                                    <th>Type</th>
                                    <th>Price</th>
                                    <th>Quantity</th>
                                    <th>Equity</th>
                                    <th>Fees</th>
                                    <th>Holdings</th>
                                    <th>Net Profit</th>
                                    <th>Unrealized</th>
                                </tr>
                            </thead>
                            <tbody>
"""

            # Add order rows (show last 50 to keep size reasonable)
            recent_orders = (
                data["orders"][-50:] if len(data["orders"]) > 50 else data["orders"]
            )
            for order in recent_orders:
                order_type_class = "buy" if order["type"] == "BUY" else "sell"
                profit_class = "positive" if order["net_profit"] > 0 else "negative"

                html += f"""
                                <tr>
                                    <td>{order["datetime"]}</td>
                                    <td class="{order_type_class}">{order["type"]}</td>
                                    <td>${order["price"]:.2f}</td>
                                    <td>{order["quantity"]:,}</td>
                                    <td>${order["equity"]:,.2f}</td>
                                    <td>${order["fees"]:.2f}</td>
                                    <td>{order["holdings"]:,}</td>
                                    <td class="{profit_class}">${order["net_profit"]:,.2f}</td>
                                    <td>${order["unrealized"]:,.2f}</td>
                                </tr>
"""

            # Add summary row
            total_fees = sum(order["fees"] for order in data["orders"])
            final_equity = (
                data["orders"][-1]["equity"]
                if data["orders"]
                else overview["start_equity"]
            )

            html += f"""
                                <tr class="summary-row">
                                    <td colspan="4">SUMMARY ({len(data["orders"])} total orders)</td>
                                    <td>${final_equity:,.2f}</td>
                                    <td>${total_fees:.2f}</td>
                                    <td>-</td>
                                    <td class="{profit_class}">${overview["net_profit"]:,.2f}%</td>
                                    <td>-</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
"""

        # Add JavaScript for charts and interactivity
        html += """
    </div>

    <script>
        function showTab(symbol, tabName) {
            // Hide all tab contents for this symbol
            const contents = document.querySelectorAll(`[id^="${symbol}-"]`);
            contents.forEach(content => content.classList.remove('active'));

            // Hide all tab buttons for this symbol
            const buttons = document.querySelectorAll(`button[onclick*="${symbol}"]`);
            buttons.forEach(button => button.classList.remove('active'));

            // Show selected tab
            document.getElementById(`${symbol}-${tabName}`).classList.add('active');
            event.target.classList.add('active');
        }

        // Generate charts for each asset
        document.addEventListener('DOMContentLoaded', function() {
            // Check if Plotly is available
            if (typeof Plotly === 'undefined') {
                console.error('Plotly not loaded');
                document.querySelectorAll('.chart-container').forEach(el => {
                    el.innerHTML = '<div style="text-align:center;padding:50px;color:#666;">' +
                                  'Charts unavailable - Plotly library not loaded</div>';
                });
                return;
            }

            try {
"""

        # Add chart data and rendering for each asset
        for symbol, asset_info in assets_data.items():
            data = asset_info["data"]

            # Create safe JavaScript variable name
            safe_symbol = (
                symbol.replace("=", "_")
                .replace("-", "_")
                .replace("/", "_")
                .replace(".", "_")
            )

            # Prepare chart data
            equity_dates = [point["date"] for point in data["equity_curve"]]
            equity_values = [point["equity"] for point in data["equity_curve"]]

            # Only show benchmark if best strategy is not BuyAndHold
            # (Not used in current implementation)

            # For now, don't show benchmark curves to avoid complexity
            # The timeframe analysis table will show all strategy comparisons

            html += f"""
            // Chart for {symbol}
            const equityTrace_{safe_symbol} = {{
                x: {json.dumps(equity_dates)},
                y: {json.dumps(equity_values)},
                type: 'scatter',
                mode: 'lines',
                name: '{symbol} Strategy',
                line: {{color: '#007bff', width: 2}}
            }};
"""

            # Simplified: no benchmark curves for now
            # Always show single equity curve
            html += f"""
            const layout_{safe_symbol} = {{
                title: '{symbol} - Equity Curve',
                xaxis: {{title: 'Date'}},
                yaxis: {{title: 'Portfolio Value ($)'}},
                hovermode: 'x unified',
                margin: {{l: 60, r: 30, t: 60, b: 60}}
            }};

            Plotly.newPlot('chart-{symbol}',
                          [equityTrace_{safe_symbol}],
                          layout_{safe_symbol}, {{responsive: true}});
"""

        html += """
            } catch (error) {
                console.error('Error generating charts:', error);
                document.querySelectorAll('.chart-container').forEach(el => {
                    el.innerHTML = '<div style="text-align:center;padding:50px;color:#666;">' +
                                  'Chart generation failed: ' + error.message + '</div>';
                });
            }
        });
    </script>
</body>
</html>"""

        return html

    def _save_compressed_report(self, html_content: str, portfolio_name: str) -> str:
        """Save HTML report with quarterly organization and compression."""
        # Create temporary file first
        reports_dir = Path("exports/reports")
        reports_dir.mkdir(exist_ok=True)

        # Generate temporary filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        temp_filename = (
            f"portfolio_report_{portfolio_name.replace(' ', '_')}_{timestamp}.html"
        )
        temp_filepath = reports_dir / temp_filename

        # Save temporary HTML file
        with temp_filepath.open("w", encoding="utf-8") as f:
            f.write(html_content)

        # Organize into quarterly structure (this will handle overriding existing reports)
        organized_path = self.report_organizer.organize_report(
            str(temp_filepath), portfolio_name, datetime.now(timezone.utc)
        )

        # Remove temporary file
        temp_filepath.unlink()

        # Save compressed version alongside organized report
        with gzip.open(
            organized_path.with_suffix(".html.gz"), "wt", encoding="utf-8"
        ) as f:
            f.write(html_content)

        # Return path to organized HTML file
        return str(organized_path)

    def _get_real_metrics_from_db(
        self, symbol: str, strategy: str, timeframe: str = "1d"
    ) -> dict:
        """Get real metrics from database for symbol/strategy/timeframe combination."""
        from src.database.db_connection import get_db_session
        from src.database.models import BacktestResult, BestStrategy

        session = get_db_session()
        try:
            # Try to get from best_strategies table first
            best_strategy = (
                session.query(BestStrategy)
                .filter_by(symbol=symbol, timeframe=timeframe)
                .first()
            )

            if best_strategy and best_strategy.best_strategy == strategy:
                return {
                    "sortino_ratio": float(best_strategy.sortino_ratio or 0),
                    "sharpe_ratio": float(best_strategy.sharpe_ratio or 0),
                    "total_return": float(best_strategy.total_return or 0),
                    "max_drawdown": float(best_strategy.max_drawdown or 0),
                    "volatility": float(best_strategy.volatility or 0),
                    "win_rate": float(best_strategy.win_rate or 0),
                    "profit_factor": float(best_strategy.profit_factor or 1),
                    "num_trades": int(best_strategy.num_trades or 0),
                    "alpha": float(best_strategy.alpha or 0),
                    "beta": float(best_strategy.beta or 1),
                    "expectancy": float(best_strategy.expectancy or 0),
                    "average_win": float(best_strategy.average_win or 0),
                    "average_loss": float(best_strategy.average_loss or 0),
                    "total_fees": float(best_strategy.total_fees or 0),
                    "portfolio_turnover": float(best_strategy.portfolio_turnover or 0),
                    "strategy_capacity": float(
                        best_strategy.strategy_capacity or 1000000
                    ),
                }

            # Fallback to BacktestResult table
            result = (
                session.query(BacktestResult)
                .filter(
                    BacktestResult.name.like(f"%{symbol}%"),
                    BacktestResult.strategy == strategy,
                )
                .order_by(BacktestResult.created_at.desc())
                .first()
            )

            if result:
                return {
                    "sortino_ratio": float(result.sortino_ratio or 0),
                    "sharpe_ratio": float(result.sharpe_ratio or 0),
                    "total_return": float(result.total_return or 0),
                    "max_drawdown": float(result.max_drawdown or 0),
                    "volatility": float(result.volatility or 0),
                    "win_rate": float(result.win_rate or 0),
                    "profit_factor": float(result.profit_factor or 1),
                    "num_trades": int(result.num_trades or 0),
                    "alpha": float(result.alpha or 0),
                    "beta": float(result.beta or 1),
                    "expectancy": float(result.expectancy or 0),
                    "average_win": float(result.average_win or 0),
                    "average_loss": float(result.average_loss or 0),
                    "total_fees": float(result.total_fees or 0),
                    "portfolio_turnover": float(result.portfolio_turnover or 0),
                    "strategy_capacity": float(result.strategy_capacity or 1000000),
                }

            # Return zero metrics if no data found
            return {
                "sortino_ratio": 0.0,
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.25,
                "win_rate": 0.0,
                "profit_factor": 1.0,
                "num_trades": 0,
                "alpha": 0.0,
                "beta": 1.0,
                "expectancy": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "total_fees": 0.0,
                "portfolio_turnover": 0.0,
                "strategy_capacity": 1000000.0,
            }

        except Exception as e:
            print(f"Error getting metrics from database: {e}")
            return {
                "sortino_ratio": 0.0,
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.25,
                "win_rate": 0.0,
                "profit_factor": 1.0,
                "num_trades": 0,
                "alpha": 0.0,
                "beta": 1.0,
                "expectancy": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "total_fees": 0.0,
                "portfolio_turnover": 0.0,
                "strategy_capacity": 1000000.0,
            }
        finally:
            session.close()
