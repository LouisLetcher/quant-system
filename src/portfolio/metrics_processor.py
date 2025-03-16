from __future__ import annotations

import json
import math
import os
from datetime import datetime

import pandas as pd

from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def extract_detailed_metrics(result, initial_capital):
    """Extract and format detailed metrics from backtest result."""
    logger.debug("Extracting detailed metrics from backtest result")

    # Check for NaN values and replace them with defaults
    def safe_get(key, default):
        val = result.get(key, default)
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            logger.warning(
                f"Found NaN or Inf value for {key}, using default: {default}"
            )
            return default
        return val

    detailed_metrics = {
        # Account metrics
        "initial_capital": initial_capital,
        "equity_final": safe_get("Equity Final [$]", initial_capital),
        "equity_peak": safe_get("Equity Peak [$]", initial_capital),
        # Return metrics
        "return_pct": safe_get("Return [%]", 0),
        "return": f"{safe_get('Return [%]', 0):.2f}%",
        "return_annualized": safe_get("Return (Ann.) [%]", 0),
        "buy_hold_return": safe_get("Buy & Hold Return [%]", 0),
        "cagr": safe_get("CAGR [%]", 0),
        # Risk metrics
        "sharpe_ratio": safe_get("Sharpe Ratio", 0),
        "sortino_ratio": safe_get("Sortino Ratio", 0),
        "calmar_ratio": safe_get("Calmar Ratio", 0),
        "max_drawdown_pct": safe_get("Max. Drawdown [%]", 0),
        "max_drawdown": f"{safe_get('Max. Drawdown [%]', 0):.2f}%",
        "avg_drawdown": safe_get("Avg. Drawdown [%]", 0),
        "avg_drawdown_duration": safe_get("Avg. Drawdown Duration", "N/A"),
        "volatility": safe_get("Volatility (Ann.) [%]", 0),
        "alpha": safe_get("Alpha", 0),
        "beta": safe_get("Beta", 0),
        # Trade metrics
        "trades_count": safe_get("# Trades", 0),
        "win_rate": safe_get("Win Rate [%]", 0),
        "profit_factor": safe_get("Profit Factor", 0),
        "tv_profit_factor": safe_get("Profit Factor", "N/A"),
        "expectancy": safe_get("Expectancy [%]", 0),
        "sqn": safe_get("SQN", 0),
        "kelly_criterion": safe_get("Kelly Criterion", 0),
        "avg_trade_pct": safe_get("Avg. Trade [%]", 0),
        "best_trade_pct": safe_get("Best Trade [%]", 0),
        "best_trade": safe_get("Best Trade [%]", 0),
        "worst_trade_pct": safe_get("Worst Trade [%]", 0),
        "worst_trade": safe_get("Worst Trade [%]", 0),
        "avg_trade_duration": safe_get("Avg. Trade Duration", "N/A"),
        "max_trade_duration": safe_get("Max. Trade Duration", "N/A"),
        "exposure_time": safe_get("Exposure Time [%]", 0),
    }

    logger.debug(
        f"Extracted basic metrics: profit_factor={detailed_metrics['profit_factor']}, return={detailed_metrics['return_pct']}%, win_rate={detailed_metrics['win_rate']}%"
    )

    # Process trades into list format expected by template
    if "_trades" in result and not result["_trades"].empty:
        trades_df = result["_trades"]
        trades_list = []
        logger.debug(f"Processing {len(trades_df)} trades")

        for _, trade in trades_df.iterrows():
            try:
                trade_data = {
                    "entry_date": str(trade["EntryTime"]),
                    "exit_date": str(trade["ExitTime"]),
                    "type": "LONG",  # Assuming all trades are LONG
                    "entry_price": float(trade["EntryPrice"]),
                    "exit_price": float(trade["ExitPrice"]),
                    "size": int(trade["Size"]),
                    "pnl": float(trade["PnL"]),
                    "return_pct": float(trade["ReturnPct"]) * 100,
                    "duration": trade["Duration"],
                }
                trades_list.append(trade_data)
            except Exception as e:
                logger.error(f"Error processing trade: {e}")
                logger.error(f"Trade data: {trade}")

        detailed_metrics["trades"] = trades_list
        detailed_metrics["total_pnl"] = sum(trade["pnl"] for trade in trades_list)

        # Calculate additional trade statistics
        if trades_list:
            winning_trades = [t for t in trades_list if t["pnl"] > 0]
            losing_trades = [t for t in trades_list if t["pnl"] < 0]

            win_count = len(winning_trades)
            loss_count = len(losing_trades)

            logger.debug(f"Trade statistics: {win_count} winning, {loss_count} losing")

            if winning_trades:
                avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades)
                max_win = max(t["pnl"] for t in winning_trades)
                detailed_metrics["avg_win"] = avg_win
                detailed_metrics["max_win"] = max_win
                logger.debug(f"Average win: ${avg_win:.2f}, Max win: ${max_win:.2f}")

            if losing_trades:
                avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades)
                max_loss = min(t["pnl"] for t in losing_trades)
                detailed_metrics["avg_loss"] = avg_loss
                detailed_metrics["max_loss"] = max_loss
                logger.debug(
                    f"Average loss: ${avg_loss:.2f}, Max loss: ${max_loss:.2f}"
                )
    else:
        # Make sure we have an empty list if no trades
        logger.debug("No trades found in backtest result")
        detailed_metrics["trades"] = []
        detailed_metrics["total_pnl"] = 0

    # Process equity curve
    if "_equity_curve" in result:
        equity_data = result["_equity_curve"]
        equity_curve = []
        logger.debug(f"Processing equity curve with {len(equity_data)} points")

        try:
            # Handle different equity curve data structures
            if isinstance(equity_data, pd.DataFrame):
                for date, row in equity_data.iterrows():
                    val = (
                        row.iloc[0]
                        if isinstance(row, pd.Series) and len(row) > 0
                        else row
                    )
                    equity_curve.append(
                        {
                            "date": str(date),
                            "value": float(val) if not pd.isna(val) else 0.0,
                        }
                    )
            else:
                for date, val in zip(equity_data.index, equity_data.values):
                    # Handle numpy values
                    if hasattr(val, "item"):
                        try:
                            val = val.item()
                        except (ValueError, TypeError):
                            val = val[0] if len(val) > 0 else 0

                    equity_curve.append(
                        {
                            "date": str(date),
                            "value": float(val) if not pd.isna(val) else 0.0,
                        }
                    )

            detailed_metrics["equity_curve"] = equity_curve
            logger.debug(f"Extracted {len(equity_curve)} equity curve points")

            # Verify equity curve data quality
            if equity_curve:
                min_value = min(point["value"] for point in equity_curve)
                max_value = max(point["value"] for point in equity_curve)
                logger.debug(f"Equity curve range: {min_value} to {max_value}")

                # Check for suspicious values
                if min_value < 0:
                    logger.warning(
                        f"Negative values detected in equity curve: minimum = {min_value}"
                    )
                if max_value == 0:
                    logger.warning("All equity curve values are zero")
        except Exception as e:
            logger.error(f"Error processing equity curve: {e}")
            import traceback

            logger.error(traceback.format_exc())
            detailed_metrics["equity_curve"] = []

    return ensure_all_metrics_exist(detailed_metrics)


def ensure_all_metrics_exist(asset_data):
    """
    Ensure all required metrics exist in the asset data.

    Args:
        asset_data: Dictionary containing asset metrics

    Returns:
        Dictionary with all required metrics (adding defaults for missing ones)
    """
    required_metrics = {
        # Return metrics
        "return_pct": 0,
        "return_annualized": 0,
        "buy_hold_return": 0,
        "cagr": 0,
        # Risk metrics
        "sharpe_ratio": 0,
        "sortino_ratio": 0,
        "calmar_ratio": 0,
        "max_drawdown_pct": 0,
        "avg_drawdown": 0,
        "avg_drawdown_duration": "N/A",
        "volatility": 0,
        "alpha": 0,
        "beta": 0,
        # Trade metrics
        "trades_count": 0,
        "win_rate": 0,
        "profit_factor": 0,
        "expectancy": 0,
        "sqn": 0,
        "kelly_criterion": 0,
        "avg_trade_pct": 0,
        "avg_trade": 0,
        "best_trade_pct": 0,
        "best_trade": 0,
        "worst_trade_pct": 0,
        "worst_trade": 0,
        "avg_trade_duration": "N/A",
        "max_trade_duration": "N/A",
        "exposure_time": 0,
        # Account metrics
        "initial_capital": 10000,
        "equity_final": 10000,
        "equity_peak": 10000,
    }

    # Add default values for missing metrics
    for metric, default_value in required_metrics.items():
        if metric not in asset_data:
            asset_data[metric] = default_value

    return asset_data


def generate_log_summary(portfolio_name, best_combinations, metric):
    """Generate a comprehensive summary of portfolio optimization results for logging."""
    logger.info("=" * 50)
    logger.info(f"PORTFOLIO OPTIMIZATION SUMMARY FOR '{portfolio_name}'")
    logger.info("=" * 50)

    # Calculate overall portfolio statistics
    total_assets = len(best_combinations)
    assets_with_valid_strategy = sum(
        1 for combo in best_combinations.values() if combo.get("strategy") is not None
    )

    logger.info(f"Total assets: {total_assets}")
    logger.info(
        f"Assets with valid strategy: {assets_with_valid_strategy} ({assets_with_valid_strategy/total_assets*100 if total_assets else 0:.1f}%)"
    )
    logger.info(f"Optimization metric: {metric}")

    # Calculate average metrics across portfolio
    if assets_with_valid_strategy > 0:
        avg_return = (
            sum(
                combo.get("return_pct", 0)
                for combo in best_combinations.values()
                if combo.get("strategy") is not None
            )
            / assets_with_valid_strategy
        )
        avg_win_rate = (
            sum(
                combo.get("win_rate", 0)
                for combo in best_combinations.values()
                if combo.get("strategy") is not None
            )
            / assets_with_valid_strategy
        )
        avg_profit_factor = (
            sum(
                combo.get("profit_factor", 0)
                for combo in best_combinations.values()
                if combo.get("strategy") is not None
            )
            / assets_with_valid_strategy
        )
        avg_trades = (
            sum(
                combo.get("trades_count", 0)
                for combo in best_combinations.values()
                if combo.get("strategy") is not None
            )
            / assets_with_valid_strategy
        )

        logger.info(f"Average return: {avg_return:.2f}%")
        logger.info(f"Average win rate: {avg_win_rate:.2f}%")
        logger.info(f"Average profit factor: {avg_profit_factor:.2f}")
        logger.info(f"Average trades per asset: {avg_trades:.1f}")

    # Strategy distribution
    strategy_counts = {}
    interval_counts = {}

    for combo in best_combinations.values():
        if combo.get("strategy") is not None:
            strategy = combo.get("strategy")
            interval = combo.get("interval")

            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            interval_counts[interval] = interval_counts.get(interval, 0) + 1

    logger.info("\nStrategy distribution:")
    for strategy, count in sorted(
        strategy_counts.items(), key=lambda x: x[1], reverse=True
    ):
        logger.info(
            f"  {strategy}: {count} assets ({count/assets_with_valid_strategy*100:.1f}%)"
        )

    logger.info("\nInterval distribution:")
    for interval, count in sorted(
        interval_counts.items(), key=lambda x: x[1], reverse=True
    ):
        logger.info(
            f"  {interval}: {count} assets ({count/assets_with_valid_strategy*100:.1f}%)"
        )

    # Individual asset results
    logger.info("\nIndividual asset results:")
    for ticker, combo in sorted(best_combinations.items()):
        if combo.get("strategy") is not None:
            logger.info(
                f"  {ticker}: {combo['strategy']} with {combo['interval']} interval"
            )
            logger.info(
                f"    {metric}: {combo['score']:.4f}, Return: {combo.get('return_pct', 0):.2f}%, Win Rate: {combo.get('win_rate', 0):.1f}%"
            )
            logger.info(
                f"    Trades: {combo.get('trades_count', 0)}, Profit Factor: {combo.get('profit_factor', 0):.2f}"
            )
        else:
            logger.info(f"  {ticker}: No valid strategy found")

    logger.info("=" * 50)


def save_backtest_results(ticker, strategy, interval, results, initial_capital):
    """Save detailed backtest results to a file for later analysis."""
    try:
        # Create a directory for backtest results
        results_dir = os.path.join("logs", "backtest_results")
        os.makedirs(results_dir, exist_ok=True)

        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_{strategy}_{interval}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)

        # Extract key metrics for saving
        metrics = extract_detailed_metrics(results, initial_capital)

        # Remove non-serializable objects like backtest_obj
        if "backtest_obj" in metrics:
            del metrics["backtest_obj"]

        # Add metadata
        metrics["ticker"] = ticker
        metrics["strategy"] = strategy
        metrics["interval"] = interval
        metrics["timestamp"] = timestamp

        # Limit the size of equity curve for storage
        if "equity_curve" in metrics and len(metrics["equity_curve"]) > 1000:
            # Sample the equity curve to reduce size
            sample_rate = max(1, len(metrics["equity_curve"]) // 1000)
            metrics["equity_curve"] = metrics["equity_curve"][::sample_rate]
            logger.debug(
                f"Sampled equity curve from {len(metrics['equity_curve'])} to {len(metrics['equity_curve'][::sample_rate])} points for storage"
            )

        # Save to file
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Saved detailed backtest results to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving backtest results: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None
