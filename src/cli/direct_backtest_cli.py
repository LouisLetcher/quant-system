"""
Direct Backtesting CLI Functions
Uses backtesting library directly for ground truth results.
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.database import get_db_session
from src.database.models import BacktestResult as DBBacktestResult
from src.database.models import BestStrategy, Trade


def save_direct_backtest_to_database(result_dict: dict, metric: str = "sortino_ratio"):
    """Save direct backtesting library results to database."""
    logger = logging.getLogger(__name__)

    if result_dict["error"]:
        logger.warning("Cannot save failed backtest: %s", result_dict["error"])
        return

    symbol = result_dict["symbol"]
    strategy = result_dict["strategy"]
    timeframe = result_dict["timeframe"]
    metrics = result_dict["metrics"]

    session = get_db_session()

    try:
        # Create BacktestResult entry
        db_result = DBBacktestResult(
            name=f"direct_{strategy}_{symbol}_{timeframe}",
            symbols=[symbol],
            strategy=strategy,
            timeframe=timeframe,
            start_date=datetime.strptime(
                "2023-01-01", "%Y-%m-%d"
            ).date(),  # Use from result_dict if available
            end_date=datetime.strptime("2023-12-31", "%Y-%m-%d").date(),
            initial_capital=metrics.get("start_value", 10000.0),
            final_value=metrics.get("end_value", 10000.0),
            total_return=metrics.get("total_return", 0.0),
            sortino_ratio=metrics.get("sortino_ratio", 0.0),
            calmar_ratio=metrics.get("calmar_ratio", 0.0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            profit_factor=metrics.get("profit_factor", 1.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            volatility=metrics.get("volatility", 0.0),
            downside_deviation=0.0,  # Not available from backtesting library directly
            win_rate=metrics.get("win_rate", 0.0),
            trades_count=metrics.get("num_trades", 0),
            average_win=0.0,  # Could be calculated from trades if needed
            average_loss=0.0,
            parameters={},
        )

        session.add(db_result)
        session.flush()  # Get the ID

        # Save real trades from backtesting library
        if result_dict["trades"] is not None and not result_dict["trades"].empty:
            trades_df = result_dict["trades"]

            for _, trade_row in trades_df.iterrows():
                # Convert backtesting library trade format to our database format
                trade_type = "BUY" if trade_row["Size"] > 0 else "SELL"

                trade_record = Trade(
                    backtest_result_id=db_result.id,
                    symbol=symbol,
                    trade_datetime=trade_row["EntryTime"],
                    trade_type=trade_type,
                    price=float(trade_row["EntryPrice"]),
                    quantity=abs(float(trade_row["Size"])),
                    value=abs(float(trade_row["Size"]))
                    * float(trade_row["EntryPrice"]),
                    fees=float(trade_row.get("Commission", 0)),
                    equity_after_trade=0.0,  # Would need calculation
                    holdings_after_trade=float(trade_row["Size"]),
                    cash_after_trade=0.0,  # Would need calculation
                    net_profit=float(trade_row["PnL"]),
                    unrealized_pnl=0.0,
                    entry_reason=f"{trade_type} signal from {strategy}",
                )
                session.add(trade_record)

        # Update BestStrategy table
        update_best_strategy_direct(session, result_dict, metric)

        session.commit()
        logger.info("Saved %s/%s results to database", symbol, strategy)

    except Exception as e:
        session.rollback()
        logger.error("Failed to save %s/%s: %s", symbol, strategy, e)
        raise e
    finally:
        session.close()


def update_best_strategy_direct(
    session, result_dict: dict, metric: str = "sortino_ratio"
):
    """Update best strategy table with direct backtesting results."""
    logger = logging.getLogger(__name__)
    symbol = result_dict["symbol"]
    strategy = result_dict["strategy"]
    timeframe = result_dict["timeframe"]
    metrics = result_dict["metrics"]

    # Check existing best strategy
    existing = (
        session.query(BestStrategy)
        .filter_by(symbol=symbol, timeframe=timeframe)
        .first()
    )

    current_metric_value = metrics.get(metric, 0)
    current_num_trades = metrics.get("num_trades", 0)

    # Determine if this is better
    is_better = False
    if not existing:
        is_better = True
    else:
        existing_metric_value = getattr(existing, metric, 0) or 0
        existing_num_trades = existing.num_trades or 0

        # Prefer strategies with actual trades
        if current_num_trades > 0 and existing_num_trades == 0:
            is_better = True
        elif current_num_trades == 0 and existing_num_trades > 0:
            is_better = False
        else:
            # Compare by metric (higher is better for most metrics)
            if metric == "max_drawdown":
                is_better = current_metric_value < existing_metric_value
            else:
                is_better = current_metric_value > existing_metric_value

    if is_better:
        if existing:
            # Update existing record
            existing.best_strategy = strategy
            existing.sortino_ratio = metrics.get("sortino_ratio", 0)
            existing.sharpe_ratio = metrics.get("sharpe_ratio", 0)
            existing.total_return = metrics.get("total_return", 0)
            existing.max_drawdown = metrics.get("max_drawdown", 0)
            existing.volatility = metrics.get("volatility", 0)
            existing.win_rate = metrics.get("win_rate", 0)
            existing.num_trades = metrics.get("num_trades", 0)
            existing.profit_factor = metrics.get("profit_factor", 1.0)
        else:
            # Create new record
            new_best = BestStrategy(
                symbol=symbol,
                timeframe=timeframe,
                best_strategy=strategy,
                sortino_ratio=metrics.get("sortino_ratio", 0),
                sharpe_ratio=metrics.get("sharpe_ratio", 0),
                total_return=metrics.get("total_return", 0),
                max_drawdown=metrics.get("max_drawdown", 0),
                volatility=metrics.get("volatility", 0),
                win_rate=metrics.get("win_rate", 0),
                num_trades=metrics.get("num_trades", 0),
                profit_factor=metrics.get("profit_factor", 1.0),
                risk_score=metrics.get("max_drawdown", 0)
                + metrics.get("volatility", 0),
            )
            session.add(new_best)

        logger.info(
            "Updated best strategy for %s/%s: %s (Sortino: %.3f)",
            symbol,
            timeframe,
            strategy,
            current_metric_value,
        )
