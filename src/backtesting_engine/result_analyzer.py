from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class BacktestResultAnalyzer:
    """Analyzes backtest results and extracts performance metrics."""

    @staticmethod
    def analyze(backtest_result, ticker=None, initial_capital=10000):
        """Extracts key performance metrics from the backtest results."""
        logger.info(f"Analyzing backtest results for {ticker if ticker else 'unknown asset'}")
        
        if backtest_result is None:
            logger.error("No results returned from Backtest Engine.")
            return {
                "strategy": "N/A",
                "asset": "N/A" if ticker is None else ticker,
                "pnl": "$0.00",
                "sharpe_ratio": 0,
                "max_drawdown": "0.00%",
                "trades": 0,
                "initial_capital": initial_capital,
                "final_value": initial_capital,
                "suspicious_result": False,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "equity_curve": [],
                "drawdown_curve": [],
                "trades_list": [],
            }

        # Extract metrics directly from Backtesting.py results
        # General infos
        asset_name = ticker if ticker else "N/A"
        logger.info(f"Asset: {asset_name}")
        
        # Time metrics
        start_date = backtest_result.get('Start')
        end_date = backtest_result.get('End')
        duration = backtest_result.get('Duration')
        exposure_time = backtest_result.get('Exposure Time [%]')
        
        logger.info(f"Time period: {start_date} to {end_date} ({duration})")
        logger.info(f"Exposure time: {exposure_time}%")
        
        # Equity and Return metrics
        equity_final = backtest_result.get('Equity Final [$]')
        equity_peak = backtest_result.get('Equity Peak [$]')
        return_percent = backtest_result.get('Return [%]')
        buy_hold_return = backtest_result.get('Buy & Hold Return [%]')
        return_annualized = backtest_result.get('Return (Ann.) [%]')
        
        logger.info(f"Final equity: ${equity_final}")
        logger.info(f"Peak equity: ${equity_peak}")
        logger.info(f"Return: {return_percent}% (Buy & Hold: {buy_hold_return}%)")
        logger.info(f"Annualized return: {return_annualized}%")
        
        # Risk metrics
        volatility = backtest_result.get('Volatility [%]')
        cagr = backtest_result.get('CAGR [%]')
        sharpe_ratio = backtest_result.get('Sharpe Ratio')
        sortino_ratio = backtest_result.get('Sortino Ratio')
        alpha = backtest_result.get('Alpha')
        beta = backtest_result.get('Beta')
        max_drawdown = backtest_result.get('Max. Drawdown [%]')
        avg_drawdown = backtest_result.get('Avg. Drawdown [%]')
        avg_drawdown_duration = backtest_result.get('Avg. Drawdown Duration')
        
        logger.info(f"Risk metrics:")
        logger.info(f"  Sharpe ratio: {sharpe_ratio}")
        logger.info(f"  Sortino ratio: {sortino_ratio}")
        logger.info(f"  Max drawdown: {max_drawdown}%")
        logger.info(f"  Volatility: {volatility}%")
        logger.info(f"  CAGR: {cagr}%")
        logger.info(f"  Alpha: {alpha}, Beta: {beta}")
        
        # Trade metrics
        trade_count = backtest_result.get("# Trades")
        trades = backtest_result.get('Trades')
        win_rate = backtest_result.get('Win Rate [%]')
        best_trade = backtest_result.get('Best Trade [%]')
        worst_trade = backtest_result.get('Worst Trade [%]')
        avg_trade = backtest_result.get('Avg. Trade [%]')
        max_trade_duration = backtest_result.get('Max. Trade Duration')
        avg_trade_duration = backtest_result.get('Avg. Trade Duration')
        
        logger.info(f"Trade metrics:")
        logger.info(f"  Total trades: {trade_count}")
        logger.info(f"  Win rate: {win_rate}%")
        logger.info(f"  Best trade: {best_trade}%")
        logger.info(f"  Worst trade: {worst_trade}%")
        logger.info(f"  Avg trade: {avg_trade}%")
        logger.info(f"  Avg trade duration: {avg_trade_duration}")
        
        # Performance metrics
        profit_factor = backtest_result.get('Profit Factor')
        expectancy = backtest_result.get('Expectancy')
        sqn = backtest_result.get('SQN')
        kelly_criterion = backtest_result.get('Kelly Criterion')
        
        logger.info(f"Performance metrics:")
        logger.info(f"  Profit factor: {profit_factor}")
        logger.info(f"  Expectancy: {expectancy}")
        logger.info(f"  SQN: {sqn}")
        logger.info(f"  Kelly criterion: {kelly_criterion}")

        # Extract equity curve and trades data
        equity_curve = BacktestResultAnalyzer._extract_equity_curve(backtest_result)
        drawdown_curve = BacktestResultAnalyzer._extract_drawdown_curve(backtest_result)
        trades_list = BacktestResultAnalyzer._extract_trades_list(backtest_result)
        
        # Log detailed trade information
        if trades_list:
            logger.info(f"Extracted {len(trades_list)} trades")
            
            # Calculate additional trade statistics
            winning_trades = [t for t in trades_list if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades_list if t.get('pnl', 0) < 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            logger.info(f"  Winning trades: {win_count} ({win_count/len(trades_list)*100 if trades_list else 0:.2f}%)")
            logger.info(f"  Losing trades: {loss_count} ({loss_count/len(trades_list)*100 if trades_list else 0:.2f}%)")
            
            if winning_trades:
                avg_win_amount = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades)
                max_win_amount = max(t.get('pnl', 0) for t in winning_trades)
                logger.info(f"  Average winning trade: ${avg_win_amount:.2f}")
                logger.info(f"  Largest winning trade: ${max_win_amount:.2f}")
            
            if losing_trades:
                avg_loss_amount = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades)
                max_loss_amount = min(t.get('pnl', 0) for t in losing_trades)
                logger.info(f"  Average losing trade: ${avg_loss_amount:.2f}")
                logger.info(f"  Largest losing trade: ${max_loss_amount:.2f}")
            
            # Save trades to CSV for further analysis
            if ticker:
                trades_df = pd.DataFrame(trades_list)
                trades_log = f"logs/analyzed_trades_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                os.makedirs(os.path.dirname(trades_log), exist_ok=True)
                trades_df.to_csv(trades_log, index=False)
                logger.info(f"Detailed trades saved to {trades_log}")
                
                # Analyze trade distribution by month/year if possible
                try:
                    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                    trades_df['month'] = trades_df['entry_date'].dt.to_period('M')
                    trades_df['year'] = trades_df['entry_date'].dt.to_period('Y')
                    
                    monthly_pnl = trades_df.groupby('month')['pnl'].sum()
                    yearly_pnl = trades_df.groupby('year')['pnl'].sum()
                    
                    logger.info("Monthly P&L distribution:")
                    for month, pnl in monthly_pnl.items():
                        logger.info(f"  {month}: ${pnl:.2f}")
                    
                    logger.info("Yearly P&L distribution:")
                    for year, pnl in yearly_pnl.items():
                        logger.info(f"  {year}: ${pnl:.2f}")
                except Exception as e:
                    logger.warning(f"Could not analyze trade distribution by period: {e}")
        else:
            logger.warning("No trades extracted from backtest results")

        # Create comprehensive results dictionary
        results = {
            "asset": asset_name,
            "start_date": start_date,
            "end_date": end_date,
            "duration": duration,
            "exposure_time": exposure_time,
            "initial_capital": initial_capital,
            "equity_final": equity_final,
            "equity_peak": equity_peak,
            "return": return_percent,
            "buy_hold_return": buy_hold_return,
            "return_annualized": return_annualized,
            "volatility": volatility,
            "cagr": cagr,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "alpha": alpha,
            "beta": beta,
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "avg_drawdown_duration": avg_drawdown_duration,
            "trade_count": trade_count,
            "trades": trades,
            "win_rate": win_rate,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_trade": avg_trade,
            "max_trade_duration": max_trade_duration,
            "avg_trade_duration": avg_trade_duration,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "sqn": sqn,
            "kelly_criterion": kelly_criterion,
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown_curve,
            "trades_list": trades_list,
        }
        
        # Save complete analysis results to JSON
        if ticker:
            analysis_log = f"logs/analysis_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(analysis_log), exist_ok=True)
            
            # Create a serializable version of the results
            serializable_results = results.copy()
            
            # Limit the size of lists for JSON serialization
            if equity_curve and len(equity_curve) > 100:
                # Sample the equity curve to reduce size
                sample_rate = max(1, len(equity_curve) // 100)
                serializable_results['equity_curve'] = equity_curve[::sample_rate]
                logger.info(f"Sampled equity curve from {len(equity_curve)} to {len(serializable_results['equity_curve'])} points for serialization")
            
            if drawdown_curve and len(drawdown_curve) > 100:
                # Sample the drawdown curve to reduce size
                sample_rate = max(1, len(drawdown_curve) // 100)
                serializable_results['drawdown_curve'] = drawdown_curve[::sample_rate]
                logger.info(f"Sampled drawdown curve from {len(drawdown_curve)} to {len(serializable_results['drawdown_curve'])} points for serialization")
            
            # Limit trades list to first 1000 trades if very large
            if trades_list and len(trades_list) > 1000:
                serializable_results['trades_list'] = trades_list[:1000]
                logger.info(f"Limited trades list from {len(trades_list)} to 1000 for serialization")
                
            try:
                with open(analysis_log, 'w') as f:
                    json.dump(serializable_results, f, indent=2, default=str)
                logger.info(f"Complete analysis results saved to {analysis_log}")
            except Exception as e:
                logger.error(f"Failed to save analysis results to JSON: {e}")
        
        logger.info("Backtest analysis completed successfully")
        return results

    @staticmethod
    def _extract_equity_curve(results):
        """Extract equity curve data from results."""
        logger.debug("Extracting equity curve from backtest results")
        
        if "_equity_curve" not in results:
            logger.warning("No equity curve data found in backtest results")
            return []
            
        equity_data = results["_equity_curve"]
        equity_curve = []
        
        try:
            # Handle different equity curve data structures
            if isinstance(equity_data, pd.DataFrame):
                logger.debug(f"Processing DataFrame equity curve with {len(equity_data)} rows")
                
                # Check if 'Equity' column exists
                if 'Equity' in equity_data.columns:
                    for date, value in zip(equity_data.index, equity_data['Equity']):
                        equity_curve.append({
                            "date": str(date),
                            "value": float(value) if not pd.isna(value) else 0.0
                        })
                else:
                    # Try to find the equity column or use the first column
                    logger.debug(f"Equity column not found. Available columns: {equity_data.columns}")
                    for date, row in equity_data.iterrows():
                        val = row.iloc[0] if isinstance(row, pd.Series) and len(row) > 0 else row
                        equity_curve.append({
                            "date": str(date),
                            "value": float(val) if not pd.isna(val) else 0.0
                        })
            else:
                # Handle case where equity curve is a Series
                logger.debug(f"Processing Series equity curve with {len(equity_data)} elements")
                for date, val in zip(equity_data.index, equity_data.values):
                    # Handle numpy values
                    if hasattr(val, "item"):
                        try:
                            val = val.item()
                        except (ValueError, TypeError):
                            val = val[0] if len(val) > 0 else 0
                    
                    equity_curve.append({
                        "date": str(date),
                        "value": float(val) if not pd.isna(val) else 0.0
                    })
                    
            logger.debug(f"Extracted {len(equity_curve)} equity curve points")
            
            # Verify data quality
            if equity_curve:
                min_value = min(point["value"] for point in equity_curve)
                max_value = max(point["value"] for point in equity_curve)
                logger.debug(f"Equity curve range: {min_value} to {max_value}")
                
                # Check for suspicious values
                if min_value < 0:
                    logger.warning(f"Negative values detected in equity curve: minimum = {min_value}")
                if max_value == 0:
                    logger.warning("All equity curve values are zero")
                    
            return equity_curve
            
        except Exception as e:
            logger.error(f"Error extracting equity curve: {e}")
            return []

    @staticmethod
    def _extract_drawdown_curve(results):
        """Extract drawdown curve data from results."""
        logger.debug("Extracting drawdown curve from backtest results")
        
        if "_equity_curve" not in results:
            logger.warning("No equity curve data found for drawdown calculation")
            return []
            
        equity_data = results["_equity_curve"]
        drawdown_curve = []
        
        try:
            # Calculate drawdown from equity curve
            if isinstance(equity_data, pd.DataFrame):
                logger.debug(f"Processing DataFrame for drawdown with {len(equity_data)} rows")
                
                # Check if 'DrawdownPct' column exists
                if 'DrawdownPct' in equity_data.columns:
                    for date, value in zip(equity_data.index, equity_data['DrawdownPct']):
                        drawdown_curve.append({
                            "date": str(date),
                            "value": float(value) if not pd.isna(value) else 0.0
                        })
                else:
                    # Calculate drawdown from equity
                    equity_col = 'Equity' if 'Equity' in equity_data.columns else equity_data.columns[0]
                    equity_series = equity_data[equity_col]
                    
                    # Calculate running maximum
                    running_max = equity_series.cummax()
                    
                    # Calculate drawdown percentage
                    drawdown_pct = (equity_series / running_max - 1) * 100
                    
                    for date, value in zip(drawdown_pct.index, drawdown_pct.values):
                        drawdown_curve.append({
                            "date": str(date),
                            "value": float(value) if not pd.isna(value) else 0.0
                        })
            else:
                # Handle case where equity curve is a Series
                logger.debug(f"Processing Series for drawdown with {len(equity_data)} elements")
                equity_series = pd.Series(equity_data.values, index=equity_data.index)
                
                # Calculate running maximum
                running_max = equity_series.cummax()
                
                # Calculate drawdown percentage
                drawdown_pct = (equity_series / running_max - 1) * 100
                
                for date, value in zip(drawdown_pct.index, drawdown_pct.values):
                    drawdown_curve.append({
                        "date": str(date),
                        "value": float(value) if not pd.isna(value) else 0.0
                    })
                    
            logger.debug(f"Extracted {len(drawdown_curve)} drawdown curve points")
            
            # Verify data quality
            if drawdown_curve:
                min_value = min(point["value"] for point in drawdown_curve)
                max_value = max(point["value"] for point in drawdown_curve)
                logger.debug(f"Drawdown curve range: {min_value}% to {max_value}%")
                
                # Check for suspicious values
                if min_value > 0:
                    logger.warning(f"Positive drawdown values detected: minimum = {min_value}%")
                if max_value < -100:
                    logger.warning(f"Extreme drawdown values detected: maximum = {max_value}%")
                    
            return drawdown_curve
            
        except Exception as e:
            logger.error(f"Error extracting drawdown curve: {e}")
            return []

    @staticmethod
    def _extract_trades_list(results):
        """Extract list of trades from results."""
        logger.debug("Extracting trades list from backtest results")
        
        if "_trades" not in results or results["_trades"] is None or results["_trades"].empty:
            logger.warning("No trades data found in backtest results")
            return []
            
        trades_df = results["_trades"]
        trades_list = []
        
        try:
            logger.debug(f"Processing trades DataFrame with {len(trades_df)} rows")
            logger.debug(f"Trades DataFrame columns: {list(trades_df.columns)}")
            
            # Map common column names from backtesting.py
            column_mapping = {
                'EntryTime': 'entry_date',
                'ExitTime': 'exit_date',
                'EntryPrice': 'entry_price',
                'ExitPrice': 'exit_price',
                'Size': 'size',
                'PnL': 'pnl',
                'ReturnPct': 'return_pct',
                'Duration': 'duration'
            }
            
            # Check if we have the expected columns
            missing_columns = [col for col in column_mapping.keys() if col not in trades_df.columns]
            if missing_columns:
                logger.warning(f"Missing expected columns in trades data: {missing_columns}")
            
            for _, trade in trades_df.iterrows():
                trade_dict = {}
                
                # Extract standard fields with error handling
                for orig_col, new_col in column_mapping.items():
                    if orig_col in trade:
                        try:
                            value = trade[orig_col]
                            
                            # Handle different data types
                            if orig_col in ['EntryTime', 'ExitTime']:
                                trade_dict[new_col] = str(value)
                            elif orig_col == 'ReturnPct':
                                # Convert decimal to percentage
                                trade_dict[new_col] = float(value) * 100
                            elif orig_col == 'Size':
                                trade_dict[new_col] = int(value)
                            else:
                                trade_dict[new_col] = float(value)
                        except Exception as e:
                            logger.warning(f"Error processing trade column {orig_col}: {e}")
                            trade_dict[new_col] = None
                
                # Add trade direction (assuming long trades by default)
                trade_dict['type'] = 'LONG'
                
                # Calculate trade duration if not provided
                if 'duration' not in trade_dict and 'entry_date' in trade_dict and 'exit_date' in trade_dict:
                    try:
                        entry = pd.to_datetime(trade_dict['entry_date'])
                        exit = pd.to_datetime(trade_dict['exit_date'])
                        trade_dict['duration'] = str(exit - entry)
                    except Exception as e:
                        logger.warning(f"Could not calculate trade duration: {e}")
                        trade_dict['duration'] = 'N/A'
                
                trades_list.append(trade_dict)
            
            logger.debug(f"Extracted {len(trades_list)} trades")
            
            # Verify data quality
            if trades_list:
                total_pnl = sum(trade.get('pnl', 0) for trade in trades_list)
                avg_return = sum(trade.get('return_pct', 0) for trade in trades_list) / len(trades_list)
                
                logger.debug(f"Total P&L from trades: ${total_pnl:.2f}")
                logger.debug(f"Average return per trade: {avg_return:.2f}%")
                
                # Check for suspicious values
                if total_pnl == 0 and len(trades_list) > 5:
                    logger.warning("All trades have zero P&L, which is suspicious")
                
                # Check for consistency with overall results
                if 'Return [%]' in results and abs(results.get('Return [%]', 0)) > 0.01:
                    strategy_return = results.get('Return [%]', 0)
                    total_return = sum(trade.get('return_pct', 0) for trade in trades_list)
                    
                    # If there's a significant discrepancy, log a warning
                    if abs(strategy_return - total_return) > max(5, strategy_return * 0.1):
                        logger.warning(f"Discrepancy between strategy return ({strategy_return}%) and sum of trade returns ({total_return}%)")
                
            return trades_list
            
        except Exception as e:
            logger.error(f"Error extracting trades list: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
