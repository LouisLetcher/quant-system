from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from backtesting import Backtest

from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class BacktestEngine:
    def __init__(
        self,
        strategy_class,
        data,
        cash=10000,
        commission=0.001,
        ticker=None,
        is_portfolio=False,
    ):
        """Initialize the backtesting engine with a strategy and data.

        Args:
            strategy_class: The strategy class to use for backtesting
            data: DataFrame containing OHLCV data or dict of DataFrames for portfolio
            cash: Initial cash amount
            commission: Commission rate or dict of commission rates
            ticker: Stock ticker symbol or portfolio name
            is_portfolio: Whether this is a portfolio backtest
        """
        logger.info(f"Initializing BacktestEngine for {'portfolio' if is_portfolio else ticker}")
        logger.info(f"Strategy: {strategy_class.__name__}, Initial cash: {cash}, Commission: {commission}")
        
        self.is_portfolio = is_portfolio
        self.portfolio_results = {}

        if is_portfolio:
            if not isinstance(data, dict):
                error_msg = "❌ Portfolio data must be a dictionary"
                logger.error(error_msg)
                raise ValueError(error_msg)

            for ticker, df in data.items():
                if df.empty or not isinstance(df, pd.DataFrame):
                    error_msg = f"❌ Data for {ticker} must be a non-empty DataFrame"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Print actual columns to debug
                logger.debug(f"{ticker} data columns: {list(df.columns)}")

                # Check if we have MultiIndex columns (column, ticker) format from yfinance
                has_multiindex = isinstance(df.columns, pd.MultiIndex)

                # Handle MultiIndex columns from yfinance
                if has_multiindex:
                    logger.debug(f"Detected MultiIndex columns for {ticker}")
                    # Extract the first level of the MultiIndex (column names)
                    level0_columns = [col[0] for col in df.columns]
                    # Create a new DataFrame with simplified column names
                    new_df = pd.DataFrame()
                    # Map required columns
                    required_columns = ["Open", "High", "Low", "Close", "Volume"]
                    for col in required_columns:
                        # Find matching column ignoring case - with type safety
                        found = False
                        for original_col in df.columns:
                            col_name = original_col[0] if isinstance(original_col, tuple) else original_col
                            if str(col_name).lower() == col.lower():
                                new_df[col] = df[original_col]
                                found = True
                                break
                            
                        if not found:
                            error_msg = f"❌ Data for {ticker} missing required column: {col}"
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                        
                    # Add any additional columns if needed
                    for i, col in enumerate(level0_columns):
                        if col not in [c for c in required_columns] and col not in [
                            "Adj Close"
                        ]:
                            original_col = df.columns[i]
                            new_df[col] = df[original_col]

                    new_df.name = ticker
                    data[ticker] = new_df
                    logger.debug(f"Processed {ticker} data: {len(new_df)} rows, columns: {list(new_df.columns)}")
                else:
                    # Original code for non-MultiIndex columns
                    required_columns = ["Open", "High", "Low", "Close", "Volume"]
                    df_cols_lower = [str(c).lower() for c in df.columns]

                    missing_columns = []
                    for col in required_columns:
                        if col.lower() not in df_cols_lower:
                            missing_columns.append(col)

                    if missing_columns:
                        error_msg = f"❌ Data for {ticker} missing required columns: {', '.join(missing_columns)}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    # Create a mapping of lowercase column names to their actual column names
                    column_map = {c.lower(): c for c in df.columns}

                    # Create a new standardized DataFrame with properly capitalized columns
                    new_df = pd.DataFrame()
                    for col in required_columns:
                        actual_col = column_map[col.lower()]
                        new_df[col] = df[actual_col]

                    # Add any additional columns that might be needed
                    for col in df.columns:
                        if col.lower() not in [rc.lower() for rc in required_columns]:
                            new_df[col] = df[col]

                    new_df.name = ticker
                    data[ticker] = new_df
                    logger.debug(f"Processed {ticker} data: {len(new_df)} rows, columns: {list(new_df.columns)}")

            self.data = data
            self.commission = (
                commission
                if isinstance(commission, dict)
                else {t: commission for t in data.keys()}
            )
            self.cash = (
                cash
                if isinstance(cash, dict)
                else {t: cash / len(data) for t in data.keys()}
            )
        else:
            if data.empty:
                error_msg = "❌ Data must be a non-empty Pandas DataFrame"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Handle potential MultiIndex from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                logger.debug("Detected MultiIndex columns in single-asset data")
                level0_columns = [col[0] for col in data.columns]
                new_data = pd.DataFrame()

                required_columns = ["Open", "High", "Low", "Close", "Volume"]
                for col in required_columns:
                    matches = []
                    for c in level0_columns:
                        # Handle different types of column identifiers
                        if isinstance(c, str):
                            c_str = c
                        elif isinstance(c, tuple):
                            c_str = c[0] if len(c) > 0 else str(c)
                        else:
                            c_str = str(c)
            
                        if c_str.lower() == col.lower():
                            matches.append(c)
                    if not matches:
                        error_msg = f"❌ Data missing required column: {col}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    original_col = data.columns[level0_columns.index(matches[0])]
                    new_data[col] = data[original_col]
                
                    # Extra verification to ensure data isn't empty after column extraction
                    if new_data[col].empty or new_data[col].isna().all():
                        logger.warning(f"Column {col} has no valid data after extraction")
            
                # Add additional columns if needed
                for i, col in enumerate(level0_columns):
                    if col not in [c.lower() for c in required_columns] and col not in ["Adj Close"]:
                        original_col = data.columns[i]
                        new_data[col] = data[original_col]

                # Preserve ticker name
                if hasattr(data, 'name'):
                    new_data.name = data.name
            
                # Add a final verification step
                if new_data.empty or new_data['Close'].isna().all():
                    logger.critical("Processed data has no valid entries")
                    logger.debug(f"Original data shape: {data.shape}, New data shape: {new_data.shape}")
                    # Print first few rows for debugging
                    logger.debug(f"Original data head:\n{data.head()}")
                    logger.debug(f"New data head:\n{new_data.head()}")
            
                self.data = new_data
                logger.debug(f"Processed data: {len(new_data)} rows, columns: {list(new_data.columns)}")
            else:
                # Ensure DataFrame has required columns
                required_columns = ["Open", "High", "Low", "Close", "Volume"]

                # Convert all column names to lowercase for case-insensitive comparison
                df_cols_lower = [str(c).lower() for c in data.columns]

                missing_columns = []
                for col in required_columns:
                    if col.lower() not in df_cols_lower:
                        missing_columns.append(col)

                if missing_columns:
                    error_msg = f"❌ Data missing required columns: {', '.join(missing_columns)}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Create a mapping of lowercase column names to their actual column names
                column_map = {c.lower(): c for c in data.columns}

                # Create a new standardized DataFrame with properly capitalized columns
                new_data = pd.DataFrame()
                for col in required_columns:
                    actual_col = column_map[col.lower()]
                    new_data[col] = data[actual_col]

                # Add any additional columns that might be needed
                for col in data.columns:
                    if col.lower() not in [rc.lower() for rc in required_columns]:
                        new_data[col] = data[col]

                self.data = new_data
                logger.debug(f"Processed data: {len(new_data)} rows, columns: {list(new_data.columns)}")

            self.cash = cash
            self.commission = commission

            # Set the ticker name on the data
            if ticker:
                self.data.name = ticker
                logger.info(f"Set ticker name: {ticker}")

        self.strategy_class = strategy_class
        self.ticker = ticker

        if not is_portfolio:
            logger.info(f"Creating Backtest instance for {ticker}")
            self.backtest = Backtest(
                data=self.data,
                strategy=self.strategy_class,
                cash=self.cash,
                commission=self.commission,
                trade_on_close=True,
                hedging=False,
                exclusive_orders=True,
            )

    def run(self):
        """Runs the backtest and returns results."""
        if self.is_portfolio:
            print(f"🚀 Running Portfolio Backtesting for {len(self.data)} assets...")

            def run_single_backtest(ticker):
                bt = Backtest(
                    data=self.data[ticker],
                    strategy=self.strategy_class,
                    cash=self.cash[ticker],
                    commission=self.commission[ticker],
                    trade_on_close=True,
                )
                result = bt.run()
                return ticker, result

            # Run backtests in parallel
            with ThreadPoolExecutor() as executor:
                results = list(
                    executor.map(
                        lambda item: run_single_backtest(item[0]), self.data.items()
                    )
                )

            self.portfolio_results = {ticker: result for ticker, result in results}

            # Aggregate portfolio results
            total_final_equity = sum(
                r["Equity Final [$]"] for r in self.portfolio_results.values()
            )
            total_initial_equity = sum(self.cash.values())

            # Create a combined result object
            combined_result = {
                "_portfolio": True,
                "_assets": list(self.portfolio_results.keys()),
                "Equity Final [$]": total_final_equity,
                "Return [%]": ((total_final_equity / total_initial_equity) - 1) * 100,
                "# Trades": sum(r["# Trades"] for r in self.portfolio_results.values()),
                "Sharpe Ratio": sum(
                    r["Sharpe Ratio"] * r["Equity Final [$]"]
                    for r in self.portfolio_results.values()
                )
                / total_final_equity,
                "Max. Drawdown [%]": max(
                    r["Max. Drawdown [%]"] for r in self.portfolio_results.values()
                ),
                "_strategy": self.strategy_class,
                "asset_results": self.portfolio_results,
            }

            print(f"📊 Portfolio Backtest complete for {len(self.data)} assets")
            print(
                f"Debug - Raw profit factor: {combined_result.get('Profit Factor', 0)}"
            )

            return combined_result
    
        # This line should be at the same indentation level as the if statement above
        print("🚀 Running Backtesting.py Engine...")
        # Add logger statement at the same indentation level
        if hasattr(self, 'ticker') and self.ticker:
            logger.info(f"Running Backtesting.py Engine for {self.ticker}...")
        else:
            logger.info("Running Backtesting.py Engine...")
        
        results = self.backtest.run()

        if results is None:
            raise RuntimeError("❌ Backtesting.py did not return any results.")
        # Log all metrics from Backtesting.py's results
        logger.info("\n📊 BACKTEST RESULTS 📊")
        logger.info("=" * 50)

        # Time metrics
        logger.info(f"Start Date: {results.get('Start', 'N/A')}")
        logger.info(f"End Date: {results.get('End', 'N/A')}")
        logger.info(f"Duration: {results.get('Duration', 'N/A')}")
        logger.info(f"Exposure Time [%]: {results.get('Exposure Time [%]', 'N/A')}")

        # Equity and Return metrics
        logger.info(f"Equity Final [$]: {results.get('Equity Final [$]', 'N/A')}")
        logger.info(f"Equity Peak [$]: {results.get('Equity Peak [$]', 'N/A')}")
        logger.info(f"Return [%]: {results.get('Return [%]', 'N/A')}")
        logger.info(f"Buy & Hold Return [%]: {results.get('Buy & Hold Return [%]', 'N/A')}")
        logger.info(f"Return (Ann.) [%] : {results.get('Return (Ann.) [%]', 'N/A')}")
        
        # Risk metrics
        logger.info(f"Volatility (Ann.) [%]: {results.get('Volatility (Ann.) [%]', 'N/A')}")
        logger.info(f"CAGR [%]: {results.get('CAGR [%]', 'N/A')}")
        logger.info(f"Sharpe Ratio: {results.get('Sharpe Ratio', 'N/A')}")
        logger.info(f"Sortino Ratio: {results.get('Sortino Ratio', 'N/A')}")
        logger.info(f"Calmar Ratio: {results.get('Calmar Ratio', 'N/A')}")
        logger.info(f"Alpha [%]: {results.get('Alpha [%]', 'N/A')}")
        logger.info(f"Beta: {results.get('Beta', 'N/A')}")
        logger.info(f"Max. Drawdown [%]: {results.get('Max. Drawdown [%]', 'N/A')}")
        logger.info(f"Avg. Drawdown [%]: {results.get('Avg. Drawdown [%]', 'N/A')}")
        logger.info(f"Avg. Drawdown Duration: {results.get('Avg. Drawdown Duration', 'N/A')}")

        # Trade metrics
        logger.info(f"# Trades: {results.get('# Trades', 'N/A')}")
        logger.info(f"Win Rate [%]: {results.get('Win Rate [%]', 'N/A')}")
        logger.info(f"Best Trade [%]: {results.get('Best Trade [%]', 'N/A')}")
        logger.info(f"Worst Trade [%]: {results.get('Worst Trade [%]', 'N/A')}")
        logger.info(f"Avg. Trade [%]: {results.get('Avg. Trade [%]', 'N/A')}")
        logger.info(f"Max. Trade Duration: {results.get('Max. Trade Duration', 'N/A')}")
        logger.info(f"Avg. Trade Duration: {results.get('Avg. Trade Duration', 'N/A')}")

        # Performance metrics
        logger.info(f"Profit Factor: {results.get('Profit Factor', 'N/A')}")
        logger.info(f"Expectancy [%]: {results.get('Expectancy [%]', 'N/A')}")
        logger.info(f"SQN: {results.get('SQN', 'N/A')}")
        logger.info(f"Kelly Criterion: {results.get('Kelly Criterion', 'N/A')}")

        # Log additional data structures (summarized to prevent excessive output)
        if "_equity_curve" in results:
            logger.info(f"\nEquity Curve: DataFrame with {len(results['_equity_curve'])} rows")
            # Save equity curve to CSV for detailed analysis
            equity_curve_log = f"logs/equity_curve_{self.ticker}_{self.strategy_class.__name__}.csv"
            os.makedirs(os.path.dirname(equity_curve_log), exist_ok=True)
            results['_equity_curve'].to_csv(equity_curve_log)
            logger.info(f"Equity curve saved to {equity_curve_log}")

        if "_trades" in results and not results["_trades"].empty:
            trades_df = results["_trades"]
            logger.info(f"\nTrades: DataFrame with {len(trades_df)} trades")
            
            # Log summary statistics of trades
            if len(trades_df) > 0:
                winning_trades = trades_df[trades_df['PnL'] > 0]
                losing_trades = trades_df[trades_df['PnL'] < 0]
                
                logger.info(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.2f}%)")
                logger.info(f"Losing trades: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.2f}%)")
                
                if len(winning_trades) > 0:
                    logger.info(f"Average winning trade: ${winning_trades['PnL'].mean():.2f}")
                    logger.info(f"Largest winning trade: ${winning_trades['PnL'].max():.2f}")
                
                if len(losing_trades) > 0:
                    logger.info(f"Average losing trade: ${losing_trades['PnL'].mean():.2f}")
                    logger.info(f"Largest losing trade: ${losing_trades['PnL'].min():.2f}")
                
                logger.info(f"\nFirst trade sample:")
                logger.info(trades_df.iloc[0].to_string())
                
                # Save all trades to CSV for detailed analysis
                trades_log = f"logs/trades_{self.ticker}_{self.strategy_class.__name__}.csv"
                os.makedirs(os.path.dirname(trades_log), exist_ok=True)
                trades_df.to_csv(trades_log)
                logger.info(f"All trades saved to {trades_log}")
                
                # Log monthly/yearly performance if data spans multiple months/years
                if hasattr(trades_df, 'EntryTime') and len(trades_df) > 5:
                    try:
                        trades_df['EntryMonth'] = pd.to_datetime(trades_df['EntryTime']).dt.to_period('M')
                        monthly_pnl = trades_df.groupby('EntryMonth')['PnL'].sum()
                        
                        logger.info("\nMonthly P&L:")
                        logger.info(monthly_pnl.to_string())
                        
                        trades_df['EntryYear'] = pd.to_datetime(trades_df['EntryTime']).dt.to_period('Y')
                        yearly_pnl = trades_df.groupby('EntryYear')['PnL'].sum()
                        
                        logger.info("\nYearly P&L:")
                        logger.info(yearly_pnl.to_string())
                    except Exception as e:
                        logger.warning(f"Could not calculate periodic P&L: {e}")

        # Save full results to JSON
        results_log = f"logs/backtest_results_{self.ticker}_{self.strategy_class.__name__}.json"
        os.makedirs(os.path.dirname(results_log), exist_ok=True)
        
        # Create a serializable version of the results
        serializable_result = {k: v for k, v in results.items() 
        if not k.startswith('_') or k == '_trades' or k == '_equity_curve'}
        
        # Convert DataFrames to CSV strings for JSON serialization
        if '_trades' in serializable_result:
            trades_csv = f"logs/trades_{self.ticker}_{self.strategy_class.__name__}.csv"
            serializable_result['_trades'].to_csv(trades_csv)
            serializable_result['_trades'] = f"Saved to {trades_csv}"
            
        if '_equity_curve' in serializable_result:
            equity_csv = f"logs/equity_{self.ticker}_{self.strategy_class.__name__}.csv"
            serializable_result['_equity_curve'].to_csv(equity_csv)
            serializable_result['_equity_curve'] = f"Saved to {equity_csv}"
        
        with open(results_log, 'w') as f:
            json.dump(serializable_result, f, indent=2, default=str)
        logger.info(f"Full backtest results saved to {results_log}")

        logger.info("=" * 50)

        return results

    def get_optimization_metrics(self):
        """Returns metrics that can be used for optimization."""
        if self.is_portfolio:
            metrics = {
                "sharpe": sum(
                    r["Sharpe Ratio"] * r["Equity Final [$]"]
                    for r in self.portfolio_results.values()
                )
                / sum(r["Equity Final [$]"] for r in self.portfolio_results.values()),
                "return": (
                    (
                        sum(
                            r["Equity Final [$]"]
                            for r in self.portfolio_results.values()
                        )
                        / sum(self.cash.values())
                    )
                    - 1
                )
                * 100,
                "drawdown": max(
                    r["Max. Drawdown [%]"] for r in self.portfolio_results.values()
                ),
            }
            logger.info(f"Portfolio optimization metrics: {metrics}")
            return metrics
            
        metrics = {
            "sharpe": self.backtest.run()["Sharpe Ratio"],
            "return": self.backtest.run()["Return [%]"],
            "drawdown": self.backtest.run()["Max. Drawdown [%]"],
        }
        logger.info(f"Optimization metrics for {self.ticker}: {metrics}")
        return metrics

    def get_backtest_object(self):
        """Return the underlying Backtest object for direct plotting."""
        logger.debug(f"Returning backtest object for {self.ticker}")
        return self.backtest
