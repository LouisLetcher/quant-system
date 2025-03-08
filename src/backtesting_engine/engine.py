from backtesting import Backtest
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class BacktestEngine:
    def __init__(self, strategy_class, data, cash=10000, commission=0.001, ticker=None, is_portfolio=False):
        """Initialize the backtesting engine with a strategy and data.
        
        Args:
            strategy_class: The strategy class to use for backtesting
            data: DataFrame containing OHLCV data or dict of DataFrames for portfolio
            cash: Initial cash amount
            commission: Commission rate or dict of commission rates
            ticker: Stock ticker symbol or portfolio name
            is_portfolio: Whether this is a portfolio backtest
        """
        self.is_portfolio = is_portfolio
        self.portfolio_results = {}
        
        if is_portfolio:
            if not isinstance(data, dict):
                raise ValueError("❌ Portfolio data must be a dictionary")
            
            for ticker, df in data.items():
                if df.empty or not isinstance(df, pd.DataFrame):
                    raise ValueError(f"❌ Data for {ticker} must be a non-empty DataFrame")
                
                # Print actual columns to debug
                print(f"Debug - {ticker} data columns: {list(df.columns)}")
                
                # Check if we have MultiIndex columns (column, ticker) format from yfinance
                has_multiindex = isinstance(df.columns, pd.MultiIndex)
                
                # Handle MultiIndex columns from yfinance
                if has_multiindex:
                    print(f"Debug - Detected MultiIndex columns for {ticker}")
                    # Extract the first level of the MultiIndex (column names)
                    level0_columns = [col[0] for col in df.columns]
                    # Create a new DataFrame with simplified column names
                    new_df = pd.DataFrame()
                    
                    # Map required columns
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_columns:
                        # Find matching column ignoring case
                        matches = [c for c in level0_columns if c.lower() == col.lower()]
                        if not matches:
                            raise ValueError(f"❌ Data for {ticker} missing required column: {col}")
                        # Use the first match
                        original_col = df.columns[level0_columns.index(matches[0])]
                        new_df[col] = df[original_col]
                    
                    # Add any additional columns if needed
                    for i, col in enumerate(level0_columns):
                        if col not in [c for c in required_columns] and col not in ['Adj Close']:
                            original_col = df.columns[i]
                            new_df[col] = df[original_col]
                    
                    new_df.name = ticker
                    data[ticker] = new_df
                else:
                    # Original code for non-MultiIndex columns
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df_cols_lower = [str(c).lower() for c in df.columns]
                    
                    missing_columns = []
                    for col in required_columns:
                        if col.lower() not in df_cols_lower:
                            missing_columns.append(col)
                    
                    if missing_columns:
                        raise ValueError(f"❌ Data for {ticker} missing required columns: {', '.join(missing_columns)}")
                    
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
            
            self.data = data
            self.commission = commission if isinstance(commission, dict) else {t: commission for t in data.keys()}
            self.cash = cash if isinstance(cash, dict) else {t: cash / len(data) for t in data.keys()}
        else:
            if data.empty:
                raise ValueError("❌ Data must be a non-empty Pandas DataFrame")
            
            # Handle potential MultiIndex from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                print("Debug - Detected MultiIndex columns in single-asset data")
                level0_columns = [col[0] for col in data.columns]
                new_data = pd.DataFrame()
                
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    matches = [c for c in level0_columns if c.lower() == col.lower()]
                    if not matches:
                        raise ValueError(f"❌ Data missing required column: {col}")
                    original_col = data.columns[level0_columns.index(matches[0])]
                    new_data[col] = data[original_col]
                
                # Add additional columns if needed
                for i, col in enumerate(level0_columns):
                    if col not in [c for c in required_columns] and col not in ['Adj Close']:
                        original_col = data.columns[i]
                        new_data[col] = data[original_col]
                
                self.data = new_data
            else:
                # Ensure DataFrame has required columns 
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                # Convert all column names to lowercase for case-insensitive comparison
                df_cols_lower = [str(c).lower() for c in data.columns]
                
                missing_columns = []
                for col in required_columns:
                    if col.lower() not in df_cols_lower:
                        missing_columns.append(col)
                
                if missing_columns:
                    raise ValueError(f"❌ Data missing required columns: {', '.join(missing_columns)}")
                
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
            
            self.cash = cash
            self.commission = commission
            
            # Set the ticker name on the data
            if ticker:
                self.data.name = ticker
        
        self.strategy_class = strategy_class
        self.ticker = ticker
        
        if not is_portfolio:
            self.backtest = Backtest(
                data=self.data,
                strategy=self.strategy_class,
                cash=self.cash,
                commission=self.commission,
                trade_on_close=True
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
                    trade_on_close=True
                )
                result = bt.run()
                return ticker, result
            
            # Run backtests in parallel
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(
                    lambda item: run_single_backtest(item[0]), 
                    self.data.items()
                ))
            
            self.portfolio_results = {ticker: result for ticker, result in results}
            
            # Aggregate portfolio results
            total_final_equity = sum(r['Equity Final [$]'] for r in self.portfolio_results.values())
            total_initial_equity = sum(self.cash.values())
            
            # Create a combined result object
            combined_result = {
                '_portfolio': True,
                '_assets': list(self.portfolio_results.keys()),
                'Equity Final [$]': total_final_equity,
                'Return [%]': ((total_final_equity / total_initial_equity) - 1) * 100,
                '# Trades': sum(r['# Trades'] for r in self.portfolio_results.values()),
                'Sharpe Ratio': sum(r['Sharpe Ratio'] * r['Equity Final [$]'] for r in self.portfolio_results.values()) / total_final_equity,
                'Max. Drawdown [%]': max(r['Max. Drawdown [%]'] for r in self.portfolio_results.values()),
                '_strategy': self.strategy_class,
                'asset_results': self.portfolio_results
            }
            
            print(f"📊 Portfolio Backtest complete for {len(self.data)} assets")
            return combined_result
        else:
            print("🚀 Running Backtesting.py Engine...")
            results = self.backtest.run()
            
            if results is None:
                raise RuntimeError("❌ Backtesting.py did not return any results.")
                
            # Add debug info about trades
            print(f"Debug - Raw trade count from backtest: {results.get('# Trades', 'Not found in results')}")
            print(f"📊 Backtest complete with {len(self.data)} data points")
            return results
