from __future__ import annotations

import json
import os

from src.backtesting_engine.data_loader import DataLoader
from src.backtesting_engine.engine import BacktestEngine
from src.backtesting_engine.result_analyzer import BacktestResultAnalyzer
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory


class StrategyRunner:
    """Executes a selected trading strategy for backtesting."""

    @staticmethod
    def execute(
        strategy_name,
        ticker,
        period="max",
        start=None,
        end=None,
        commission=0.001,
        initial_capital=10000,
        take_profit=None,
        stop_loss=None,
    ):
        """
        Loads data, runs a strategy, and analyzes the result.
        """
        if period and (start or end):
            print(
                "⚠️ Both period and start/end dates provided. Using period for data fetching."
            )

        # 🔍 Ensure strategy exists
        strategy_class = StrategyFactory.get_strategy(strategy_name)
        if strategy_class is None:
            raise ValueError(f"❌ Strategy '{strategy_name}' not found.")

        # Check if ticker is a portfolio from assets_config.json
        config_path = os.path.join("config", "assets_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                assets_config = json.load(f)

            if ticker in assets_config.get("portfolios", {}):
                return StrategyRunner.execute_portfolio(
                    strategy_name=strategy_name,
                    portfolio_name=ticker,
                    portfolio_config=assets_config["portfolios"][ticker],
                    start=start,
                    end=end,
                )

        # Single asset execution
        # Print what we're loading
        if period:
            print(f"📥 Loading data for {ticker} with period={period}...")
        else:
            print(f"📥 Loading data for {ticker} from {start} to {end}...")

        # Load data using period parameter
        data = DataLoader.load_data(ticker, period=period, start=start, end=end)
        print(f"✅ Successfully loaded {len(data)} rows for {ticker}.")

        # 🚀 Initialize Backtrader engine with supported parameters
        print(f"🚀 Running Backtrader Engine for {ticker}...")

        # Create a custom strategy class that enforces position sizing limits
        strategy_class = StrategyFactory.get_strategy(strategy_name)

        # Create a subclass that enforces position sizing
        class SizeLimitedStrategy(strategy_class):
            def init(self):
                # Store initial capital for reference
                self._initial_capital = initial_capital
                # Call the original init method
                super().init()

            def buy(self, *args, **kwargs):
                # If size is not specified, calculate it based on available cash
                if "size" not in kwargs:
                    price = self.data.Close[-1]
                    # Limit size to initial capital
                    max_size = min(self.equity, self._initial_capital) / price
                    kwargs["size"] = int(max_size)  # Ensure whole number of shares
                else:
                    # If size is specified, ensure it doesn't exceed initial capital
                    price = self.data.Close[-1]
                    max_size = self._initial_capital / price
                    kwargs["size"] = min(int(kwargs["size"]), int(max_size))

                return super().buy(*args, **kwargs)

        # Use the modified strategy class instead of the original
        engine = BacktestEngine(
            SizeLimitedStrategy,
            data,
            ticker=ticker,
            commission=commission,
            cash=initial_capital,
        )

        engine.params = {
            "initial_capital": initial_capital,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        }

        if not engine or engine is None:
            raise RuntimeError("❌ Engine failed to initialize.")

        # ✅ Run backtest
        results = engine.run()

        # 🔍 Ensure results exist before proceeding
        if results is None:
            raise RuntimeError("❌ No results returned from Backtest Engine.")

        # Add debugging to help troubleshoot
        print(f"Debug - Raw backtest results type: {type(results)}")
        print(
            f"Debug - Available metrics: {[k for k in results.keys() if not k.startswith('_')]}"
        )
        print(
            f"Debug - Trade count from raw results: {results.get('# Trades', 'Not found')}"
        )

        print("📊 Strategy finished. Analyzing results...")
        analyzed_results = BacktestResultAnalyzer.analyze(
            results, ticker=ticker, initial_capital=initial_capital
        )

        # Add the additional parameters to the results
        analyzed_results.update(
            {"initial_capital": initial_capital, "commission": commission}
        )

        if take_profit:
            analyzed_results["take_profit"] = take_profit
        if stop_loss:
            analyzed_results["stop_loss"] = stop_loss

        if not isinstance(analyzed_results, dict):
            raise TypeError(
                f"❌ Expected results in dict format, got {type(analyzed_results)}."
            )

        if "profit_factor" in results:
            analyzed_results["profit_factor"] = results["profit_factor"]
        elif "_trades" in results and not results["_trades"].empty:
            # Re-calculate profit factor from trade data
            trades = results["_trades"]
            gross_profit = trades[trades["PnL"] > 0]["PnL"].sum()
            gross_loss = abs(trades[trades["PnL"] < 0]["PnL"].sum())
            profit_factor = (
                float("inf") if gross_loss == 0 else gross_profit / gross_loss
            )
            analyzed_results["profit_factor"] = profit_factor

        print(f"✅ Backtest Complete! Results: {analyzed_results}")

        return analyzed_results

    @staticmethod
    def execute_portfolio(
        strategy_name, portfolio_name, portfolio_config, start=None, end=None
    ):
        """
        Execute a strategy on a portfolio of assets defined in assets_config.json
        """
        # Get strategy class
        strategy_class = StrategyFactory.get_strategy(strategy_name)
        if strategy_class is None:
            raise ValueError(f"❌ Strategy '{strategy_name}' not found.")

        print(
            f"📂 Running portfolio backtest for '{portfolio_name}' with {len(portfolio_config['assets'])} assets..."
        )

        # Extract initial capital from portfolio config
        initial_capital = portfolio_config.get("initial_capital", 10000)

        # Load data for each asset in the portfolio
        portfolio_data = {}
        commission_rates = {}

        for asset in portfolio_config["assets"]:
            ticker = asset["ticker"]
            period = asset.get("period", "max")
            commission = asset.get("commission", 0.001)

            print(f"📥 Loading data for {ticker} with period={period}...")

            # Use period if provided, otherwise use start/end dates
            if period and not (start or end):
                data = DataLoader.load_data(ticker, period=period)
            else:
                data = DataLoader.load_data(ticker, start=start, end=end)

            print(f"✅ Successfully loaded {len(data)} rows for {ticker}.")

            portfolio_data[ticker] = data
            commission_rates[ticker] = commission

        # Create a subclass that enforces position sizing
        class SizeLimitedStrategy(strategy_class):
            def init(self):
                # Store initial capital for reference
                self._initial_capital = initial_capital / len(
                    portfolio_data
                )  # Per asset
                # Call the original init method
                super().init()

            def buy(self, *args, **kwargs):
                # If size is not specified, calculate it based on available cash
                if "size" not in kwargs:
                    price = self.data.Close[-1]
                    # Limit size to initial capital
                    max_size = min(self.equity, self._initial_capital) / price
                    kwargs["size"] = int(max_size)  # Ensure whole number of shares
                else:
                    # If size is specified, ensure it doesn't exceed initial capital
                    price = self.data.Close[-1]
                    max_size = self._initial_capital / price
                    kwargs["size"] = min(int(kwargs["size"]), int(max_size))

                return super().buy(*args, **kwargs)

        # Initialize portfolio backtest engine
        print(f"🚀 Initializing portfolio backtest for {portfolio_name}...")

        # Use the modified strategy class
        engine = BacktestEngine(
            SizeLimitedStrategy,
            portfolio_data,
            cash=initial_capital,
            commission=commission_rates,
            ticker=portfolio_name,
            is_portfolio=True,
        )

        # Run the portfolio backtest
        results = engine.run()

        # 🔍 Ensure results exist before proceeding
        if results is None:
            raise RuntimeError("❌ No results returned from Portfolio Backtest Engine.")

        print("📊 Portfolio strategy finished. Analyzing results...")
        analyzed_results = BacktestResultAnalyzer.analyze(
            results, ticker=portfolio_name, initial_capital=initial_capital
        )

        # Add portfolio metadata
        analyzed_results.update(
            {
                "portfolio_name": portfolio_name,
                "portfolio_description": portfolio_config.get("description", ""),
                "asset_count": len(portfolio_config["assets"]),
                "initial_capital": initial_capital,
            }
        )

        print(
            f"✅ Portfolio Backtest Complete! Overall Return: {analyzed_results['return_pct']}"
        )

        return analyzed_results

    @staticmethod
    def optimize(
        strategy_name,
        ticker,
        param_space,
        metric="sharpe",
        period="max",
        iterations=50,
        initial_capital=10000,
        commission=0.001,
    ):
        """
        Optimizes strategy parameters using Bayesian optimization.

        Args:
            strategy_name: Name of the strategy to optimize
            ticker: Stock ticker symbol
            param_space: Dictionary of parameter ranges
            metric: Metric to optimize ('sharpe', 'return', etc.)
            period: Data period
            iterations: Number of optimization iterations
            initial_capital: Initial capital amount
            commission: Commission rate
        """
        from src.optimizer.optimization_runner import OptimizationRunner

        print(f"🔍 Optimizing {strategy_name} for {ticker} using {metric} metric...")

        # Load data
        data = DataLoader.load_data(ticker, period=period)

        # Get strategy class
        strategy_class = StrategyFactory.get_strategy(strategy_name)
        if strategy_class is None:
            raise ValueError(f"❌ Strategy '{strategy_name}' not found.")

        # Run optimization
        optimizer = OptimizationRunner(strategy_class, data, param_space)
        results = optimizer.run(
            metric=metric,
            iterations=iterations,
            initial_capital=initial_capital,
            commission=commission,
        )

        print(f"✅ Optimization complete. Best parameters: {results['best_params']}")
        print(f"   Best {metric} score: {results['best_score']:.4f}")

        return results

    @staticmethod
    def execute_multi_timeframe(
        strategy_name,
        ticker,
        timeframes=None,
        commission=0.001,
        initial_capital=10000,
        take_profit=None,
        stop_loss=None,
    ):
        """
        Tests a strategy across multiple timeframes to find the optimal period.

        Args:
            strategy_name: Name of the strategy to run
            ticker: Stock ticker symbol
            timeframes: List of timeframes to test (e.g., ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
            commission: Commission rate
            initial_capital: Initial capital amount
            take_profit: Take profit percentage
            stop_loss: Stop loss percentage

        Returns:
            Dictionary with results for each timeframe and the best timeframe
        """
        if timeframes is None:
            # Standard Timeframes von kurzfristig nach langfristig
            timeframes = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]

        # 🔍 Ensure strategy exists
        strategy_class = StrategyFactory.get_strategy(strategy_name)
        if strategy_class is None:
            raise ValueError(f"❌ Strategy '{strategy_name}' not found.")

        print(
            f"🔎 Testing {strategy_name} on {ticker} across {len(timeframes)} timeframes..."
        )

        results = {}
        best_score = -float("inf")
        best_timeframe = None
        best_result = None

        for period in timeframes:
            print(f"  ⏱️ Testing timeframe: {period}")

            try:
                # Run backtest for this timeframe
                result = StrategyRunner.execute(
                    strategy_name,
                    ticker,
                    period=period,
                    commission=commission,
                    initial_capital=initial_capital,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                )

                # Store result
                results[period] = result

                # Evaluate performance (default to Sharpe ratio)
                score = result.get("profit_factor", 0)
                trade_count = result.get("trades", result.get("# Trades", 0))

                # Only consider valid results with trades
                if score > best_score and trade_count > 0:
                    best_score = score
                    best_timeframe = period
                    best_result = result

                print(
                    f"    {period}: Profit Factor = {score}, Sharpe = {result.get('sharpe_ratio', 0)}, Trades = {trade_count}"
                )

            except Exception as e:
                print(f"    ❌ Error testing {period}: {e!s}")
                results[period] = {"error": str(e)}

        # If no valid results with trades found, just pick the best score
        if best_timeframe is None and results:
            for period, result in results.items():
                if isinstance(result, dict) and "sharpe_ratio" in result:
                    score = result.get("sharpe_ratio", 0)
                    if score > best_score:
                        best_score = score
                        best_timeframe = period
                        best_result = result

        print(
            f"✅ Best timeframe for {strategy_name} on {ticker}: {best_timeframe} (Sharpe: {best_score})"
        )

        return {
            "all_results": results,
            "best_timeframe": best_timeframe,
            "best_result": best_result,
            "strategy": strategy_name,
            "ticker": ticker,
        }
