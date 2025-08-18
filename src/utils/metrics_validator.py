"""
Metrics Validation - Compare custom backtesting metrics with the backtesting library
to ensure accuracy and reliability of our calculations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from backtesting import Backtest
from backtesting.lib import SignalStrategy

from src.core.data_manager import DataManager
from src.core.result_analyzer import UnifiedResultAnalyzer
from src.database import get_db_session
from src.database.models import BestStrategy


class BacktestingLibStrategy(SignalStrategy):
    """Adapter to use our signals with the backtesting library."""

    def __init__(self, broker, data, params, signals_df):
        super().__init__(broker, data, params)
        self.signals_df = signals_df

    def init(self):
        """Initialize the strategy with signals."""
        # Align signals with the data index
        aligned_signals = self.signals_df.reindex(self.data.index, fill_value=0)
        self.signals = self.I(lambda: aligned_signals.values, name="signals")

    def next(self):
        """Execute trades based on signals."""
        if self.signals[-1] == 1 and not self.position:
            self.buy()
        elif self.signals[-1] == -1 and self.position:
            self.sell()


class MetricsValidator:
    """Validate custom metrics against the backtesting library."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_manager = DataManager()
        self.result_analyzer = UnifiedResultAnalyzer()

    def validate_best_strategy_metrics(
        self,
        symbol: str,
        strategy_name: str,
        timeframe: str = "1d",
        tolerance: float = 0.05,  # 5% tolerance for differences
    ) -> dict[str, Any]:
        """
        Validate metrics for a best strategy by comparing our calculations
        with the backtesting library results.
        """
        self.logger.info(
            "Validating metrics for %s/%s using backtesting library",
            symbol,
            strategy_name,
        )

        # Get the best strategy from database
        validation_results = {
            "symbol": symbol,
            "strategy": strategy_name,
            "timeframe": timeframe,
            "validation_passed": False,
            "differences": {},
            "our_metrics": {},
            "backtesting_lib_metrics": {},
            "errors": [],
        }

        try:
            # Load best strategy from database
            db_session = get_db_session()
            best_strategy = (
                db_session.query(BestStrategy)
                .filter_by(
                    symbol=symbol, best_strategy=strategy_name, timeframe=timeframe
                )
                .first()
            )

            if not best_strategy:
                validation_results["errors"].append(
                    f"No best strategy found for {symbol}/{strategy_name}"
                )
                return validation_results

            # Get historical data
            data_df = self._get_historical_data(symbol, timeframe)
            if data_df is None or data_df.empty:
                validation_results["errors"].append(f"No data available for {symbol}")
                return validation_results

            # Generate signals using our strategy
            signals_df = self._generate_strategy_signals(symbol, strategy_name, data_df)
            if signals_df is None or signals_df.empty:
                validation_results["errors"].append(
                    f"Failed to generate signals for {symbol}/{strategy_name}"
                )
                return validation_results

            # Run backtesting library validation
            lib_metrics = self._run_backtesting_lib(data_df, signals_df)

            # Our metrics from database
            our_metrics = {
                "total_return": float(best_strategy.total_return or 0),
                "sharpe_ratio": float(best_strategy.sharpe_ratio or 0),
                "sortino_ratio": float(best_strategy.sortino_ratio or 0),
                "max_drawdown": float(best_strategy.max_drawdown or 0),
                "win_rate": float(best_strategy.win_rate or 0),
                "num_trades": int(best_strategy.num_trades or 0),
            }

            validation_results["our_metrics"] = our_metrics
            validation_results["backtesting_lib_metrics"] = lib_metrics

            # Compare metrics
            differences = self._compare_metrics(our_metrics, lib_metrics, tolerance)
            validation_results["differences"] = differences

            # Overall validation
            validation_results["validation_passed"] = all(
                abs(diff) <= tolerance for diff in differences.values()
            )

            if validation_results["validation_passed"]:
                self.logger.info(
                    "✅ Metrics validation PASSED for %s/%s", symbol, strategy_name
                )
            else:
                self.logger.warning(
                    "❌ Metrics validation FAILED for %s/%s. Max difference: %.2f%%",
                    symbol,
                    strategy_name,
                    max(abs(d) for d in differences.values()) * 100,
                )

        except Exception as e:
            self.logger.error(
                "Error validating metrics for %s/%s: %s", symbol, strategy_name, e
            )
            validation_results["errors"].append(str(e))

        finally:
            db_session.close()

        return validation_results

    def _get_historical_data(
        self, symbol: str, timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Get historical data for the symbol."""
        try:
            # Use same date range as used in backtesting
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=730)  # 2 years of data

            data = self.data_manager.get_data(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                timeframe,
            )

            if data is None or data.empty:
                return None

            # Ensure required columns for backtesting library
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in data.columns for col in required_cols):
                self.logger.error("Missing required columns for backtesting library")
                return None

            return data

        except Exception as e:
            self.logger.error("Error getting historical data for %s: %s", symbol, e)
            return None

    def _generate_strategy_signals(
        self, symbol: str, strategy_name: str, data_df: pd.DataFrame
    ) -> Optional[pd.Series]:
        """Generate trading signals using our strategy implementation."""
        try:
            from src.core.external_strategy_loader import ExternalStrategyLoader

            loader = ExternalStrategyLoader()
            strategy_class = loader.get_strategy_class(strategy_name)

            if not strategy_class:
                self.logger.error("Strategy class %s not found", strategy_name)
                return None

            strategy_instance = strategy_class()
            signals = strategy_instance.generate_signals(data_df)

            # Convert to pandas Series with same index as data
            if isinstance(signals, pd.Series):
                return signals
            if isinstance(signals, (list, tuple)):
                return pd.Series(signals, index=data_df.index)
            self.logger.error(
                "Invalid signals format from strategy %s", strategy_name
            )
            return None

        except Exception as e:
            self.logger.error(
                "Error generating signals for %s/%s: %s", symbol, strategy_name, e
            )
            return None

    def _run_backtesting_lib(
        self, data_df: pd.DataFrame, signals: pd.Series
    ) -> dict[str, float]:
        """Run backtesting using the backtesting library."""
        try:
            # Prepare data for backtesting library (requires OHLCV)
            bt_data = data_df[["Open", "High", "Low", "Close", "Volume"]].copy()

            # Create strategy instance with our signals
            bt = Backtest(
                bt_data,
                BacktestingLibStrategy,
                cash=10000,
                commission=0.002,  # 0.2% commission
                exclusive_orders=True,
            )

            # Pass signals to strategy
            results = bt.run(signals_df=signals)

            # Extract comparable metrics
            metrics = {
                "total_return": (results["Return [%]"] / 100.0)
                if "Return [%]" in results
                else 0.0,
                "sharpe_ratio": results.get("Sharpe Ratio", 0.0),
                "sortino_ratio": results.get("Sortino Ratio", 0.0),
                "max_drawdown": abs(results.get("Max. Drawdown [%]", 0.0) / 100.0),
                "win_rate": results.get("Win Rate [%]", 0.0) / 100.0,
                "num_trades": int(results.get("# Trades", 0)),
            }

            return metrics

        except Exception as e:
            self.logger.error("Error running backtesting library: %s", e)
            return {}

    def _compare_metrics(
        self, our_metrics: dict, lib_metrics: dict, tolerance: float
    ) -> dict[str, float]:
        """Compare metrics and return relative differences."""
        differences = {}

        for metric_name in our_metrics.keys():
            if metric_name in lib_metrics:
                our_value = our_metrics[metric_name]
                lib_value = lib_metrics[metric_name]

                if lib_value != 0:
                    # Relative difference
                    diff = (our_value - lib_value) / abs(lib_value)
                else:
                    # Absolute difference when baseline is 0
                    diff = our_value

                differences[metric_name] = diff

        return differences

    def validate_multiple_strategies(
        self, symbols: Optional[list[str]] = None, limit: int = 10
    ) -> dict[str, Any]:
        """Validate metrics for multiple best strategies."""
        self.logger.info("Starting batch validation of multiple strategies")

        # Get best strategies from database
        db_session = get_db_session()

        query = db_session.query(BestStrategy).filter(BestStrategy.sortino_ratio > 1.0)

        if symbols:
            query = query.filter(BestStrategy.symbol.in_(symbols))

        strategies = (
            query.order_by(BestStrategy.sortino_ratio.desc()).limit(limit).all()
        )

        results = {
            "total_validated": 0,
            "passed": 0,
            "failed": 0,
            "error_count": 0,
            "strategy_results": [],
            "summary": {
                "avg_total_return_diff": 0.0,
                "avg_sharpe_diff": 0.0,
                "avg_sortino_diff": 0.0,
            },
        }

        total_return_diffs = []
        sharpe_diffs = []
        sortino_diffs = []

        for strategy in strategies:
            validation_result = self.validate_best_strategy_metrics(
                strategy.symbol, strategy.best_strategy, strategy.timeframe
            )

            results["strategy_results"].append(validation_result)
            results["total_validated"] += 1

            if validation_result["errors"]:
                results["error_count"] += 1
            elif validation_result["validation_passed"]:
                results["passed"] += 1

                # Collect differences for summary
                diffs = validation_result["differences"]
                if "total_return" in diffs:
                    total_return_diffs.append(abs(diffs["total_return"]))
                if "sharpe_ratio" in diffs:
                    sharpe_diffs.append(abs(diffs["sharpe_ratio"]))
                if "sortino_ratio" in diffs:
                    sortino_diffs.append(abs(diffs["sortino_ratio"]))
            else:
                results["failed"] += 1

        # Calculate summary statistics
        if total_return_diffs:
            results["summary"]["avg_total_return_diff"] = sum(total_return_diffs) / len(
                total_return_diffs
            )
        if sharpe_diffs:
            results["summary"]["avg_sharpe_diff"] = sum(sharpe_diffs) / len(
                sharpe_diffs
            )
        if sortino_diffs:
            results["summary"]["avg_sortino_diff"] = sum(sortino_diffs) / len(
                sortino_diffs
            )

        db_session.close()

        self.logger.info(
            "Batch validation complete: %d total, %d passed, %d failed, %d errors",
            results["total_validated"],
            results["passed"],
            results["failed"],
            results["error_count"],
        )

        return results

    def generate_validation_report(self, results: dict) -> str:
        """Generate a human-readable validation report."""
        if "strategy_results" in results:
            # Batch validation report
            return self._generate_batch_report(results)
        # Single strategy report
        return self._generate_single_report(results)

    def _generate_single_report(self, result: dict) -> str:
        """Generate report for single strategy validation."""
        report = []
        report.append("=== Metrics Validation Report ===")
        report.append(f"Symbol: {result['symbol']}")
        report.append(f"Strategy: {result['strategy']}")
        report.append(f"Timeframe: {result['timeframe']}")
        report.append(
            f"Status: {'✅ PASSED' if result['validation_passed'] else '❌ FAILED'}"
        )
        report.append("")

        if result["errors"]:
            report.append("Errors:")
            for error in result["errors"]:
                report.append(f"  - {error}")
            report.append("")

        if result["our_metrics"] and result["backtesting_lib_metrics"]:
            report.append("Metric Comparison:")
            report.append(
                f"{'Metric':<15} {'Our Value':<12} {'Lib Value':<12} {'Difference':<12}"
            )
            report.append("-" * 55)

            for metric in result["our_metrics"]:
                our_val = result["our_metrics"][metric]
                lib_val = result["backtesting_lib_metrics"].get(metric, 0)
                diff = result["differences"].get(metric, 0)

                report.append(
                    f"{metric:<15} {our_val:<12.4f} {lib_val:<12.4f} {diff * 100:<11.2f}%"
                )

        return "\n".join(report)

    def _generate_batch_report(self, results: dict) -> str:
        """Generate report for batch validation."""
        report = []
        report.append("=== Batch Metrics Validation Report ===")
        report.append(f"Total Strategies Validated: {results['total_validated']}")
        report.append(f"Passed: {results['passed']}")
        report.append(f"Failed: {results['failed']}")
        report.append(f"Errors: {results['error_count']}")
        report.append(
            f"Success Rate: {results['passed'] / max(1, results['total_validated']) * 100:.1f}%"
        )
        report.append("")

        summary = results["summary"]
        report.append("Average Differences:")
        report.append(f"  Total Return: {summary['avg_total_return_diff'] * 100:.2f}%")
        report.append(f"  Sharpe Ratio: {summary['avg_sharpe_diff'] * 100:.2f}%")
        report.append(f"  Sortino Ratio: {summary['avg_sortino_diff'] * 100:.2f}%")
        report.append("")

        # List failed validations
        failed_strategies = [
            r for r in results["strategy_results"] if not r["validation_passed"]
        ]
        if failed_strategies:
            report.append("Failed Validations:")
            for result in failed_strategies[:10]:  # Show first 10
                report.append(f"  - {result['symbol']}/{result['strategy']}")

        return "\n".join(report)
