"""
Advanced reporting system with caching, persistence, and interactive visualizations.
Supports comprehensive portfolio analysis, strategy comparison, and optimization reporting.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader
from plotly.subplots import make_subplots

from src.core.backtest_engine import BacktestResult
from src.portfolio.advanced_optimizer import OptimizationResult

warnings.filterwarnings("ignore")


class AdvancedReportGenerator:
    """
    Advanced report generator with interactive visualizations and caching.
    Supports multiple output formats and comprehensive analysis.
    """

    def __init__(self, output_dir: str = "exports/reports", cache_reports: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cache_reports = cache_reports
        self.logger = logging.getLogger(__name__)

        # Setup template environment
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        self.template_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True,  # Enable XSS protection
        )

        # Ensure template files exist
        self._ensure_templates()

        # Configure Plotly
        pio.templates.default = "plotly_white"

    def generate_portfolio_report(
        self,
        results: list[BacktestResult],
        title: str = "Portfolio Analysis Report",
        include_charts: bool = True,
        format: str = "html",
    ) -> str:
        """
        Generate comprehensive portfolio analysis report.

        Args:
            results: List of backtest results
            title: Report title
            include_charts: Whether to include interactive charts
            format: Output format ('html', 'pdf', 'json')

        Returns:
            Path to generated report
        """
        start_time = time.time()

        # Check cache
        cache_key = self._get_report_cache_key(
            "portfolio", results, title, include_charts, format
        )
        if self.cache_reports:
            cached_report = self._get_cached_report(cache_key)
            if cached_report:
                self.logger.info("Using cached portfolio report")
                return cached_report

        self.logger.info("Generating portfolio report for %s results", len(results))

        # Prepare data
        report_data = self._prepare_portfolio_data(results)

        # Generate charts
        charts = {}
        if include_charts:
            charts = self._generate_portfolio_charts(report_data)

        # Generate report based on format
        if format == "html":
            report_path = self._generate_html_portfolio_report(
                report_data, charts, title
            )
        elif format == "json":
            report_path = self._generate_json_portfolio_report(report_data, title)
        else:
            msg = f"Unsupported format: {format}"
            raise ValueError(msg)

        # Cache report
        if self.cache_reports:
            self._cache_report(cache_key, report_path)

        generation_time = time.time() - start_time
        self.logger.info(
            "Portfolio report generated in %ss: %s", generation_time, report_path
        )

        return str(report_path)

    def generate_strategy_comparison_report(
        self,
        results: dict[str, list[BacktestResult]],
        title: str = "Strategy Comparison Report",
        include_charts: bool = True,
        format: str = "html",
    ) -> str:
        """
        Generate strategy comparison report.

        Args:
            results: Dictionary mapping strategy names to results
            title: Report title
            include_charts: Whether to include interactive charts
            format: Output format

        Returns:
            Path to generated report
        """
        start_time = time.time()

        # Check cache
        cache_key = self._get_report_cache_key(
            "strategy_comparison", results, title, include_charts, format
        )
        if self.cache_reports:
            cached_report = self._get_cached_report(cache_key)
            if cached_report:
                self.logger.info("Using cached strategy comparison report")
                return cached_report

        self.logger.info(
            "Generating strategy comparison report for %s strategies", len(results)
        )

        # Prepare data
        comparison_data = self._prepare_strategy_comparison_data(results)

        # Generate charts
        charts = {}
        if include_charts:
            charts = self._generate_strategy_comparison_charts(comparison_data)

        # Generate report
        if format == "html":
            report_path = self._generate_html_strategy_comparison_report(
                comparison_data, charts, title
            )
        elif format == "json":
            report_path = self._generate_json_strategy_comparison_report(
                comparison_data, title
            )
        else:
            msg = f"Unsupported format: {format}"
            raise ValueError(msg)

        # Cache report
        if self.cache_reports:
            self._cache_report(cache_key, report_path)

        generation_time = time.time() - start_time
        self.logger.info(
            "Strategy comparison report generated in %ss: %s",
            generation_time,
            report_path,
        )

        return str(report_path)

    def generate_optimization_report(
        self,
        optimization_results: dict[str, dict[str, OptimizationResult]],
        title: str = "Optimization Analysis Report",
        include_charts: bool = True,
        format: str = "html",
    ) -> str:
        """
        Generate optimization analysis report.

        Args:
            optimization_results: Nested dict of optimization results
            title: Report title
            include_charts: Whether to include interactive charts
            format: Output format

        Returns:
            Path to generated report
        """
        start_time = time.time()

        # Check cache
        cache_key = self._get_report_cache_key(
            "optimization", optimization_results, title, include_charts, format
        )
        if self.cache_reports:
            cached_report = self._get_cached_report(cache_key)
            if cached_report:
                self.logger.info("Using cached optimization report")
                return cached_report

        self.logger.info("Generating optimization analysis report")

        # Prepare data
        optimization_data = self._prepare_optimization_data(optimization_results)

        # Generate charts
        charts = {}
        if include_charts:
            charts = self._generate_optimization_charts(optimization_data)

        # Generate report
        if format == "html":
            report_path = self._generate_html_optimization_report(
                optimization_data, charts, title
            )
        elif format == "json":
            report_path = self._generate_json_optimization_report(
                optimization_data, title
            )
        else:
            msg = f"Unsupported format: {format}"
            raise ValueError(msg)

        # Cache report
        if self.cache_reports:
            self._cache_report(cache_key, report_path)

        generation_time = time.time() - start_time
        self.logger.info(
            "Optimization report generated in %ss: %s", generation_time, report_path
        )

        return str(report_path)

    def _prepare_portfolio_data(self, results: list[BacktestResult]) -> dict[str, Any]:
        """Prepare data for portfolio analysis."""
        # Create summary DataFrame
        rows = []
        for result in results:
            if result.error:
                continue

            row = {
                "symbol": result.symbol,
                "strategy": result.strategy,
                "total_return": result.metrics.get("total_return", 0),
                "sharpe_ratio": result.metrics.get("sharpe_ratio", 0),
                "max_drawdown": result.metrics.get("max_drawdown", 0),
                "win_rate": result.metrics.get("win_rate", 0),
                "profit_factor": result.metrics.get("profit_factor", 0),
                "num_trades": result.metrics.get("num_trades", 0),
                "data_points": result.data_points,
                "duration_seconds": result.duration_seconds,
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Calculate portfolio statistics
        portfolio_stats = {
            "total_strategies": len(df["strategy"].unique()) if not df.empty else 0,
            "total_symbols": len(df["symbol"].unique()) if not df.empty else 0,
            "total_backtests": len(df),
            "successful_backtests": (
                len(df[df["total_return"] > 0]) if not df.empty else 0
            ),
            "avg_return": df["total_return"].mean() if not df.empty else 0,
            "avg_sharpe": df["sharpe_ratio"].mean() if not df.empty else 0,
            "best_strategy": (
                df.loc[df["total_return"].idxmax(), "strategy"]
                if not df.empty
                else None
            ),
            "best_symbol": (
                df.loc[df["total_return"].idxmax(), "symbol"] if not df.empty else None
            ),
            "worst_drawdown": df["max_drawdown"].min() if not df.empty else 0,
        }

        # Strategy performance
        strategy_performance = {}
        if not df.empty:
            for strategy in df["strategy"].unique():
                strategy_df = df[df["strategy"] == strategy]
                strategy_performance[strategy] = {
                    "count": len(strategy_df),
                    "avg_return": strategy_df["total_return"].mean(),
                    "avg_sharpe": strategy_df["sharpe_ratio"].mean(),
                    "win_rate": len(strategy_df[strategy_df["total_return"] > 0])
                    / len(strategy_df)
                    * 100,
                    "best_symbol": strategy_df.loc[
                        strategy_df["total_return"].idxmax(), "symbol"
                    ],
                }

        # Symbol performance
        symbol_performance = {}
        if not df.empty:
            for symbol in df["symbol"].unique():
                symbol_df = df[df["symbol"] == symbol]
                symbol_performance[symbol] = {
                    "count": len(symbol_df),
                    "avg_return": symbol_df["total_return"].mean(),
                    "avg_sharpe": symbol_df["sharpe_ratio"].mean(),
                    "best_strategy": symbol_df.loc[
                        symbol_df["total_return"].idxmax(), "strategy"
                    ],
                }

        return {
            "summary_df": df,
            "portfolio_stats": portfolio_stats,
            "strategy_performance": strategy_performance,
            "symbol_performance": symbol_performance,
            "generation_time": datetime.now().isoformat(),
        }

    def _prepare_strategy_comparison_data(
        self, results: dict[str, list[BacktestResult]]
    ) -> dict[str, Any]:
        """Prepare data for strategy comparison."""
        comparison_stats = {}
        all_results = []

        for strategy, strategy_results in results.items():
            strategy_metrics = []
            for result in strategy_results:
                if not result.error:
                    strategy_metrics.append(result.metrics)
                    all_results.append(
                        {
                            "strategy": strategy,
                            "symbol": result.symbol,
                            **result.metrics,
                        }
                    )

            if strategy_metrics:
                comparison_stats[strategy] = {
                    "count": len(strategy_metrics),
                    "avg_return": np.mean(
                        [m.get("total_return", 0) for m in strategy_metrics]
                    ),
                    "std_return": np.std(
                        [m.get("total_return", 0) for m in strategy_metrics]
                    ),
                    "avg_sharpe": np.mean(
                        [m.get("sharpe_ratio", 0) for m in strategy_metrics]
                    ),
                    "avg_drawdown": np.mean(
                        [m.get("max_drawdown", 0) for m in strategy_metrics]
                    ),
                    "win_rate": np.mean(
                        [m.get("win_rate", 0) for m in strategy_metrics]
                    ),
                    "best_return": max(
                        [m.get("total_return", 0) for m in strategy_metrics]
                    ),
                    "worst_return": min(
                        [m.get("total_return", 0) for m in strategy_metrics]
                    ),
                }

        df = pd.DataFrame(all_results)

        return {
            "comparison_stats": comparison_stats,
            "results_df": df,
            "generation_time": datetime.now().isoformat(),
        }

    def _prepare_optimization_data(
        self, optimization_results: dict[str, dict[str, OptimizationResult]]
    ) -> dict[str, Any]:
        """Prepare data for optimization analysis."""
        optimization_summary = {}
        convergence_data = []
        parameter_analysis = {}

        for symbol, strategies in optimization_results.items():
            for strategy, result in strategies.items():
                key = f"{symbol}_{strategy}"
                optimization_summary[key] = {
                    "symbol": symbol,
                    "strategy": strategy,
                    "best_score": result.best_score,
                    "total_evaluations": result.total_evaluations,
                    "optimization_time": result.optimization_time,
                    "convergence_generation": result.convergence_generation,
                    "best_parameters": result.best_parameters,
                }

                # Convergence data
                if result.optimization_history:
                    for entry in result.optimization_history:
                        convergence_data.append(
                            {
                                "symbol": symbol,
                                "strategy": strategy,
                                "key": key,
                                **entry,
                            }
                        )

                # Parameter analysis
                if result.best_parameters:
                    for param, value in result.best_parameters.items():
                        if strategy not in parameter_analysis:
                            parameter_analysis[strategy] = {}
                        if param not in parameter_analysis[strategy]:
                            parameter_analysis[strategy][param] = []
                        parameter_analysis[strategy][param].append(value)

        return {
            "optimization_summary": optimization_summary,
            "convergence_data": convergence_data,
            "parameter_analysis": parameter_analysis,
            "generation_time": datetime.now().isoformat(),
        }

    def _generate_portfolio_charts(self, data: dict[str, Any]) -> dict[str, str]:
        """Generate interactive charts for portfolio analysis."""
        charts = {}
        df = data["summary_df"]

        if df.empty:
            return charts

        # Returns distribution
        fig_returns = px.histogram(
            df, x="total_return", nbins=30, title="Distribution of Returns"
        )
        fig_returns.update_layout(
            xaxis_title="Total Return (%)", yaxis_title="Frequency"
        )
        charts["returns_distribution"] = fig_returns.to_html(include_plotlyjs="cdn")

        # Strategy performance comparison
        strategy_stats = (
            df.groupby("strategy")
            .agg(
                {
                    "total_return": ["mean", "std"],
                    "sharpe_ratio": "mean",
                    "max_drawdown": "mean",
                }
            )
            .round(2)
        )

        fig_strategy = go.Figure()
        strategies = strategy_stats.index
        fig_strategy.add_trace(
            go.Bar(
                name="Average Return",
                x=strategies,
                y=strategy_stats[("total_return", "mean")],
                text=strategy_stats[("total_return", "mean")],
                textposition="auto",
            )
        )
        fig_strategy.update_layout(
            title="Strategy Performance Comparison",
            xaxis_title="Strategy",
            yaxis_title="Average Return (%)",
        )
        charts["strategy_performance"] = fig_strategy.to_html(include_plotlyjs="cdn")

        # Risk-Return scatter
        fig_scatter = px.scatter(
            df,
            x="max_drawdown",
            y="total_return",
            color="strategy",
            size="sharpe_ratio",
            hover_data=["symbol"],
            title="Risk-Return Analysis",
        )
        fig_scatter.update_layout(
            xaxis_title="Max Drawdown (%)", yaxis_title="Total Return (%)"
        )
        charts["risk_return"] = fig_scatter.to_html(include_plotlyjs="cdn")

        # Top performers table
        top_performers = df.nlargest(10, "total_return")[
            ["symbol", "strategy", "total_return", "sharpe_ratio"]
        ]
        fig_table = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["Symbol", "Strategy", "Return (%)", "Sharpe Ratio"],
                        fill_color="paleturquoise",
                        align="left",
                    ),
                    cells=dict(
                        values=[
                            top_performers.symbol,
                            top_performers.strategy,
                            top_performers.total_return.round(2),
                            top_performers.sharpe_ratio.round(2),
                        ],
                        fill_color="lavender",
                        align="left",
                    ),
                )
            ]
        )
        fig_table.update_layout(title="Top 10 Performers")
        charts["top_performers"] = fig_table.to_html(include_plotlyjs="cdn")

        return charts

    def _generate_strategy_comparison_charts(
        self, data: dict[str, Any]
    ) -> dict[str, str]:
        """Generate charts for strategy comparison."""
        charts = {}
        comparison_stats = data["comparison_stats"]

        if not comparison_stats:
            return charts

        # Strategy metrics comparison
        strategies = list(comparison_stats.keys())
        metrics = ["avg_return", "avg_sharpe", "avg_drawdown", "win_rate"]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Average Return",
                "Average Sharpe Ratio",
                "Average Drawdown",
                "Win Rate",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1

            values = [comparison_stats[s][metric] for s in strategies]

            fig.add_trace(
                go.Bar(x=strategies, y=values, name=metric, showlegend=False),
                row=row,
                col=col,
            )

        fig.update_layout(title_text="Strategy Metrics Comparison", height=600)
        charts["strategy_metrics"] = fig.to_html(include_plotlyjs="cdn")

        return charts

    def _generate_optimization_charts(self, data: dict[str, Any]) -> dict[str, str]:
        """Generate charts for optimization analysis."""
        charts = {}
        convergence_data = data["convergence_data"]

        if not convergence_data:
            return charts

        # Convergence plots
        convergence_df = pd.DataFrame(convergence_data)

        if not convergence_df.empty:
            fig_convergence = px.line(
                convergence_df,
                x="generation",
                y="best_score",
                color="key",
                title="Optimization Convergence",
            )
            fig_convergence.update_layout(
                xaxis_title="Generation", yaxis_title="Best Score"
            )
            charts["convergence"] = fig_convergence.to_html(include_plotlyjs="cdn")

        return charts

    def _generate_html_portfolio_report(
        self, data: dict[str, Any], charts: dict[str, str], title: str
    ) -> Path:
        """Generate HTML portfolio report."""
        template = self.template_env.get_template("portfolio_report.html")

        html_content = template.render(
            title=title,
            data=data,
            charts=charts,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        filename = f"portfolio_report_{int(time.time())}.html"
        report_path = self.output_dir / filename
        report_path.write_text(html_content, encoding="utf-8")

        return report_path

    def _generate_html_strategy_comparison_report(
        self, data: dict[str, Any], charts: dict[str, str], title: str
    ) -> Path:
        """Generate HTML strategy comparison report."""
        template = self.template_env.get_template("strategy_comparison_report.html")

        html_content = template.render(
            title=title,
            data=data,
            charts=charts,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        filename = f"strategy_comparison_{int(time.time())}.html"
        report_path = self.output_dir / filename
        report_path.write_text(html_content, encoding="utf-8")

        return report_path

    def _generate_html_optimization_report(
        self, data: dict[str, Any], charts: dict[str, str], title: str
    ) -> Path:
        """Generate HTML optimization report."""
        template = self.template_env.get_template("optimization_report.html")

        html_content = template.render(
            title=title,
            data=data,
            charts=charts,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        filename = f"optimization_report_{int(time.time())}.html"
        report_path = self.output_dir / filename
        report_path.write_text(html_content, encoding="utf-8")

        return report_path

    def _generate_json_portfolio_report(self, data: dict[str, Any], title: str) -> Path:
        """Generate JSON portfolio report."""
        report_data = {
            "title": title,
            "type": "portfolio_analysis",
            "data": data,
            "generation_time": datetime.now().isoformat(),
        }

        filename = f"portfolio_report_{int(time.time())}.json"
        report_path = self.output_dir / filename

        with report_path.open("w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return report_path

    def _generate_json_strategy_comparison_report(
        self, data: dict[str, Any], title: str
    ) -> Path:
        """Generate JSON strategy comparison report."""
        report_data = {
            "title": title,
            "type": "strategy_comparison",
            "data": data,
            "generation_time": datetime.now().isoformat(),
        }

        filename = f"strategy_comparison_{int(time.time())}.json"
        report_path = self.output_dir / filename

        with report_path.open("w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return report_path

    def _generate_json_optimization_report(
        self, data: dict[str, Any], title: str
    ) -> Path:
        """Generate JSON optimization report."""
        report_data = {
            "title": title,
            "type": "optimization_analysis",
            "data": data,
            "generation_time": datetime.now().isoformat(),
        }

        filename = f"optimization_report_{int(time.time())}.json"
        report_path = self.output_dir / filename

        with report_path.open("w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return report_path

    def _get_report_cache_key(self, report_type: str, data: Any, *args) -> str:
        """Generate cache key for report."""
        import hashlib

        # Create a hash of the input data and parameters
        data_str = str(data) + str(args)
        cache_key = hashlib.sha256(data_str.encode()).hexdigest()[:16]

        return f"{report_type}_{cache_key}"

    def _get_cached_report(self, cache_key: str) -> str | None:
        """Get cached report if available."""
        # Implementation would check advanced_cache for cached report
        # For now, return None to always generate fresh reports
        return None

    def _cache_report(self, cache_key: str, report_path: Path):
        """Cache generated report."""
        # Implementation would cache the report using advanced_cache
        # For now, just log that we would cache it
        self.logger.debug("Would cache report %s with key %s", report_path, cache_key)

    def _ensure_templates(self):
        """Ensure HTML templates exist."""
        template_dir = Path(__file__).parent / "templates"

        # Basic portfolio report template
        portfolio_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .stat-card { background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }
        .chart { margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated: {{ generation_time }}</p>
    </div>

    <div class="stats">
        {% for key, value in data.portfolio_stats.items() %}
        <div class="stat-card">
            <h3>{{ key.replace('_', ' ').title() }}</h3>
            <p>{{ value }}</p>
        </div>
        {% endfor %}
    </div>

    {% for chart_name, chart_html in charts.items() %}
    <div class="chart">
        <h2>{{ chart_name.replace('_', ' ').title() }}</h2>
        {{ chart_html|safe }}
    </div>
    {% endfor %}
</body>
</html>
        """

        # Save template
        portfolio_template_path = template_dir / "portfolio_report.html"
        if not portfolio_template_path.exists():
            portfolio_template_path.write_text(portfolio_template)

        # Create other templates similarly
        strategy_template = portfolio_template.replace(
            "{{ title }}", "Strategy Comparison Report"
        )
        strategy_template_path = template_dir / "strategy_comparison_report.html"
        if not strategy_template_path.exists():
            strategy_template_path.write_text(strategy_template)

        optimization_template = portfolio_template.replace(
            "{{ title }}", "Optimization Analysis Report"
        )
        optimization_template_path = template_dir / "optimization_report.html"
        if not optimization_template_path.exists():
            optimization_template_path.write_text(optimization_template)


class ReportScheduler:
    """Scheduler for automated report generation."""

    def __init__(self, report_generator: AdvancedReportGenerator):
        self.report_generator = report_generator
        self.scheduled_reports = []
        self.logger = logging.getLogger(__name__)

    def schedule_daily_portfolio_report(
        self, results_function: Callable, title: str = "Daily Portfolio Report"
    ):
        """Schedule daily portfolio report generation."""
        # Implementation for scheduling would go here

    def schedule_weekly_optimization_report(
        self, optimization_function: Callable, title: str = "Weekly Optimization Report"
    ):
        """Schedule weekly optimization report generation."""
        # Implementation for scheduling would go here

    def run_scheduled_reports(self):
        """Run all scheduled reports."""
        # Implementation for running scheduled reports would go here
