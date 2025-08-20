"""
AI Investment Recommendations - AI-powered analysis of backtest results
to recommend optimal asset allocation and investment decisions.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.ai.ai_report_generator import AIReportGenerator
from src.ai.llm_client import LLMClient
from src.ai.models import AssetRecommendation, PortfolioRecommendation
from src.database.models import AIRecommendation, BacktestResult, BestStrategy
from src.database.models import AssetRecommendation as DbAssetRecommendation


class AIInvestmentRecommendations:
    """
    AI-powered investment recommendation system that analyzes backtest results
    to provide optimal asset allocation and investment decisions.
    """

    def __init__(self, db_session: Session = None):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        self.llm_client = LLMClient()

        # Risk tolerance levels
        self.risk_levels = {
            "conservative": {"max_drawdown": 0.10, "min_sortino": 1.0},
            "moderate": {"max_drawdown": 0.20, "min_sortino": 0.75},
            "aggressive": {"max_drawdown": 0.35, "min_sortino": 0.5},
        }

        # Scoring weights
        self.scoring_weights = {
            "sortino_ratio": 0.35,
            "calmar_ratio": 0.25,
            "profit_factor": 0.20,
            "max_drawdown": 0.10,
            "win_rate": 0.10,
        }

    @staticmethod
    def _ensure_python_type(val):
        """Convert any numpy type to Python native type."""
        if val is None:
            return None

        if isinstance(val, (np.floating, np.integer, np.bool_)):
            return val.item()  # Convert to Python native type
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return val.item()
            return val.tolist()
        if hasattr(val, "item"):  # Other numpy scalars
            return val.item()
        if isinstance(val, (list, tuple)):
            return [
                AIInvestmentRecommendations._ensure_python_type(item) for item in val
            ]
        if isinstance(val, dict):
            return {
                k: AIInvestmentRecommendations._ensure_python_type(v)
                for k, v in val.items()
            }
        return val

    def generate_recommendations(
        self,
        risk_tolerance: str = "moderate",
        min_confidence: float = 0.7,
        max_assets: int = 10,
        quarter: Optional[str] = None,
        timeframe: str = "1h",
        portfolio_name: Optional[str] = None,
    ) -> PortfolioRecommendation:
        """
        Generate AI-powered investment recommendations based on backtest results.

        Args:
            risk_tolerance: Risk level (conservative, moderate, aggressive)
            min_confidence: Minimum confidence score for recommendations
            max_assets: Maximum number of assets to recommend
            quarter: Specific quarter to analyze (e.g., "Q3_2025")

        Returns:
            PortfolioRecommendation with AI analysis
        """
        self.logger.info(
            "Generating AI recommendations for %s risk profile", risk_tolerance
        )

        # Load backtest results
        backtest_data = self._load_backtest_results(quarter)
        if not backtest_data:
            raise ValueError("No backtest results found")

        # Performance-based scoring
        scored_assets = self._calculate_performance_scores(backtest_data)

        # Risk-adjusted filtering
        filtered_assets = self._apply_risk_filters(scored_assets, risk_tolerance)

        # Portfolio correlation analysis
        correlation_data = self._analyze_correlations(filtered_assets)

        # Strategy-asset matching
        optimized_assets = self._optimize_strategy_asset_matching(filtered_assets)

        # Generate allocation suggestions
        allocations = self._suggest_allocations(
            optimized_assets, risk_tolerance, max_assets
        )

        # Red flag detection
        flagged_assets = self._detect_red_flags(allocations)

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            flagged_assets, backtest_data
        )

        # Create asset recommendations for ALL assets (no filtering by confidence)
        recommendations = []
        for asset_data in confidence_scores:
            # Determine investment recommendation
            base_reasoning = asset_data.get(
                "reasoning",
                f"Sortino: {asset_data['sortino_ratio']:.2f}, Max DD: {asset_data['max_drawdown']:.1%}",
            )

            if asset_data["confidence"] >= 0.6 and asset_data["score"] >= 0.3:
                invest_decision = "INVEST_WITH_RISK_MANAGEMENT"
                reasoning = f"Moderate performance metrics suggest cautious investment. {base_reasoning}"
            elif asset_data["confidence"] >= 0.4 and asset_data["score"] >= 0.2:
                invest_decision = "CONSIDER_WITH_HIGH_CAUTION"
                reasoning = f"Below-average performance requires extreme caution. {base_reasoning}"
            else:
                invest_decision = "DO_NOT_INVEST"
                reasoning = (
                    f"Poor performance metrics indicate high risk. {base_reasoning}"
                )

            # Calculate trading parameters
            trading_params = self._calculate_trading_parameters(asset_data, timeframe)

            recommendation = AssetRecommendation(
                symbol=asset_data["symbol"],
                strategy=asset_data["strategy"],
                score=self._ensure_python_type(asset_data["score"]),
                confidence=self._ensure_python_type(asset_data["confidence"]),
                allocation_percentage=self._ensure_python_type(
                    asset_data["allocation"]
                ),  # Always show suggested allocation
                risk_level=self._classify_risk_level(asset_data),
                reasoning=reasoning,
                red_flags=[*asset_data.get("red_flags", []), invest_decision],
                sortino_ratio=asset_data["sortino_ratio"],
                calmar_ratio=asset_data["calmar_ratio"],
                max_drawdown=asset_data["max_drawdown"],
                win_rate=asset_data["win_rate"],
                profit_factor=asset_data["profit_factor"],
                total_return=asset_data["total_return"],
                # Trading parameters
                trading_style=trading_params["trading_style"],
                timeframe=trading_params["timeframe"],
                risk_per_trade=trading_params["risk_per_trade"],
                stop_loss_points=trading_params["stop_loss_points"],
                take_profit_points=trading_params["take_profit_points"],
                position_size_percent=trading_params["position_size_percent"],
            )
            recommendations.append(recommendation)

        # Generate AI analysis
        ai_analysis = self._generate_ai_analysis(
            recommendations, correlation_data, risk_tolerance
        )

        portfolio_rec = PortfolioRecommendation(
            recommendations=recommendations,
            total_score=self._ensure_python_type(
                np.mean([r.score for r in recommendations])
            ),
            risk_profile=risk_tolerance,
            diversification_score=self._ensure_python_type(
                correlation_data["diversification_score"]
            ),
            correlation_analysis=correlation_data["correlations"],
            overall_reasoning=ai_analysis["reasoning"],
            warnings=ai_analysis["warnings"],
            confidence=self._ensure_python_type(
                np.mean([r.confidence for r in recommendations])
            ),
        )

        # Save to database and exports
        self._save_to_database(portfolio_rec, quarter, portfolio_name)
        self._save_to_exports(recommendations, risk_tolerance, quarter)

        return portfolio_rec

    def _generate_portfolio_filtered_recommendations(
        self,
        symbols: list[str],
        risk_tolerance: str = "moderate",
        min_confidence: float = 0.6,
        max_assets: int = 10,
        quarter: Optional[str] = None,
        timeframe: str = "1h",
        portfolio_name: Optional[str] = None,
    ) -> PortfolioRecommendation:
        """Generate recommendations specifically for portfolio symbols only."""
        self.logger.info(
            "Generating AI recommendations for %s risk profile", risk_tolerance
        )

        # Load backtest results and filter by portfolio symbols immediately
        backtest_data = self._load_backtest_results(quarter)
        if not backtest_data:
            raise ValueError("No backtest results found")

        # Filter to only include portfolio symbols and get best strategy per asset
        portfolio_backtest_data = []
        symbol_best_strategies = {}

        # Group by symbol and find best strategy for each
        for asset in backtest_data:
            if asset["symbol"] in symbols:
                symbol = asset["symbol"]

                # Keep track of best strategy per symbol (highest sortino ratio)
                if symbol not in symbol_best_strategies or (
                    asset["sortino_ratio"]
                    > symbol_best_strategies[symbol]["sortino_ratio"]
                ):
                    symbol_best_strategies[symbol] = asset

        # Convert to list format (only best strategy per asset)
        portfolio_backtest_data = list(symbol_best_strategies.values())

        if not portfolio_backtest_data:
            # Return empty portfolio recommendation
            return PortfolioRecommendation(
                recommendations=[],
                total_score=0.0,
                risk_profile=risk_tolerance,
                diversification_score=0.0,
                correlation_analysis={},
                overall_reasoning="No backtested assets found in portfolio. Only assets with backtest or optimization data are analyzed.",
                warnings=["Portfolio contains no backtested assets"],
                confidence=0.0,
            )

        self.logger.info(
            "Found %d backtested assets from portfolio symbols",
            len(portfolio_backtest_data),
        )

        # Performance-based scoring (no filtering, just scoring)
        scored_assets = self._calculate_performance_scores(portfolio_backtest_data)

        # Skip most filtering - just use scored assets directly
        # Portfolio correlation analysis
        correlation_data = self._analyze_correlations(scored_assets)

        # Calculate equal allocation for portfolio assets (like typical bond portfolios)
        num_assets = len(scored_assets)
        if num_assets > 0:
            base_allocation = 100.0 / num_assets  # Equal weight allocation
            for asset in scored_assets:
                # Adjust allocation slightly based on performance score
                score_multiplier = 0.8 + (
                    asset["score"] * 0.4
                )  # 0.8x to 1.2x based on score
                asset["allocation"] = self._ensure_python_type(
                    min(20.0, max(2.0, base_allocation * score_multiplier))
                )

        # Use scored assets directly (no filtering)
        confidence_scores = self._calculate_confidence_scores(
            scored_assets, portfolio_backtest_data
        )

        # Create asset recommendations for ALL portfolio assets (no filtering by confidence)
        recommendations = []
        for asset_data in confidence_scores:
            # Determine investment recommendation
            base_reasoning = asset_data.get(
                "reasoning",
                f"Sortino: {asset_data['sortino_ratio']:.2f}, Max DD: {asset_data['max_drawdown']:.1%}",
            )

            if asset_data["confidence"] >= 0.6 and asset_data["score"] >= 0.3:
                invest_decision = "INVEST_WITH_RISK_MANAGEMENT"
                reasoning = f"Moderate performance metrics suggest cautious investment. {base_reasoning}"
            elif asset_data["confidence"] >= 0.4 and asset_data["score"] >= 0.2:
                invest_decision = "CONSIDER_WITH_HIGH_CAUTION"
                reasoning = f"Below-average performance requires extreme caution. {base_reasoning}"
            else:
                invest_decision = "DO_NOT_INVEST"
                reasoning = (
                    f"Poor performance metrics indicate high risk. {base_reasoning}"
                )

            # Calculate trading parameters
            trading_params = self._calculate_trading_parameters(asset_data, timeframe)

            recommendation = AssetRecommendation(
                symbol=asset_data["symbol"],
                strategy=asset_data["strategy"],
                score=self._ensure_python_type(asset_data["score"]),
                confidence=self._ensure_python_type(asset_data["confidence"]),
                allocation_percentage=self._ensure_python_type(
                    asset_data["allocation"]
                ),  # Always show suggested allocation
                risk_level=self._classify_risk_level(asset_data),
                reasoning=reasoning,
                red_flags=[*asset_data.get("red_flags", []), invest_decision],
                sortino_ratio=asset_data["sortino_ratio"],
                calmar_ratio=asset_data["calmar_ratio"],
                max_drawdown=asset_data["max_drawdown"],
                win_rate=asset_data["win_rate"],
                profit_factor=asset_data["profit_factor"],
                total_return=asset_data["total_return"],
                # Trading parameters
                trading_style=trading_params["trading_style"],
                timeframe=trading_params["timeframe"],
                risk_per_trade=trading_params["risk_per_trade"],
                stop_loss_points=trading_params["stop_loss_points"],
                take_profit_points=trading_params["take_profit_points"],
                position_size_percent=trading_params["position_size_percent"],
            )
            recommendations.append(recommendation)

        # Generate AI analysis
        ai_analysis = self._generate_ai_analysis(
            recommendations, correlation_data, risk_tolerance
        )

        portfolio_rec = PortfolioRecommendation(
            recommendations=recommendations,
            total_score=self._ensure_python_type(
                np.mean([r.score for r in recommendations])
            ),
            risk_profile=risk_tolerance,
            diversification_score=self._ensure_python_type(
                correlation_data["diversification_score"]
            ),
            correlation_analysis=correlation_data["correlations"],
            overall_reasoning=ai_analysis["reasoning"],
            warnings=ai_analysis["warnings"],
            confidence=self._ensure_python_type(
                np.mean([r.confidence for r in recommendations])
            ),
        )

        return portfolio_rec

    def generate_portfolio_recommendations(
        self,
        portfolio_config_path: str,
        risk_tolerance: str = "moderate",
        min_confidence: float = 0.6,
        max_assets: int = 10,
        quarter: Optional[str] = None,
        timeframe: str = "1h",
        generate_html: bool = True,
    ) -> tuple[PortfolioRecommendation, str]:
        """Generate AI recommendations for a specific portfolio with HTML report."""
        import json
        from pathlib import Path

        # Load portfolio configuration
        portfolio_path = Path(portfolio_config_path)
        with portfolio_path.open() as f:
            portfolio_config = json.load(f)

        # Handle nested portfolio configuration
        if len(portfolio_config) == 1:
            # Single key, assume it's the portfolio config
            portfolio_key = list(portfolio_config.keys())[0]
            portfolio_data = portfolio_config[portfolio_key]
        else:
            # Direct configuration
            portfolio_data = portfolio_config

        portfolio_name = portfolio_data.get(
            "name", portfolio_path.stem.replace("_", " ").title()
        )
        symbols = portfolio_data.get("symbols", [])

        self.logger.info(
            "Generating AI recommendations for %s portfolio (%d symbols)",
            portfolio_name,
            len(symbols),
        )

        # Generate recommendations for only the portfolio symbols
        # (filter backtest data first before generating recommendations)
        portfolio_filtered_recommendations = (
            self._generate_portfolio_filtered_recommendations(
                symbols=symbols,
                risk_tolerance=risk_tolerance,
                min_confidence=min_confidence,
                max_assets=max_assets,
                quarter=quarter,
                timeframe=timeframe,
                portfolio_name=portfolio_name,
            )
        )

        portfolio_recommendations = portfolio_filtered_recommendations.recommendations

        self.logger.info(
            "Generated recommendations for %d backtested assets from %d portfolio symbols",
            len(portfolio_recommendations),
            len(symbols),
        )

        # Use the filtered portfolio recommendations
        filtered_portfolio = portfolio_filtered_recommendations

        # Save to database
        self._save_to_database(filtered_portfolio, quarter, portfolio_name)

        html_path = ""
        if generate_html:
            # Generate HTML report
            report_generator = AIReportGenerator()
            html_path = report_generator.generate_portfolio_html_report(
                portfolio_name=portfolio_name,
                recommendations=filtered_portfolio,
                quarter=quarter,
            )

        return filtered_portfolio, html_path

    def _load_backtest_results(self, quarter: Optional[str] = None) -> list[dict]:
        """Load backtest results from database (primary) or reports (fallback)."""
        if self.db_session:
            # Try database first, but check if metrics are properly calculated
            db_results = self._load_from_database(quarter)

            # Check if we have proper metrics (non-null Sortino ratios)
            valid_results = [r for r in db_results if r.get("sortino_ratio", 0) != 0]

            if (
                len(valid_results) < len(db_results) * 0.1
            ):  # Less than 10% have valid metrics
                self.logger.warning(
                    "Database metrics incomplete - falling back to HTML reports"
                )
                return self._load_from_reports(quarter)

            return db_results
        self.logger.warning("No database session - using reports as fallback")
        return self._load_from_reports(quarter)

    def _load_from_database(self, quarter: Optional[str] = None) -> list[dict]:
        """Load best strategies from database for faster and cleaner recommendations."""
        from datetime import datetime

        # Query best_strategies table directly - much more efficient
        query = self.db_session.query(BestStrategy)

        if quarter:
            # Filter by quarter if specified
            year, q = quarter.split("_")
            quarter_num = int(q[1])
            start_month = (quarter_num - 1) * 3 + 1
            end_month = quarter_num * 3

            start_date = datetime(int(year), start_month, 1)
            if quarter_num == 4:
                end_date = datetime(int(year) + 1, 1, 1)
            else:
                end_date = datetime(int(year), end_month + 1, 1)

            query = query.filter(
                BestStrategy.updated_at >= start_date,
                BestStrategy.updated_at < end_date,
            )

        # Order by primary metric (Sortino ratio) descending
        query = query.order_by(BestStrategy.sortino_ratio.desc())
        results = query.all()

        self.logger.info("Loaded %d best strategies from database", len(results))

        return [
            {
                "symbol": result.symbol,
                "strategy": result.strategy,
                "sortino_ratio": float(result.sortino_ratio or 0),
                "calmar_ratio": float(result.calmar_ratio or 0),
                "sharpe_ratio": float(result.sharpe_ratio or 0),
                "total_return": float(result.total_return or 0),
                "max_drawdown": float(result.max_drawdown or 0),
                "created_at": result.updated_at.isoformat()
                if result.updated_at
                else None,
                "initial_capital": 10000,  # Default value
                "final_value": 10000 * (1 + float(result.total_return or 0) / 100),
            }
            for result in results
        ]

    def _load_from_reports(self, quarter: Optional[str] = None) -> list[dict]:
        """Load backtest results from HTML reports."""
        reports_dir = Path("exports/reports")

        if quarter:
            year, q = quarter.split("_")
            reports_path = reports_dir / year / q
        else:
            # Get latest quarter
            reports_path = reports_dir / "2025" / "Q3"

        if not reports_path.exists():
            self.logger.warning("Reports directory %s not found", reports_path)
            return []

        # Parse HTML reports to extract metrics
        return self._parse_html_reports(reports_path)

    def _parse_html_reports(self, reports_path: Path) -> list[dict]:
        """Parse HTML reports to extract backtest metrics."""

        from bs4 import BeautifulSoup

        parsed_data = []

        # Find HTML reports in the directory
        html_files = list(reports_path.glob("*.html"))

        for html_file in html_files:
            try:
                with Path(html_file).open(encoding="utf-8") as f:
                    content = f.read()

                soup = BeautifulSoup(content, "html.parser")

                # Find asset sections
                asset_sections = soup.find_all("div", class_="asset-section")

                for section in asset_sections:
                    # Extract asset symbol from the title
                    asset_title = section.find("h2", class_="asset-title")
                    if not asset_title:
                        continue

                    symbol = asset_title.text.strip()

                    # Extract best strategy from the badge
                    strategy_badge = section.find("span", class_="strategy-badge")
                    if not strategy_badge:
                        continue

                    # Parse "Best: Strategy Name"
                    strategy_text = strategy_badge.text.strip()
                    if strategy_text.startswith("Best: "):
                        strategy = strategy_text[6:].strip()  # Remove "Best: "
                    else:
                        continue

                    # Extract metrics from metric cards
                    metrics_data = {
                        "symbol": symbol,
                        "strategy": strategy,
                        "sortino_ratio": 0.0,
                        "calmar_ratio": 0.0,
                        "sharpe_ratio": 0.0,
                        "profit_factor": 0.0,
                        "max_drawdown": 0.0,
                        "volatility": 0.0,
                        "win_rate": 0.0,
                        "total_return": 0.0,
                        "num_trades": 0,
                        "created_at": "2025-08-14",
                        "initial_capital": 10000,
                        "final_value": 10000,
                    }

                    # Find metric cards and extract values
                    metric_cards = section.find_all("div", class_="metric-card")
                    for card in metric_cards:
                        label_elem = card.find("div", class_="metric-label")
                        value_elem = card.find("div", class_="metric-value")

                        if not label_elem or not value_elem:
                            continue

                        label = label_elem.text.strip().lower()
                        value_text = value_elem.text.strip()

                        # Parse metric values
                        try:
                            # Remove % and convert to float
                            if "%" in value_text:
                                value = float(value_text.replace("%", "")) / 100
                            else:
                                value = float(value_text)

                            # Map labels to our metric keys
                            if "sortino" in label:
                                metrics_data["sortino_ratio"] = value
                            elif "calmar" in label:
                                metrics_data["calmar_ratio"] = value
                            elif "sharpe" in label:
                                metrics_data["sharpe_ratio"] = value
                            elif "profit factor" in label:
                                metrics_data["profit_factor"] = value
                            elif "max drawdown" in label or "maximum drawdown" in label:
                                metrics_data["max_drawdown"] = value
                            elif "volatility" in label:
                                metrics_data["volatility"] = value
                            elif "win rate" in label:
                                metrics_data["win_rate"] = value
                            elif "total return" in label:
                                metrics_data["total_return"] = value
                        except ValueError:
                            continue

                    parsed_data.append(metrics_data)

            except Exception as e:
                self.logger.warning("Error parsing HTML report %s: %s", html_file, e)
                continue

        self.logger.info("Parsed %d asset metrics from HTML reports", len(parsed_data))
        return parsed_data

    def _calculate_performance_scores(self, backtest_data: list[dict]) -> list[dict]:
        """Calculate performance scores for each asset based on metrics."""
        scored_assets = []

        for asset in backtest_data:
            # Normalize metrics for scoring
            sortino_score = min(max(asset["sortino_ratio"], 0), 5) / 5.0
            calmar_score = min(max(asset["calmar_ratio"], 0), 5) / 5.0
            drawdown_score = max(0, 1 - abs(asset["max_drawdown"]) / 100)
            return_score = min(max(asset["total_return"] / 100, 0), 1.0)

            # Calculate weighted score using available metrics
            score = (
                sortino_score * 0.4  # Primary metric
                + calmar_score * 0.3  # Secondary metric
                + return_score * 0.2  # Return component
                + drawdown_score * 0.1  # Risk component
            )

            asset["score"] = self._ensure_python_type(score)
            scored_assets.append(asset)

        return sorted(scored_assets, key=lambda x: x["score"], reverse=True)

    def _apply_risk_filters(
        self, assets: list[dict], risk_tolerance: str
    ) -> list[dict]:
        """Filter assets based on risk tolerance."""
        risk_criteria = self.risk_levels[risk_tolerance]

        filtered = []
        for asset in assets:
            max_dd = abs(asset["max_drawdown"])
            sortino = asset["sortino_ratio"]

            if (
                max_dd <= risk_criteria["max_drawdown"]
                and sortino >= risk_criteria["min_sortino"]
            ):
                filtered.append(asset)

        return filtered

    def _analyze_correlations(self, assets: list[dict]) -> dict:
        """Analyze portfolio correlations for diversification."""
        # This would calculate actual correlations using price data
        # TODO: Implement actual correlation calculation using price data
        symbols = [asset["symbol"] for asset in assets]

        # Placeholder correlation matrix - to be implemented
        correlations = {}
        # Calculate a basic diversification score based on number of assets
        # More assets generally means better diversification
        diversification_score = min(0.9, 0.3 + (len(symbols) * 0.1))
        _ = symbols  # Unused for now

        return {
            "correlations": correlations,
            "diversification_score": diversification_score,
        }

    def _optimize_strategy_asset_matching(self, assets: list[dict]) -> list[dict]:
        """Find optimal strategy-asset combinations."""
        # Group by symbol and find best strategy for each
        symbol_strategies = {}

        for asset in assets:
            symbol = asset["symbol"]
            if symbol not in symbol_strategies:
                symbol_strategies[symbol] = []
            symbol_strategies[symbol].append(asset)

        # Select best strategy per symbol
        optimized = []
        for symbol, strategies in symbol_strategies.items():
            best_strategy = max(strategies, key=lambda x: x["score"])
            optimized.append(best_strategy)

        return optimized

    def _suggest_allocations(
        self, assets: list[dict], _: str, max_assets: int
    ) -> list[dict]:
        """Suggest portfolio allocations based on scores and risk."""
        # Take top assets
        top_assets = assets[:max_assets]

        if not top_assets:
            return []

        # Calculate allocations based on scores
        total_score = sum(asset["score"] for asset in top_assets)

        for asset in top_assets:
            if total_score > 0:
                allocation = (asset["score"] / total_score) * 100
            else:
                allocation = 100 / len(top_assets)

            asset["allocation"] = self._ensure_python_type(allocation)

        return top_assets

    def _detect_red_flags(self, assets: list[dict]) -> list[dict]:
        """Detect potential issues with recommended assets."""
        for asset in assets:
            red_flags = []

            # High drawdown warning
            if abs(asset["max_drawdown"]) > 0.3:
                red_flags.append("High maximum drawdown risk")

            # Low Sortino ratio
            if asset["sortino_ratio"] < 0.5:
                red_flags.append("Low risk-adjusted returns")

            # High volatility
            if asset["volatility"] > 0.4:
                red_flags.append("High volatility")

            # Low win rate
            if asset["win_rate"] < 0.4:
                red_flags.append("Low win rate")

            # Insufficient trades
            if asset["num_trades"] < 10:
                red_flags.append("Insufficient trading history")

            asset["red_flags"] = red_flags

        return assets

    def _calculate_confidence_scores(
        self, assets: list[dict], _: list[dict]
    ) -> list[dict]:
        """Calculate confidence scores based on data quality and consistency."""
        for asset in assets:
            confidence_factors = []

            # Data quality (number of trades)
            trade_factor = float(min(asset.get("num_trades", 0) / 50, 1.0))
            confidence_factors.append(trade_factor)

            # Consistency (low volatility of returns)
            volatility_factor = float(max(0, 1 - asset["volatility"]))
            confidence_factors.append(volatility_factor)

            # Performance stability (Sortino ratio consistency)
            sortino_factor = float(min(asset["sortino_ratio"] / 2, 1.0))
            confidence_factors.append(sortino_factor)

            # Risk management (drawdown control)
            drawdown_factor = float(max(0, 1 - abs(asset["max_drawdown"]) * 2))
            confidence_factors.append(drawdown_factor)

            # Calculate weighted confidence and ensure it's a Python float
            asset["confidence"] = self._ensure_python_type(np.mean(confidence_factors))

        return assets

    def _classify_risk_level(self, asset_data: dict) -> str:
        """Classify asset risk level."""
        max_dd = abs(asset_data["max_drawdown"])
        volatility = asset_data["volatility"]

        if max_dd <= 0.1 and volatility <= 0.15:
            return "Low"
        if max_dd <= 0.25 and volatility <= 0.3:
            return "Medium"
        return "High"

    def _calculate_trading_parameters(
        self, asset_data: dict, timeframe: str = "1h"
    ) -> dict:
        """Calculate trading parameters based on timeframe and asset characteristics."""

        # Determine trading style based on timeframe
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        is_scalping = timeframe_minutes < 60  # Less than 1 hour = scalping

        trading_style = "scalp" if is_scalping else "swing"

        # Get asset volatility for parameter adjustment
        volatility = asset_data.get("volatility", 0.02)  # Default 2% volatility
        max_drawdown = abs(asset_data.get("max_drawdown", 0.05))

        if is_scalping:
            # Scalping parameters (tighter, more frequent trades)
            base_risk = 0.5  # 0.5% base risk per trade for scalping
            base_sl_points = max(
                5, volatility * 1000
            )  # Minimum 5 points, volatility-adjusted
            base_tp_points = base_sl_points * 2  # 1:2 risk-reward for scalping
            position_size = 5.0  # Smaller position sizes for scalping

            # Adjust based on volatility
            if volatility > 0.05:  # High volatility assets
                base_risk *= 0.7  # Reduce risk
                base_sl_points *= 1.5
                base_tp_points *= 1.5
        else:
            # Swing trading parameters (wider, longer-term trades)
            base_risk = 2.0  # 2% base risk per trade for swing
            base_sl_points = max(
                20, volatility * 3000
            )  # Minimum 20 points, volatility-adjusted
            base_tp_points = base_sl_points * 3  # 1:3 risk-reward for swing
            position_size = 10.0  # Larger position sizes for swing

            # Adjust based on volatility and drawdown
            if volatility > 0.03:  # High volatility assets
                base_risk *= 0.8
                base_sl_points *= 1.2
                base_tp_points *= 1.2

            if max_drawdown > 0.2:  # High drawdown history
                base_risk *= 0.6
                position_size *= 0.8

        # Risk level adjustments
        risk_level = self._classify_risk_level(asset_data)
        if risk_level == "High":
            base_risk *= 0.5
            position_size *= 0.7
        elif risk_level == "Low":
            base_risk *= 1.2
            position_size *= 1.1

        return {
            "trading_style": trading_style,
            "timeframe": timeframe,
            "risk_per_trade": round(base_risk, 1),
            "stop_loss_points": round(base_sl_points, 0),
            "take_profit_points": round(base_tp_points, 0),
            "position_size_percent": round(position_size, 1),
        }

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        timeframe = timeframe.lower()

        if "m" in timeframe:
            return int(timeframe.replace("m", ""))
        if "h" in timeframe:
            return int(timeframe.replace("h", "")) * 60
        if "d" in timeframe:
            return int(timeframe.replace("d", "")) * 24 * 60
        if "w" in timeframe:
            return int(timeframe.replace("w", "")) * 7 * 24 * 60
        return 60  # Default to 1 hour

    def _generate_ai_analysis(
        self,
        recommendations: list[AssetRecommendation],
        correlation_data: dict,
        risk_tolerance: str,
    ) -> dict[str, Any]:
        """Generate AI-powered analysis and reasoning."""
        if not recommendations:
            return {
                "reasoning": "No backtested assets found in portfolio. Only assets with backtest or optimization data are analyzed.",
                "warnings": ["Portfolio contains no backtested assets"],
            }

        # Prepare data for AI analysis
        analysis_data = {
            "risk_tolerance": risk_tolerance,
            "num_recommendations": len(recommendations),
            "avg_sortino": np.mean([r.sortino_ratio for r in recommendations]),
            "avg_calmar": np.mean([r.calmar_ratio for r in recommendations]),
            "max_drawdown_range": [r.max_drawdown for r in recommendations],
            "diversification_score": correlation_data["diversification_score"],
            "total_allocation": sum(r.allocation_percentage for r in recommendations),
            "red_flags_count": sum(len(r.red_flags) for r in recommendations),
        }

        # Generate AI reasoning
        try:
            ai_response = self.llm_client.analyze_portfolio(
                analysis_data, recommendations
            )
            return {
                "reasoning": ai_response.get(
                    "reasoning", "Analysis completed successfully"
                ),
                "warnings": ai_response.get("warnings", []),
            }
        except Exception as e:
            self.logger.error("AI analysis failed: %s", e)
            return {
                "reasoning": f"Quantitative analysis complete. {len(recommendations)} assets recommended with average Sortino ratio of {analysis_data['avg_sortino']:.2f}",
                "warnings": [
                    "AI analysis unavailable - using quantitative metrics only"
                ],
            }

    def get_asset_comparison(
        self, symbols: list[str], strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """Compare assets side by side with key metrics."""
        if self.db_session:
            from sqlalchemy import or_

            # Filter for results that contain any of the requested symbols
            symbol_filters = [BacktestResult.symbols.any(symbol) for symbol in symbols]
            query = self.db_session.query(BacktestResult).filter(or_(*symbol_filters))
            if strategy:
                query = query.filter(BacktestResult.strategy == strategy)

            results = query.all()

            comparison_data = []
            for result in results:
                comparison_data.append(
                    {
                        "Symbol": result.symbols[0] if result.symbols else "UNKNOWN",
                        "Strategy": result.strategy,
                        "Sortino Ratio": float(result.sortino_ratio or 0),
                        "Calmar Ratio": float(result.calmar_ratio or 0),
                        "Max Drawdown": float(result.max_drawdown or 0),
                        "Total Return": float(result.total_return or 0),
                        "Win Rate": float(result.win_rate or 0),
                        "Profit Factor": float(result.profit_factor or 0),
                    }
                )

            return pd.DataFrame(comparison_data)

        return pd.DataFrame()

    def explain_recommendation(self, symbol: str, strategy: str) -> dict[str, Any]:
        """Get detailed explanation for a specific recommendation."""
        # Load specific asset data
        asset_data = self._get_asset_data(symbol, strategy)

        if not asset_data:
            return {"error": "Asset data not found"}

        # Generate detailed AI explanation
        try:
            explanation = self.llm_client.explain_asset_recommendation(asset_data)
            return explanation
        except Exception as e:
            self.logger.error("Failed to generate explanation: %s", e)
            return {
                "summary": f"Asset {symbol} with {strategy} strategy shows Sortino ratio of {asset_data.get('sortino_ratio', 0):.2f}",
                "strengths": ["Quantitative metrics available"],
                "concerns": ["AI explanation unavailable"],
                "recommendation": "Review metrics manually",
            }

    def _get_asset_data(self, symbol: str, strategy: str) -> dict:
        """Get specific asset backtest data."""
        if self.db_session:
            result = (
                self.db_session.query(BacktestResult)
                .filter(
                    BacktestResult.symbols.contains([symbol]),
                    BacktestResult.strategy == strategy,
                )
                .first()
            )

            if result:
                return {
                    "symbol": symbol,
                    "strategy": strategy,
                    "sortino_ratio": float(result.sortino_ratio or 0),
                    "calmar_ratio": float(result.calmar_ratio or 0),
                    "max_drawdown": float(result.max_drawdown or 0),
                    "total_return": float(result.total_return or 0),
                    "win_rate": float(result.win_rate or 0),
                    "profit_factor": float(result.profit_factor or 0),
                    "volatility": float(result.volatility or 0),
                }

        return {}

    def _save_to_exports(
        self,
        recommendations: list[AssetRecommendation],
        risk_tolerance: str,
        quarter: str,
    ):
        """Save recommendations to exports/recommendations folder with year/quarter structure."""
        from datetime import datetime
        from pathlib import Path

        # Parse quarter and year or use current
        if quarter and "_" in quarter:
            # quarter might be like "Q3_2025"
            quarter_part, year_part = quarter.split("_")
        else:
            current_date = datetime.now()
            quarter_part = quarter or f"Q{(current_date.month - 1) // 3 + 1}"
            year_part = str(current_date.year)

        # Create organized exports directory
        exports_dir = Path("exports/recommendations") / year_part / quarter_part
        exports_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename matching other exporters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"AI_Recommendations_{risk_tolerance}_{quarter_part}_{year_part}_{timestamp}.json"

        # Prepare data
        data = {
            "generated_at": datetime.now().isoformat(),
            "quarter": f"{quarter_part}_{year_part}",
            "risk_tolerance": risk_tolerance,
            "total_recommendations": len(recommendations),
            "recommendations": [
                {
                    "symbol": rec.symbol,
                    "strategy": rec.strategy,
                    "score": rec.score,
                    "confidence": rec.confidence,
                    "allocation_percentage": rec.allocation_percentage,
                    "risk_level": rec.risk_level,
                    "metrics": {
                        "sortino_ratio": rec.sortino_ratio,
                        "calmar_ratio": rec.calmar_ratio,
                        "max_drawdown": rec.max_drawdown,
                        "win_rate": rec.win_rate,
                        "profit_factor": rec.profit_factor,
                    },
                    "reasoning": rec.reasoning,
                    "red_flags": rec.red_flags,
                }
                for rec in recommendations
            ],
        }

        # Save to file
        output_path = exports_dir / filename
        with output_path.open("w") as f:
            json.dump(data, f, indent=2)

        self.logger.info("AI recommendations saved to %s", output_path)

    def _save_to_database(
        self,
        portfolio_rec: PortfolioRecommendation,
        quarter: str,
        portfolio_name: Optional[str] = None,
    ):
        """Save AI recommendations to PostgreSQL database using normalized structure."""
        if not self.db_session:
            self.logger.warning("No database session - skipping database save")
            return

        from datetime import datetime

        # Determine which LLM model was used
        llm_model = "unknown"
        if os.getenv("OPENAI_API_KEY"):
            llm_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        elif os.getenv("ANTHROPIC_API_KEY"):
            llm_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

        # Parse quarter and year
        quarter_str = (
            quarter or f"Q{(datetime.now().month - 1) // 3 + 1}_{datetime.now().year}"
        )
        if "_" in quarter_str:
            q_part, year_part = quarter_str.split("_")
            year = int(year_part)
            quarter_only = q_part
        else:
            year = datetime.now().year
            quarter_only = quarter_str

        # Calculate portfolio-level metrics with type conversion
        total_return = self._ensure_python_type(
            sum(
                rec.allocation_percentage * rec.total_return
                for rec in portfolio_rec.recommendations
            )
            / 100
        )
        portfolio_risk = self._ensure_python_type(
            sum(
                rec.allocation_percentage * rec.max_drawdown
                for rec in portfolio_rec.recommendations
            )
            / 100
        )

        try:
            # Create main AI recommendation record
            ai_rec = AIRecommendation(
                portfolio_name=portfolio_name or "default",
                risk_profile=portfolio_rec.risk_profile,
                confidence_score=self._ensure_python_type(portfolio_rec.confidence),
                recommendation_data={
                    "total_score": self._ensure_python_type(portfolio_rec.total_score),
                    "diversification_score": self._ensure_python_type(
                        portfolio_rec.diversification_score
                    ),
                    "quarter": quarter_only,
                    "year": year,
                    "total_assets": len(portfolio_rec.recommendations),
                    "expected_return": total_return,
                    "portfolio_risk": portfolio_risk,
                    "overall_reasoning": portfolio_rec.overall_reasoning,
                    "warnings": self._ensure_python_type(portfolio_rec.warnings),
                    "correlation_analysis": self._ensure_python_type(
                        portfolio_rec.correlation_analysis
                    ),
                    "llm_model": llm_model,
                },
            )

            self.db_session.add(ai_rec)
            self.db_session.flush()  # Get the ID

            # Create individual asset recommendation records using manual conversion
            for rec in portfolio_rec.recommendations:
                # Find corresponding best strategy
                best_strategy = (
                    self.db_session.query(BestStrategy)
                    .filter_by(symbol=rec.symbol)
                    .first()
                )

                # Convert to plain dict to avoid dataclass numpy issues
                # Ultimate safety conversion - manually check each field
                def force_native_type(val):
                    """Forcefully convert to native Python type."""
                    if val is None:
                        return None
                    val_str = str(type(val))
                    if "numpy" in val_str:
                        return float(val)
                    return val

                asset_rec = DbAssetRecommendation(
                    ai_recommendation_id=ai_rec.id,
                    symbol=rec.symbol,
                    allocation_percentage=force_native_type(rec.allocation_percentage),
                    confidence_score=force_native_type(rec.confidence),
                    performance_score=force_native_type(rec.score),
                    risk_score=force_native_type(abs(rec.max_drawdown)),
                    reasoning=rec.reasoning,
                    red_flags=list(rec.red_flags)
                    if isinstance(rec.red_flags, list)
                    else [str(rec.red_flags)],
                    risk_per_trade=force_native_type(rec.risk_per_trade),
                    stop_loss_pct=force_native_type(rec.stop_loss_points),
                    take_profit_pct=force_native_type(rec.take_profit_points),
                    position_size_usd=force_native_type(rec.position_size_percent),
                    best_strategy_id=best_strategy.id if best_strategy else None,
                )
                self.db_session.add(asset_rec)

            self.db_session.commit()
            self.logger.info(
                "AI recommendations saved to database: %s_%s, %s",
                quarter_only,
                year,
                portfolio_rec.risk_profile,
            )

        except Exception as e:
            self.db_session.rollback()
            self.logger.error("Failed to save AI recommendations to database: %s", e)
            raise
