"""
Database query helpers for efficient data retrieval from normalized tables.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from .models import AIRecommendation
from .models import AssetRecommendation as DbAssetRecommendation
from .models import BestOptimizationResult, BestStrategy


class DatabaseQueryHelper:
    """Helper class for common database queries across exporters and reports."""

    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)

    def get_best_strategies(
        self,
        quarter: str = None,
        year: int = None,
        symbols: list[str] = None,
        min_sortino: float = None,
    ) -> list[dict]:
        """Get best strategies with optional filtering."""
        query = self.db_session.query(BestStrategy)

        # Apply filters
        if quarter and year:
            start_date, end_date = self._get_quarter_dates(quarter, year)
            query = query.filter(
                BestStrategy.last_updated >= start_date,
                BestStrategy.last_updated < end_date,
            )

        if symbols:
            query = query.filter(BestStrategy.symbol.in_(symbols))

        if min_sortino:
            query = query.filter(BestStrategy.sortino_ratio >= min_sortino)

        # Order by performance
        results = query.order_by(BestStrategy.sortino_ratio.desc()).all()

        return [
            {
                "symbol": result.symbol,
                "strategy": result.best_strategy,
                "timeframe": result.timeframe,
                "sortino_ratio": float(result.sortino_ratio or 0),
                "sharpe_ratio": float(result.sharpe_ratio or 0),
                "calmar_ratio": float(result.calmar_ratio or 0),
                "profit_factor": float(result.profit_factor or 0),
                "total_return": float(result.total_return or 0),
                "max_drawdown": float(result.max_drawdown or 0),
                "volatility": float(result.volatility or 0),
                "win_rate": float(result.win_rate or 0),
                "num_trades": int(result.num_trades or 0),
                "risk_score": float(result.risk_score or 0),
                "risk_per_trade": float(result.risk_per_trade or 0),
                "stop_loss_pct": float(result.stop_loss_pct or 0),
                "take_profit_pct": float(result.take_profit_pct or 0),
                "parameters": result.best_parameters or {},
                "last_updated": result.last_updated.isoformat()
                if result.last_updated
                else None,
            }
            for result in results
        ]

    def get_best_optimizations(
        self,
        quarter: str = None,
        year: int = None,
        symbols: list[str] = None,
        strategies: list[str] = None,
        min_sortino: float = None,
    ) -> list[dict]:
        """Get best optimization results with optional filtering."""
        query = self.db_session.query(BestOptimizationResult)

        # Apply filters
        if quarter and year:
            start_date, end_date = self._get_quarter_dates(quarter, year)
            query = query.filter(
                BestOptimizationResult.last_updated >= start_date,
                BestOptimizationResult.last_updated < end_date,
            )

        if symbols:
            query = query.filter(BestOptimizationResult.symbol.in_(symbols))

        if strategies:
            query = query.filter(BestOptimizationResult.strategy.in_(strategies))

        if min_sortino:
            query = query.filter(
                BestOptimizationResult.best_sortino_ratio >= min_sortino
            )

        # Order by performance
        results = query.order_by(BestOptimizationResult.best_sortino_ratio.desc()).all()

        return [
            {
                "symbol": result.symbol,
                "strategy": result.strategy,
                "timeframe": result.timeframe,
                "best_sortino_ratio": float(result.best_sortino_ratio or 0),
                "best_sharpe_ratio": float(result.best_sharpe_ratio or 0),
                "best_calmar_ratio": float(result.best_calmar_ratio or 0),
                "best_profit_factor": float(result.best_profit_factor or 0),
                "best_total_return": float(result.best_total_return or 0),
                "best_max_drawdown": float(result.best_max_drawdown or 0),
                "best_volatility": float(result.best_volatility or 0),
                "best_win_rate": float(result.best_win_rate or 0),
                "best_num_trades": int(result.best_num_trades or 0),
                "best_parameters": result.best_parameters or {},
                "total_iterations": result.total_iterations or 0,
                "optimization_time": float(result.optimization_time_seconds or 0),
                "last_updated": result.last_updated.isoformat()
                if result.last_updated
                else None,
            }
            for result in results
        ]

    def get_ai_recommendations_with_assets(
        self,
        portfolio_name: str = None,
        quarter: str = None,
        year: int = None,
        risk_tolerance: str = None,
    ) -> dict:
        """Get AI recommendations with associated asset recommendations."""
        query = self.db_session.query(AIRecommendation)

        # Apply filters
        if portfolio_name:
            query = query.filter(AIRecommendation.portfolio_name == portfolio_name)
        if quarter:
            query = query.filter(AIRecommendation.quarter == quarter)
        if year:
            query = query.filter(AIRecommendation.year == year)
        if risk_tolerance:
            query = query.filter(AIRecommendation.risk_tolerance == risk_tolerance)

        # Get latest recommendation
        ai_rec = query.order_by(AIRecommendation.created_at.desc()).first()

        if not ai_rec:
            return {}

        # Get associated asset recommendations
        asset_recs = (
            self.db_session.query(DbAssetRecommendation)
            .filter_by(ai_recommendation_id=ai_rec.id)
            .all()
        )

        return {
            "portfolio_info": {
                "portfolio_name": ai_rec.portfolio_name,
                "quarter": ai_rec.quarter,
                "year": ai_rec.year,
                "risk_tolerance": ai_rec.risk_tolerance,
                "total_score": float(ai_rec.total_score or 0),
                "confidence": float(ai_rec.confidence or 0),
                "diversification_score": float(ai_rec.diversification_score or 0),
                "total_assets": ai_rec.total_assets,
                "expected_return": float(ai_rec.expected_return or 0),
                "portfolio_risk": float(ai_rec.portfolio_risk or 0),
                "overall_reasoning": ai_rec.overall_reasoning,
                "warnings": ai_rec.warnings or [],
                "correlation_analysis": ai_rec.correlation_analysis or {},
                "created_at": ai_rec.created_at.isoformat()
                if ai_rec.created_at
                else None,
            },
            "asset_recommendations": [
                {
                    "symbol": asset.symbol,
                    "allocation_percentage": float(asset.allocation_percentage),
                    "confidence_score": float(asset.confidence_score),
                    "performance_score": float(asset.performance_score),
                    "risk_score": float(asset.risk_score),
                    "reasoning": asset.reasoning,
                    "red_flags": asset.red_flags or [],
                    "risk_per_trade": float(asset.risk_per_trade or 0),
                    "stop_loss_pct": float(asset.stop_loss_pct or 0),
                    "take_profit_pct": float(asset.take_profit_pct or 0),
                    "position_size_usd": float(asset.position_size_usd or 0),
                }
                for asset in asset_recs
            ],
        }

    def get_performance_summary(
        self, quarter: str = None, year: int = None, top_n: int = 20
    ) -> dict:
        """Get performance summary from best strategies and optimizations."""

        # Get top strategies
        best_strategies = self.get_best_strategies(quarter, year)[:top_n]

        # Get top optimizations
        best_optimizations = self.get_best_optimizations(quarter, year)[:top_n]

        # Calculate summary statistics
        if best_strategies:
            avg_sortino = sum(s["sortino_ratio"] for s in best_strategies) / len(
                best_strategies
            )
            avg_return = sum(s["total_return"] for s in best_strategies) / len(
                best_strategies
            )
            avg_drawdown = sum(s["max_drawdown"] for s in best_strategies) / len(
                best_strategies
            )
        else:
            avg_sortino = avg_return = avg_drawdown = 0

        return {
            "summary": {
                "total_strategies": len(best_strategies),
                "total_optimizations": len(best_optimizations),
                "avg_sortino_ratio": avg_sortino,
                "avg_total_return": avg_return,
                "avg_max_drawdown": avg_drawdown,
                "period": f"{quarter}_{year}" if quarter and year else "all_time",
            },
            "top_strategies": best_strategies,
            "top_optimizations": best_optimizations,
        }

    def _get_quarter_dates(self, quarter: str, year: int) -> tuple[datetime, datetime]:
        """Convert quarter string to date range."""
        if not quarter or len(quarter) < 2 or not quarter.startswith("Q"):
            raise ValueError(
                f"Invalid quarter format: '{quarter}'. Expected format: Q1, Q2, Q3, or Q4"
            )
        quarter_num = int(quarter[1])  # Extract number from Q1, Q2, etc.
        start_month = (quarter_num - 1) * 3 + 1
        end_month = quarter_num * 3

        # Ensure year is an integer
        year_int = int(year) if isinstance(year, str) else year
        start_date = datetime(year_int, start_month, 1)
        if quarter_num == 4:
            end_date = datetime(year_int + 1, 1, 1)
        else:
            end_date = datetime(year_int, end_month + 1, 1)

        return start_date, end_date
