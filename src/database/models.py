"""Database models for the Quant Trading System."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class PriceHistory(Base):
    """Market price data storage."""

    __tablename__ = "price_history"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "timestamp",
            "data_source",
            name="uq_price_symbol_timestamp_source",
        ),
        Index("idx_price_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_price_data_source", "data_source"),
        {"schema": "market_data"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Numeric(20, 8), nullable=False)
    high = Column(Numeric(20, 8), nullable=False)
    low = Column(Numeric(20, 8), nullable=False)
    close = Column(Numeric(20, 8), nullable=False)
    volume = Column(BigInteger)
    data_source = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<PriceHistory(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"


class BacktestResult(Base):
    """Legacy backtest results for backward compatibility."""

    __tablename__ = "results"
    __table_args__ = (
        Index("idx_backtest_created_at", "created_at"),
        Index("idx_backtest_strategy", "strategy"),
        Index("idx_backtest_sortino_ratio", "sortino_ratio"),
        Index("idx_backtest_total_return", "total_return"),
        {"schema": "backtests"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    symbols = Column(ARRAY(Text), nullable=False)
    strategy = Column(String(100), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    initial_capital = Column(Numeric(20, 2), nullable=False)
    final_value = Column(Numeric(20, 2), nullable=False)
    total_return = Column(Numeric(10, 4), nullable=False)

    # Primary metrics (Sortino-focused approach)
    sortino_ratio = Column(Numeric(10, 4))  # Primary metric
    calmar_ratio = Column(Numeric(10, 4))  # Secondary metric
    sharpe_ratio = Column(Numeric(10, 4))  # Tertiary metric
    profit_factor = Column(Numeric(10, 4))  # Supplementary metric

    # Risk metrics
    max_drawdown = Column(Numeric(10, 4))
    volatility = Column(Numeric(10, 4))

    # Performance metrics
    win_rate = Column(Numeric(10, 4))
    num_trades = Column(Integer)

    # Advanced metrics
    alpha = Column(Numeric(10, 4))
    beta = Column(Numeric(10, 4))
    expectancy = Column(Numeric(10, 4))
    average_win = Column(Numeric(10, 4))
    average_loss = Column(Numeric(10, 4))
    total_fees = Column(Numeric(20, 2))
    portfolio_turnover = Column(Numeric(10, 4))
    strategy_capacity = Column(Numeric(20, 2))

    # Configuration and metadata
    parameters = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<BacktestResult(name='{self.name}', sortino_ratio={self.sortino_ratio}, total_return={self.total_return})>"

    @property
    def primary_metric(self) -> Optional[Decimal]:
        """Get the primary performance metric (Sortino ratio)."""
        return self.sortino_ratio

    @property
    def metric_hierarchy(self) -> dict:
        """Get all metrics in order of importance."""
        return {
            "primary": self.sortino_ratio,
            "secondary": self.calmar_ratio,
            "tertiary": self.sharpe_ratio,
            "supplementary": self.profit_factor,
        }


class BestStrategy(Base):
    """Best strategies per asset - curated data for recommendations."""

    __tablename__ = "best_strategies"
    __table_args__ = (
        Index("idx_best_strategies_symbol", "symbol"),
        Index("idx_best_strategies_sortino", "sortino_ratio"),
        Index("idx_best_strategies_risk", "risk_score"),
        UniqueConstraint(
            "symbol", "timeframe", name="uq_best_strategy_symbol_timeframe"
        ),
        {"schema": "backtests"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    best_strategy = Column(String(100), nullable=False)

    # Best performance metrics
    sortino_ratio = Column(Numeric(10, 4), nullable=False)
    sharpe_ratio = Column(Numeric(10, 4))
    calmar_ratio = Column(Numeric(10, 4))
    profit_factor = Column(Numeric(10, 4))
    total_return = Column(Numeric(10, 4), nullable=False)
    max_drawdown = Column(Numeric(10, 4))
    volatility = Column(Numeric(10, 4))
    win_rate = Column(Numeric(10, 4))
    num_trades = Column(Integer)

    # Advanced metrics (same as BacktestResult)
    alpha = Column(Numeric(10, 4))
    beta = Column(Numeric(10, 4))
    expectancy = Column(Numeric(10, 4))
    average_win = Column(Numeric(10, 4))
    average_loss = Column(Numeric(10, 4))
    total_fees = Column(Numeric(20, 2))
    portfolio_turnover = Column(Numeric(10, 4))
    strategy_capacity = Column(Numeric(20, 2))

    # Risk metrics for recommendations
    risk_score = Column(Numeric(10, 4))  # Calculated risk based on volatility/drawdown
    correlation_group = Column(String(50))  # For portfolio correlation analysis

    # Trading parameters for position sizing
    risk_per_trade = Column(Numeric(10, 4))
    stop_loss_pct = Column(Numeric(10, 4))
    take_profit_pct = Column(Numeric(10, 4))

    # Best strategy parameters
    best_parameters = Column(JSONB)

    # Metadata
    backtest_run_id = Column(
        String(100), nullable=False
    )  # Links to BacktestResult.run_id
    last_updated = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<BestStrategy(symbol='{self.symbol}', strategy='{self.best_strategy}', sortino={self.sortino_ratio})>"


class PortfolioConfiguration(Base):
    """Portfolio configuration and settings."""

    __tablename__ = "configurations"
    __table_args__ = (
        Index("idx_portfolio_name", "name"),
        Index("idx_portfolio_created_at", "created_at"),
        Index("idx_portfolio_optimization_metric", "optimization_metric"),
        {"schema": "portfolios"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    symbols = Column(ARRAY(Text), nullable=False)
    weights = Column(ARRAY(Numeric))
    initial_capital = Column(Numeric(20, 2), nullable=False)
    commission = Column(Numeric(8, 6), default=0.001)
    slippage = Column(Numeric(8, 6), default=0.002)

    # Performance optimization settings (Sortino-first approach)
    optimization_metric = Column(String(50), default="sortino_ratio")
    secondary_metrics = Column(
        ARRAY(Text), default=["calmar_ratio", "sharpe_ratio", "profit_factor"]
    )

    # Configuration details
    config = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships (commented out - not in current schema)
    # backtest_results = relationship(
    #     "BacktestResult", back_populates="portfolio", lazy="dynamic"
    # )

    def __repr__(self) -> str:
        return f"<PortfolioConfiguration(name='{self.name}', symbols={len(self.symbols)}, metric='{self.optimization_metric}')>"

    @property
    def is_sortino_optimized(self) -> bool:
        """Check if portfolio uses Sortino ratio as primary metric."""
        return self.optimization_metric == "sortino_ratio"

    @property
    def metric_priority(self) -> List[str]:
        """Get metric priority list with primary metric first."""
        metrics = [self.optimization_metric]
        if self.secondary_metrics:
            metrics.extend(
                [m for m in self.secondary_metrics if m != self.optimization_metric]
            )
        return metrics


# Portfolio relationships (commented out - not in current schema)
# BacktestResult.portfolio_id = Column(
#     UUID(as_uuid=True), ForeignKey("portfolios.configurations.id")
# )
# BacktestResult.portfolio = relationship(
#     "PortfolioConfiguration", back_populates="backtest_results"
# )


class AllOptimizationResult(Base):
    """All optimization results - historical record of every optimization iteration."""

    __tablename__ = "all_optimization_results"
    __table_args__ = (
        Index("idx_all_opt_results_run_id", "run_id"),
        Index("idx_all_opt_results_symbol_strategy", "symbol", "strategy"),
        Index("idx_all_opt_results_sortino", "sortino_ratio"),
        UniqueConstraint(
            "run_id",
            "symbol",
            "strategy",
            "timeframe",
            "iteration_number",
            name="uq_opt_run_iteration",
        ),
        {"schema": "backtests"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    strategy = Column(String(100), nullable=False)
    timeframe = Column(String(10), nullable=False)

    # Parameter combination being tested
    parameters = Column(JSONB, nullable=False)

    # Performance metrics for this parameter set
    sortino_ratio = Column(Numeric(10, 4))
    sharpe_ratio = Column(Numeric(10, 4))
    calmar_ratio = Column(Numeric(10, 4))
    profit_factor = Column(Numeric(10, 4))
    total_return = Column(Numeric(10, 4))
    max_drawdown = Column(Numeric(10, 4))
    volatility = Column(Numeric(10, 4))
    win_rate = Column(Numeric(10, 4))
    num_trades = Column(Integer)

    # Optimization metadata
    iteration_number = Column(Integer)
    optimization_metric = Column(String(50), default="sortino_ratio")
    portfolio_name = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<AllOptimizationResult(symbol='{self.symbol}', strategy='{self.strategy}', iteration={self.iteration_number}, sortino={self.sortino_ratio})>"


class BestOptimizationResult(Base):
    """Best optimization results per asset/strategy - curated data."""

    __tablename__ = "best_optimization_results"
    __table_args__ = (
        Index("idx_best_opt_results_symbol", "symbol"),
        Index("idx_best_opt_results_strategy", "symbol", "strategy"),
        Index("idx_best_opt_results_sortino", "best_sortino_ratio"),
        UniqueConstraint(
            "symbol", "strategy", "timeframe", name="uq_best_opt_symbol_strategy"
        ),
        {"schema": "backtests"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False)
    strategy = Column(String(100), nullable=False)
    timeframe = Column(String(10), nullable=False)

    # Best performance metrics
    best_sortino_ratio = Column(Numeric(10, 4), nullable=False)
    best_sharpe_ratio = Column(Numeric(10, 4))
    best_calmar_ratio = Column(Numeric(10, 4))
    best_profit_factor = Column(Numeric(10, 4))
    best_total_return = Column(Numeric(10, 4))
    best_max_drawdown = Column(Numeric(10, 4))
    best_volatility = Column(Numeric(10, 4))
    best_win_rate = Column(Numeric(10, 4))
    best_num_trades = Column(Integer)

    # Optimization summary
    best_parameters = Column(JSONB, nullable=False)
    optimization_metric = Column(String(50), default="sortino_ratio")
    total_iterations = Column(Integer)
    optimization_time_seconds = Column(Numeric(10, 2))
    parameter_ranges = Column(JSONB)

    # Metadata
    optimization_run_id = Column(String(100), nullable=False)
    portfolio_name = Column(String(255))
    last_updated = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<BestOptimizationResult(symbol='{self.symbol}', strategy='{self.strategy}', best_sortino={self.best_sortino_ratio})>"


class OptimizationResult(Base):
    """Legacy strategy optimization results for backward compatibility."""

    __tablename__ = "optimization_results"
    __table_args__ = (
        Index("idx_optimization_created_at", "created_at"),
        Index("idx_optimization_portfolio_id", "portfolio_id"),
        Index("idx_optimization_best_sortino", "best_sortino_ratio"),
        {"schema": "backtests"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(
        UUID(as_uuid=True), ForeignKey("portfolios.configurations.id"), nullable=False
    )
    strategy = Column(String(100), nullable=False)
    optimization_metric = Column(String(50), default="sortino_ratio")

    # Best results found
    best_sortino_ratio = Column(Numeric(10, 4))
    best_calmar_ratio = Column(Numeric(10, 4))
    best_sharpe_ratio = Column(Numeric(10, 4))
    best_profit_factor = Column(Numeric(10, 4))
    best_parameters = Column(JSONB)

    # Optimization metadata
    iterations = Column(Integer)
    optimization_time = Column(Numeric(10, 2))  # seconds
    parameter_ranges = Column(JSONB)
    results_summary = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    portfolio = relationship("PortfolioConfiguration")

    def __repr__(self) -> str:
        return f"<OptimizationResult(strategy='{self.strategy}', best_sortino={self.best_sortino_ratio})>"


class AIRecommendation(Base):
    """AI-generated investment recommendations - portfolio-level insights."""

    __tablename__ = "ai_recommendations"
    __table_args__ = (
        Index("idx_ai_rec_created_at", "created_at"),
        Index("idx_ai_rec_portfolio", "portfolio_name"),
        Index("idx_ai_rec_quarter", "quarter", "year"),
        UniqueConstraint(
            "portfolio_name",
            "quarter",
            "year",
            "risk_tolerance",
            name="uq_ai_rec_portfolio_period",
        ),
        {"schema": "backtests"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_name = Column(String(255), nullable=False)
    quarter = Column(String(10), nullable=False)  # e.g., "Q3"
    year = Column(Integer, nullable=False)  # e.g., 2025
    risk_tolerance = Column(
        String(20), nullable=False
    )  # conservative, moderate, aggressive

    # Portfolio-level metrics
    total_score = Column(Numeric(10, 4))
    confidence = Column(Numeric(10, 4))
    diversification_score = Column(Numeric(10, 4))
    total_assets = Column(Integer)
    expected_return = Column(Numeric(10, 4))
    portfolio_risk = Column(Numeric(10, 4))

    # AI analysis
    overall_reasoning = Column(Text)
    warnings = Column(ARRAY(Text))
    correlation_analysis = Column(JSONB)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    llm_model = Column(String(50))  # Which AI model was used

    def __repr__(self) -> str:
        return f"<AIRecommendation(portfolio='{self.portfolio_name}', quarter='{self.quarter}_{self.year}', assets={self.total_assets})>"


class AssetRecommendation(Base):
    """Individual asset recommendations - detailed recommendations per asset."""

    __tablename__ = "asset_recommendations"
    __table_args__ = (
        Index("idx_asset_rec_ai_id", "ai_recommendation_id"),
        Index("idx_asset_rec_symbol", "symbol"),
        Index("idx_asset_rec_allocation", "allocation_percentage"),
        UniqueConstraint(
            "ai_recommendation_id", "symbol", name="uq_asset_rec_ai_symbol"
        ),
        {"schema": "backtests"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ai_recommendation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("backtests.ai_recommendations.id"),
        nullable=False,
    )
    symbol = Column(String(20), nullable=False)

    # Recommendation details
    allocation_percentage = Column(Numeric(10, 4), nullable=False)
    confidence_score = Column(Numeric(10, 4), nullable=False)
    performance_score = Column(Numeric(10, 4), nullable=False)
    risk_score = Column(Numeric(10, 4), nullable=False)

    # AI reasoning
    reasoning = Column(Text)
    red_flags = Column(ARRAY(Text))

    # Trading parameters
    risk_per_trade = Column(Numeric(10, 4))
    stop_loss_pct = Column(Numeric(10, 4))
    take_profit_pct = Column(Numeric(10, 4))
    position_size_usd = Column(Numeric(20, 2))

    # Links to best strategy
    best_strategy_id = Column(
        UUID(as_uuid=True), ForeignKey("backtests.best_strategies.id")
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    ai_recommendation = relationship("AIRecommendation")
    best_strategy = relationship("BestStrategy")

    def __repr__(self) -> str:
        return f"<AssetRecommendation(symbol='{self.symbol}', allocation={self.allocation_percentage}%, confidence={self.confidence_score})>"


class Trade(Base):
    """Individual trade records from backtest results."""

    __tablename__ = "trades"
    __table_args__ = (
        Index("idx_trade_backtest_id", "backtest_result_id"),
        Index("idx_trade_symbol", "symbol"),
        Index("idx_trade_datetime", "trade_datetime"),
        {"schema": "backtests"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backtest_result_id = Column(
        UUID(as_uuid=True), ForeignKey("backtests.results.id"), nullable=False
    )
    symbol = Column(String(20), nullable=False)

    # Trade details
    trade_datetime = Column(DateTime(timezone=True), nullable=False)
    trade_type = Column(String(10), nullable=False)  # BUY or SELL
    price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    value = Column(Numeric(20, 2), nullable=False)  # price * quantity

    # Portfolio state after trade
    equity_after_trade = Column(Numeric(20, 2), nullable=False)
    holdings_after_trade = Column(Numeric(20, 8), nullable=False)
    cash_after_trade = Column(Numeric(20, 2), nullable=False)

    # Trade performance
    fees = Column(Numeric(20, 2), default=0)
    net_profit = Column(Numeric(20, 2), default=0)  # P&L from this trade
    unrealized_pnl = Column(
        Numeric(20, 2), default=0
    )  # Unrealized P&L at time of trade

    # Strategy signal info
    signal_strength = Column(Numeric(10, 4))  # Signal strength if available
    entry_reason = Column(Text)  # Reason for entry/exit

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    backtest_result = relationship("BacktestResult")

    def __repr__(self) -> str:
        return f"<Trade(symbol='{self.symbol}', type='{self.trade_type}', price={self.price}, value=${self.value})>"
