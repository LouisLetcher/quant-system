"""Database models for the Quant Trading System."""

from __future__ import annotations

import uuid
from datetime import datetime, date
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    Column, String, DateTime, Date, Numeric, BigInteger, Integer,
    Text, ARRAY, Boolean, Index, UniqueConstraint, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class PriceHistory(Base):
    """Market price data storage."""
    
    __tablename__ = 'price_history'
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', 'data_source', name='uq_price_symbol_timestamp_source'),
        Index('idx_price_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_price_data_source', 'data_source'),
        {'schema': 'market_data'}
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
    """Backtest results and performance metrics."""
    
    __tablename__ = 'results'
    __table_args__ = (
        Index('idx_backtest_created_at', 'created_at'),
        Index('idx_backtest_strategy', 'strategy'),
        Index('idx_backtest_sortino_ratio', 'sortino_ratio'),
        Index('idx_backtest_total_return', 'total_return'),
        {'schema': 'backtests'}
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
    calmar_ratio = Column(Numeric(10, 4))   # Secondary metric
    sharpe_ratio = Column(Numeric(10, 4))   # Tertiary metric
    profit_factor = Column(Numeric(10, 4))  # Supplementary metric
    
    # Risk metrics
    max_drawdown = Column(Numeric(10, 4))
    volatility = Column(Numeric(10, 4))
    downside_deviation = Column(Numeric(10, 4))
    
    # Performance metrics
    win_rate = Column(Numeric(10, 4))
    average_win = Column(Numeric(10, 4))
    average_loss = Column(Numeric(10, 4))
    trades_count = Column(Integer)
    
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
            'primary': self.sortino_ratio,
            'secondary': self.calmar_ratio,
            'tertiary': self.sharpe_ratio,
            'supplementary': self.profit_factor
        }


class PortfolioConfiguration(Base):
    """Portfolio configuration and settings."""
    
    __tablename__ = 'configurations'
    __table_args__ = (
        Index('idx_portfolio_name', 'name'),
        Index('idx_portfolio_created_at', 'created_at'),
        Index('idx_portfolio_optimization_metric', 'optimization_metric'),
        {'schema': 'portfolios'}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    symbols = Column(ARRAY(Text), nullable=False)
    weights = Column(ARRAY(Numeric))
    initial_capital = Column(Numeric(20, 2), nullable=False)
    commission = Column(Numeric(8, 6), default=0.001)
    slippage = Column(Numeric(8, 6), default=0.002)
    
    # Performance optimization settings (Sortino-first approach)
    optimization_metric = Column(String(50), default='sortino_ratio')
    secondary_metrics = Column(ARRAY(Text), default=['calmar_ratio', 'sharpe_ratio', 'profit_factor'])
    
    # Configuration details
    config = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    backtest_results = relationship("BacktestResult", back_populates="portfolio", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<PortfolioConfiguration(name='{self.name}', symbols={len(self.symbols)}, metric='{self.optimization_metric}')>"

    @property
    def is_sortino_optimized(self) -> bool:
        """Check if portfolio uses Sortino ratio as primary metric."""
        return self.optimization_metric == 'sortino_ratio'

    @property
    def metric_priority(self) -> List[str]:
        """Get metric priority list with primary metric first."""
        metrics = [self.optimization_metric]
        if self.secondary_metrics:
            metrics.extend([m for m in self.secondary_metrics if m != self.optimization_metric])
        return metrics


# Add relationship to BacktestResult
BacktestResult.portfolio_id = Column(UUID(as_uuid=True), ForeignKey('portfolios.configurations.id'))
BacktestResult.portfolio = relationship("PortfolioConfiguration", back_populates="backtest_results")


class OptimizationResult(Base):
    """Strategy optimization results."""
    
    __tablename__ = 'optimization_results'
    __table_args__ = (
        Index('idx_optimization_created_at', 'created_at'),
        Index('idx_optimization_portfolio_id', 'portfolio_id'),
        Index('idx_optimization_best_sortino', 'best_sortino_ratio'),
        {'schema': 'backtests'}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey('portfolios.configurations.id'), nullable=False)
    strategy = Column(String(100), nullable=False)
    optimization_metric = Column(String(50), default='sortino_ratio')
    
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
