from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database.db import Base

class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    pnl = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)

    strategy = relationship("Strategy", back_populates="backtest_results")
    asset = relationship("Asset", back_populates="backtest_results")