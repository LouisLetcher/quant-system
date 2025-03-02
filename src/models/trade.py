from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database.db import Base

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    trade_type = Column(String, nullable=False)  # Buy or Sell
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    executed_at = Column(DateTime, nullable=False)

    asset = relationship("Asset", back_populates="trades")
    user = relationship("User", back_populates="trades")