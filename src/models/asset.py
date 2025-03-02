from sqlalchemy import Column, Integer, String, Float
from database.db import Base

class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    market = Column(String, nullable=False)
    price = Column(Float, nullable=True)