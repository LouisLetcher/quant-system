from sqlalchemy import Column, Integer, String, Text
from database.db import Base

class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    parameters = Column(Text, nullable=True)  # JSON serialized parameters