from database.db import engine, Base
from models.asset import Asset
from models.backtest_result import BacktestResult
from models.trade import Trade
from models.user import User
from models.strategy import Strategy

def create_tables():
    """Creates all database tables."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")

if __name__ == "__main__":
    create_tables()