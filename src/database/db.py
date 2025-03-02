from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from utils.config import config

DATABASE_URL = config.get("database")["url"]

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()