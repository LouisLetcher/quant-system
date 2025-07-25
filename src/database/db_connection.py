"""Database connection management for PostgreSQL."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self):
        self._sync_engine: Optional[Engine] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._async_session_factory: Optional[sessionmaker] = None

    def _get_database_url(self, async_mode: bool = False) -> str:
        """Get database URL from environment variables."""
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://quantuser:quantpass@localhost:5432/quant_system",
        )

        if async_mode and database_url.startswith("postgresql://"):
            # Convert to async URL
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        elif not async_mode and database_url.startswith("postgresql+asyncpg://"):
            # Convert to sync URL
            database_url = database_url.replace(
                "postgresql+asyncpg://", "postgresql://", 1
            )

        return database_url

    @property
    def sync_engine(self) -> Engine:
        """Get synchronous database engine."""
        if self._sync_engine is None:
            database_url = self._get_database_url(async_mode=False)
            self._sync_engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=os.getenv("LOG_LEVEL") == "DEBUG",
            )
            logger.info("✅ Synchronous database engine created")
        return self._sync_engine

    @property
    def async_engine(self) -> AsyncEngine:
        """Get asynchronous database engine."""
        if self._async_engine is None:
            database_url = self._get_database_url(async_mode=True)
            self._async_engine = create_async_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=os.getenv("LOG_LEVEL") == "DEBUG",
            )
            logger.info("✅ Asynchronous database engine created")
        return self._async_engine

    def get_sync_session(self) -> Session:
        """Create a new synchronous database session."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.sync_engine, autocommit=False, autoflush=False
            )
        return self._session_factory()

    def get_async_session_factory(self) -> sessionmaker:
        """Get async session factory."""
        if self._async_session_factory is None:
            self._async_session_factory = sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
            )
        return self._async_session_factory

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup."""
        session_factory = self.get_async_session_factory()
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self):
        """Close all database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
            logger.info("✅ Async database engine closed")

        if self._sync_engine:
            self._sync_engine.dispose()
            logger.info("✅ Sync database engine closed")


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
def get_db_session() -> Session:
    """Get a synchronous database session."""
    return db_manager.get_sync_session()


async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an asynchronous database session."""
    async with db_manager.get_async_session() as session:
        yield session


def get_sync_engine() -> Engine:
    """Get the synchronous database engine."""
    return db_manager.sync_engine


def get_async_engine() -> AsyncEngine:
    """Get the asynchronous database engine."""
    return db_manager.async_engine
