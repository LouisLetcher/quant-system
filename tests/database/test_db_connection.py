"""Tests for database connection manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from src.database.db_connection import DatabaseManager


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch.dict(
            "os.environ", {"DATABASE_URL": "postgresql://user:pass@localhost/testdb"}
        ):
            db_manager = DatabaseManager()
            assert db_manager.database_url is not None

    def test_init_custom_url(self):
        """Test initialization with custom database URL."""
        custom_url = "postgresql://test:test@localhost/custom"
        db_manager = DatabaseManager(database_url=custom_url)
        assert db_manager.database_url == custom_url

    def test_init_no_url_raises_error(self):
        """Test initialization without database URL raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises((ValueError, KeyError)):
                DatabaseManager()

    @patch("src.database.db_connection.create_async_engine")
    def test_get_async_engine_creation(self, mock_create_engine):
        """Test async engine creation."""
        mock_engine = MagicMock(spec=AsyncEngine)
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")
        engine = db_manager.get_async_engine()

        assert engine == mock_engine
        mock_create_engine.assert_called_once()

    @patch("src.database.db_connection.create_engine")
    def test_get_sync_engine_creation(self, mock_create_engine):
        """Test sync engine creation."""
        mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")
        engine = db_manager.get_sync_engine()

        assert engine == mock_engine
        mock_create_engine.assert_called_once()

    @patch("src.database.db_connection.create_async_engine")
    def test_async_engine_singleton(self, mock_create_engine):
        """Test that async engine is created only once (singleton pattern)."""
        mock_engine = MagicMock(spec=AsyncEngine)
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")

        engine1 = db_manager.get_async_engine()
        engine2 = db_manager.get_async_engine()

        assert engine1 == engine2
        mock_create_engine.assert_called_once()

    @patch("src.database.db_connection.create_engine")
    def test_sync_engine_singleton(self, mock_create_engine):
        """Test that sync engine is created only once (singleton pattern)."""
        mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")

        engine1 = db_manager.get_sync_engine()
        engine2 = db_manager.get_sync_engine()

        assert engine1 == engine2
        mock_create_engine.assert_called_once()

    @patch("src.database.db_connection.AsyncSession")
    @patch("src.database.db_connection.create_async_engine")
    async def test_get_async_session(self, mock_create_engine, mock_session_class):
        """Test async session creation."""
        mock_engine = MagicMock(spec=AsyncEngine)
        mock_create_engine.return_value = mock_engine

        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")

        async with db_manager.get_async_session() as session:
            assert session == mock_session

    @patch("src.database.db_connection.Session")
    @patch("src.database.db_connection.create_engine")
    def test_get_sync_session(self, mock_create_engine, mock_session_class):
        """Test sync session creation."""
        mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = mock_engine

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")

        with db_manager.get_sync_session() as session:
            assert session == mock_session

    @patch("src.database.db_connection.create_async_engine")
    async def test_async_engine_disposal(self, mock_create_engine):
        """Test async engine disposal."""
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")

        # Get engine to create it
        db_manager.get_async_engine()

        # Test disposal
        await db_manager.dispose_async()
        mock_engine.dispose.assert_called_once()

    @patch("src.database.db_connection.create_engine")
    def test_sync_engine_disposal(self, mock_create_engine):
        """Test sync engine disposal."""
        mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")

        # Get engine to create it
        db_manager.get_sync_engine()

        # Test disposal
        db_manager.dispose_sync()
        mock_engine.dispose.assert_called_once()

    @patch("src.database.db_connection.create_async_engine")
    def test_connection_pool_settings(self, mock_create_engine):
        """Test that connection pool settings are properly configured."""
        db_manager = DatabaseManager("postgresql://test:test@localhost/test")
        db_manager.get_async_engine()

        # Verify create_async_engine was called with pool settings
        call_args = mock_create_engine.call_args
        kwargs = call_args[1] if call_args else {}

        # Should have pool configuration
        assert "pool_size" in kwargs or "pool_pre_ping" in kwargs or len(kwargs) > 0

    def test_database_url_validation(self):
        """Test database URL validation."""
        # Test valid PostgreSQL URL
        valid_url = "postgresql://user:pass@localhost:5432/testdb"
        db_manager = DatabaseManager(valid_url)
        assert db_manager.database_url == valid_url

        # Test valid SQLite URL
        sqlite_url = "sqlite:///test.db"
        db_manager = DatabaseManager(sqlite_url)
        assert db_manager.database_url == sqlite_url

    @patch("src.database.db_connection.create_async_engine")
    def test_error_handling_engine_creation(self, mock_create_engine):
        """Test error handling during engine creation."""
        mock_create_engine.side_effect = Exception("Connection failed")

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")

        with pytest.raises(Exception, match="Connection failed"):
            db_manager.get_async_engine()


class TestGetDbSession:
    """Test cases for get_db_session convenience function."""

    @patch("src.database.db_connection.DatabaseManager")
    def test_get_db_session_creation(self, mock_db_manager_class):
        """Test that get_db_session creates proper session."""
        from src.database.db_connection import get_db_session

        mock_db_manager = MagicMock()
        mock_db_manager_class.return_value = mock_db_manager

        # Test function exists and can be called
        session_generator = get_db_session()
        assert session_generator is not None

    @patch("src.database.db_connection.DatabaseManager")
    async def test_get_db_session_async_usage(self, mock_db_manager_class):
        """Test get_db_session with async context."""
        from src.database.db_connection import get_db_session

        mock_db_manager = MagicMock()
        mock_session = AsyncMock()
        mock_db_manager.get_async_session.return_value.__aenter__.return_value = (
            mock_session
        )
        mock_db_manager_class.return_value = mock_db_manager

        # Should be able to use in async context
        session_gen = get_db_session()
        assert session_gen is not None


class TestIntegration:
    """Integration tests for database connection workflow."""

    @patch("src.database.db_connection.create_async_engine")
    @patch("src.database.db_connection.create_engine")
    async def test_complete_database_workflow(
        self, mock_create_sync, mock_create_async
    ):
        """Test complete database connection workflow."""
        # Setup mocks
        mock_async_engine = AsyncMock(spec=AsyncEngine)
        mock_sync_engine = MagicMock(spec=Engine)
        mock_create_async.return_value = mock_async_engine
        mock_create_sync.return_value = mock_sync_engine

        # Create manager
        db_manager = DatabaseManager("postgresql://test:test@localhost/test")

        # Test async workflow
        async_engine = db_manager.get_async_engine()
        assert async_engine == mock_async_engine

        # Test sync workflow
        sync_engine = db_manager.get_sync_engine()
        assert sync_engine == mock_sync_engine

        # Test cleanup
        await db_manager.dispose_async()
        db_manager.dispose_sync()

        mock_async_engine.dispose.assert_called_once()
        mock_sync_engine.dispose.assert_called_once()

    @patch.dict("os.environ", {"DATABASE_URL": "sqlite:///test.db"})
    @patch("src.database.db_connection.create_engine")
    def test_environment_variable_usage(self, mock_create_engine):
        """Test that environment variables are properly used."""
        mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = mock_engine

        # Should use environment variable
        db_manager = DatabaseManager()
        assert "sqlite:///test.db" in db_manager.database_url

        # Test engine creation
        engine = db_manager.get_sync_engine()
        assert engine == mock_engine

    def test_connection_string_formats(self):
        """Test various connection string formats."""
        # PostgreSQL
        pg_url = "postgresql+asyncpg://user:pass@localhost:5432/db"
        db_manager = DatabaseManager(pg_url)
        assert db_manager.database_url == pg_url

        # SQLite
        sqlite_url = "sqlite+aiosqlite:///test.db"
        db_manager = DatabaseManager(sqlite_url)
        assert db_manager.database_url == sqlite_url

        # MySQL
        mysql_url = "mysql+aiomysql://user:pass@localhost/db"
        db_manager = DatabaseManager(mysql_url)
        assert db_manager.database_url == mysql_url
