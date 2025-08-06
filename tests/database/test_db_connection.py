"""Simple tests for database connection manager."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from src.database.db_connection import DatabaseManager


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        db_manager = DatabaseManager()
        assert db_manager._sync_engine is None
        assert db_manager._async_engine is None

    @patch.dict(
        "os.environ", {"DATABASE_URL": "postgresql://test:test@localhost/custom"}
    )
    def test_get_database_url_from_env(self):
        """Test database URL retrieval from environment."""
        db_manager = DatabaseManager()
        url = db_manager._get_database_url(async_mode=False)
        assert url == "postgresql://test:test@localhost/custom"

    def test_get_database_url_async_conversion(self):
        """Test async URL conversion."""
        db_manager = DatabaseManager()
        with patch.dict(
            "os.environ", {"DATABASE_URL": "postgresql://test:test@localhost/db"}
        ):
            async_url = db_manager._get_database_url(async_mode=True)
            assert "postgresql+asyncpg://" in async_url

    @patch("src.database.db_connection.create_async_engine")
    def test_async_engine_creation(self, mock_create_engine):
        """Test async engine creation."""
        mock_engine = MagicMock(spec=AsyncEngine)
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager()
        engine = db_manager.async_engine

        assert engine == mock_engine
        mock_create_engine.assert_called_once()

    @patch("src.database.db_connection.create_engine")
    def test_sync_engine_creation(self, mock_create_engine):
        """Test sync engine creation."""
        mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager()
        engine = db_manager.sync_engine

        assert engine == mock_engine
        mock_create_engine.assert_called_once()

    @patch("src.database.db_connection.create_engine")
    def test_sync_session_creation(self, mock_create_engine):
        """Test sync session creation."""
        mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager()
        session = db_manager.get_sync_session()

        assert session is not None
        mock_create_engine.assert_called_once()


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


class TestIntegration:
    """Integration tests for database connection workflow."""

    @patch("src.database.db_connection.create_async_engine")
    @patch("src.database.db_connection.create_engine")
    def test_basic_workflow(self, mock_create_sync, mock_create_async):
        """Test basic database connection workflow."""
        # Setup mocks
        mock_async_engine = MagicMock(spec=AsyncEngine)
        mock_sync_engine = MagicMock(spec=Engine)
        mock_create_async.return_value = mock_async_engine
        mock_create_sync.return_value = mock_sync_engine

        # Create manager
        db_manager = DatabaseManager()

        # Test async workflow
        async_engine = db_manager.async_engine
        assert async_engine == mock_async_engine

        # Test sync workflow
        sync_engine = db_manager.sync_engine
        assert sync_engine == mock_sync_engine

    @patch.dict("os.environ", {"DATABASE_URL": "sqlite:///test.db"})
    @patch("src.database.db_connection.create_engine")
    def test_environment_variable_usage(self, mock_create_engine):
        """Test that environment variables are properly used."""
        mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = mock_engine

        # Should use environment variable
        db_manager = DatabaseManager()
        url = db_manager._get_database_url(async_mode=False)
        assert "sqlite:///test.db" in url

        # Test engine creation
        engine = db_manager.sync_engine
        assert engine == mock_engine
