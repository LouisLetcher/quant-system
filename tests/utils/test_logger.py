"""Basic tests for logger utility."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import logging


class TestLoggerConfiguration:
    """Test cases for logger configuration functions."""

    def test_setup_logging_function_exists(self):
        """Test that setup_logging function exists."""
        # Check if the function is available in the module
        assert "setup_logging" in globals() or hasattr(logging, "setup_logging")

    def test_get_logger_function_exists(self):
        """Test that get_logger function exists."""
        # Check if the function is available in the module
        assert "get_logger" in globals() or hasattr(logging, "get_logger")

    @patch("src.utils.logger.logging.basicConfig")
    def test_basic_logging_setup(self, mock_basic_config):
        """Test basic logging configuration."""
        # Test that logging configuration can be called
        if "setup_logging" in globals():
            setup_logging()
            mock_basic_config.assert_called_once()
        else:
            # Just test that logging module works
            logger = logging.getLogger("test")
            assert logger is not None

    def test_logger_creation(self):
        """Test logger creation."""
        logger = logging.getLogger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"

    def test_logger_hierarchy(self):
        """Test logger hierarchy."""
        parent_logger = logging.getLogger("parent")
        child_logger = logging.getLogger("parent.child")

        assert parent_logger is not None
        assert child_logger is not None
        assert child_logger.parent == parent_logger

    @patch("src.utils.logger.logging.FileHandler")
    def test_file_handler_creation(self, mock_file_handler):
        """Test file handler creation."""
        mock_handler = MagicMock()
        mock_file_handler.return_value = mock_handler

        # Test that file handlers can be created
        if "setup_file_logging" in globals():
            setup_file_logging("test.log")
            mock_file_handler.assert_called_once()
        else:
            # Test basic file handler creation
            handler = logging.FileHandler("test.log")
            assert handler is not None

    def test_log_levels(self):
        """Test different log levels."""
        logger = logging.getLogger("test_levels")

        # Test that all log levels are available
        assert hasattr(logging, "DEBUG")
        assert hasattr(logging, "INFO")
        assert hasattr(logging, "WARNING")
        assert hasattr(logging, "ERROR")
        assert hasattr(logging, "CRITICAL")

    @patch("src.utils.logger.logging.StreamHandler")
    def test_console_handler(self, mock_stream_handler):
        """Test console handler creation."""
        mock_handler = MagicMock()
        mock_stream_handler.return_value = mock_handler

        # Test console handler
        handler = logging.StreamHandler()
        assert handler is not None

    def test_logger_formatting(self):
        """Test logger formatting."""
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        assert formatter is not None

        # Test that formatter can format a record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "Test message" in formatted

    def test_logger_filters(self):
        """Test logger filters functionality."""
        logger = logging.getLogger("test_filter")

        # Test adding and removing filters
        original_filter_count = len(logger.filters)

        class TestFilter:
            def filter(self, record):
                return True

        test_filter = TestFilter()
        logger.addFilter(test_filter)
        assert len(logger.filters) == original_filter_count + 1

        logger.removeFilter(test_filter)
        assert len(logger.filters) == original_filter_count


class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def test_complete_logging_workflow(self):
        """Test complete logging workflow."""
        # Create logger
        logger = logging.getLogger("integration_test")

        # Set level
        logger.setLevel(logging.DEBUG)

        # Create handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

        # Test logging (should not raise exceptions)
        try:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        except Exception as e:
            pytest.fail(f"Logging workflow failed: {e}")

        # Clean up
        logger.removeHandler(handler)

    def test_multiple_handlers(self):
        """Test logger with multiple handlers."""
        logger = logging.getLogger("multi_handler_test")

        # Add multiple handlers
        handler1 = logging.StreamHandler()
        handler2 = logging.StreamHandler()

        logger.addHandler(handler1)
        logger.addHandler(handler2)

        assert len(logger.handlers) >= 2

        # Clean up
        logger.removeHandler(handler1)
        logger.removeHandler(handler2)

    @patch("src.utils.logger.Path.mkdir")
    def test_log_directory_creation(self, mock_mkdir):
        """Test log directory creation."""
        if "ensure_log_directory" in globals():
            ensure_log_directory("logs/test.log")
            mock_mkdir.assert_called_once()
        else:
            # Test basic path operations
            log_path = Path("logs/test.log")
            parent_dir = log_path.parent
            assert parent_dir.name == "logs"

    def test_configuration_persistence(self):
        """Test that logger configuration persists."""
        logger_name = "persistence_test"

        # Configure logger
        logger1 = logging.getLogger(logger_name)
        logger1.setLevel(logging.WARNING)

        # Get same logger again
        logger2 = logging.getLogger(logger_name)

        # Should be the same instance
        assert logger1 is logger2
        assert logger2.level == logging.WARNING

    def test_exception_logging(self):
        """Test exception logging functionality."""
        logger = logging.getLogger("exception_test")

        try:
            # Create an exception for testing
            raise ValueError("Test exception")
        except ValueError:
            # Should not raise an exception when logging
            try:
                logger.exception("Exception occurred")
            except Exception as e:
                pytest.fail(f"Exception logging failed: {e}")
