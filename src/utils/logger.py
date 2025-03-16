import logging
import os
import sys
from datetime import datetime
from typing import Optional, Union

from src.utils.config_manager import ConfigManager


class Logger:
    """
    Centralized logging utility for the quant system.
    Provides consistent logging across all modules and CLI commands.
    """
    
    # Log levels mapping
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO, 
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    # Initialize class variables
    _initialized = False
    _config = None
    _cli_log_file = None
    
    @classmethod
    def initialize(cls, config: Optional[ConfigManager] = None):
        """Initialize the logging system with configuration."""
        if cls._initialized:
            return
            
        # Load config if not provided
        if config is None:
            cls._config = ConfigManager()
        else:
            cls._config = config
            
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(cls._config.config["logging"]["log_file"])
        os.makedirs(log_dir, exist_ok=True)
        
        # Create CLI logs directory
        cli_log_dir = os.path.join(log_dir, "cli")
        os.makedirs(cli_log_dir, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(cls.LOG_LEVELS.get(
            cls._config.config["logging"]["level"].upper(), 
            logging.INFO
        ))
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add file handler for the main log file
        file_handler = logging.FileHandler(cls._config.config["logging"]["log_file"])
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        cls._initialized = True
        
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger for a specific module."""
        if not cls._initialized:
            cls.initialize()
        return logging.getLogger(name)
    
    @classmethod
    def setup_cli_logging(cls, command_name: str) -> str:
        """Set up logging for CLI commands"""
        if not cls._initialized:
            cls.initialize()

        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/{command_name}_{timestamp}.log"
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)
        
        cls._cli_log_file = log_file
        
        return cls._cli_log_file    
    @classmethod
    def capture_stdout(cls) -> None:
        """
        Capture stdout and stderr to the logger.
        Call this after setup_cli_logging for full CLI output capture.
        """
        if not cls._initialized or not cls._cli_log_file:
            return
            
        # Create stdout/stderr redirectors
        class LoggerWriter:
            def __init__(self, logger, level):
                self.logger = logger
                self.level = level
                self.buffer = ''
        
            def write(self, message):
                if message and not message.isspace():
                    try:
                        self.buffer += message
                        if '\n' in self.buffer:
                            self.flush()
                    except UnicodeEncodeError:
                        # Handle Unicode encoding errors by replacing problematic characters
                        safe_message = message.encode('ascii', 'replace').decode('ascii')
                        self.buffer += safe_message
                        if '\n' in self.buffer:
                            self.flush()
                
            def flush(self):
                if self.buffer:
                    try:
                        for line in self.buffer.rstrip().splitlines():
                            self.level(line)
                    except UnicodeEncodeError:
                        # Handle Unicode encoding errors by replacing problematic characters
                        safe_buffer = self.buffer.encode('ascii', 'replace').decode('ascii')
                        for line in safe_buffer.rstrip().splitlines():
                            self.level(line)
                    self.buffer = ''                            
                    self.flush()
                
            def flush(self):
                if self.buffer:
                    try:
                        for line in self.buffer.rstrip().splitlines():
                            self.level(line)
                    except UnicodeEncodeError:
                        # Handle Unicode encoding errors by replacing problematic characters
                        safe_buffer = self.buffer.encode('ascii', 'replace').decode('ascii')
                        for line in safe_buffer.rstrip().splitlines():
                            self.level(line)
                    self.buffer = ''        # Get the root logger
        logger = logging.getLogger()
        
        # Save original stdout/stderr
        cls._orig_stdout = sys.stdout
        cls._orig_stderr = sys.stderr
        
        # Redirect stdout and stderr
        sys.stdout = LoggerWriter(logger, logger.info)
        sys.stderr = LoggerWriter(logger, logger.error)
    
    @classmethod
    def restore_stdout(cls) -> None:
        """Restore original stdout and stderr."""
        if hasattr(cls, '_orig_stdout') and hasattr(cls, '_orig_stderr'):
            sys.stdout = cls._orig_stdout
            sys.stderr = cls._orig_stderr


# Create a convenience function for getting a logger
def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return Logger.get_logger(name)

def setup_command_logging(args):
    """Set up logging based on command arguments"""
    if hasattr(args, "log") and args.log:
        # Initialize logger if needed
        Logger.initialize()
        
        # Get command name
        command = args.command if hasattr(args, "command") else "unknown"
        
        # Setup CLI logging
        log_file = Logger.setup_cli_logging(command)
        
        # Capture stdout/stderr
        Logger.capture_stdout()
        
        print(f"📝 Logging enabled. Output will be saved to: {log_file}")
        
        return log_file
    return None