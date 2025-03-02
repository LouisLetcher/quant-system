import logging
import os
from logging.config import dictConfig

def setup_logging():
    log_file = "logs/app.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": log_file,
                "formatter": "default",
                "level": "DEBUG",
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
        },
    }

    dictConfig(logging_config)

setup_logging()
logger = logging.getLogger(__name__)