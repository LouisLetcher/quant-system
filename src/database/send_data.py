from __future__ import annotations

import json
import logging
from typing import Any

import requests
from sqlalchemy.orm import Session

from src.database.db_connection import get_db_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSender:
    """Handles sending data to various endpoints such as APIs, databases, and messaging systems."""

    @staticmethod
    def send_to_api(
        endpoint: str, data: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict:
        """Sends data to an external API endpoint."""
        headers = headers or {"Content-Type": "application/json"}
        try:
            response = requests.post(endpoint, json=data, headers=headers, timeout=10)
            response.raise_for_status()
            logger.info("✅ Successfully sent data to %s", endpoint)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("❌ API request failed: %s", e)
            return {"status": "error", "message": str(e)}

    @staticmethod
    def save_to_database(
        data: dict[str, Any], table_model, session: Session | None = None
    ):
        """Saves data to a database table using SQLAlchemy."""
        if session is None:
            session = get_db_session()
            
        try:
            record = table_model(**data)
            session.add(record)
            session.commit()
            logger.info("✅ Data successfully saved to %s", table_model.__tablename__)
            return {"status": "success", "message": "Data saved successfully"}
        except Exception as e:
            session.rollback()
            logger.error("❌ Database save failed: %s", e)
            return {"status": "error", "message": str(e)}
        finally:
            session.close()

    @staticmethod
    def send_to_messaging_queue(queue_name: str, data: dict[str, Any]):
        """Sends data to a messaging queue (RabbitMQ, Kafka, etc.)."""
        try:
            # Hypothetical message queue connection
            from src.messaging.queue_service import QueueService  # Hypothetical module

            QueueService.publish(queue_name, json.dumps(data))
            logger.info("✅ Data successfully sent to queue: %s", queue_name)
            return {"status": "success", "message": f"Data sent to queue {queue_name}"}
        except Exception as e:
            logger.error("❌ Messaging queue send failed: %s", e)
            return {"status": "error", "message": str(e)}
