import requests
import json
import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from src.database.db_connection import get_db_session  # Hypothetical DB session import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSender:
    """Handles sending data to various endpoints such as APIs, databases, and messaging systems."""

    @staticmethod
    def send_to_api(endpoint: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict:
        """Sends data to an external API endpoint."""
        headers = headers or {"Content-Type": "application/json"}
        try:
            response = requests.post(endpoint, json=data, headers=headers, timeout=10)
            response.raise_for_status()
            logger.info(f"✅ Successfully sent data to {endpoint}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    @staticmethod
    def save_to_database(data: Dict[str, Any], table_model, session: Optional[Session] = None):
        """Saves data to a database table using SQLAlchemy."""
        session = session or get_db_session()
        try:
            record = table_model(**data)
            session.add(record)
            session.commit()
            logger.info(f"✅ Data successfully saved to {table_model.__tablename__}")
            return {"status": "success", "message": "Data saved successfully"}
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Database save failed: {str(e)}")
            return {"status": "error", "message": str(e)}
        finally:
            session.close()

    @staticmethod
    def send_to_messaging_queue(queue_name: str, data: Dict[str, Any]):
        """Sends data to a messaging queue (RabbitMQ, Kafka, etc.)."""
        try:
            # Hypothetical message queue connection
            from src.messaging.queue_service import QueueService  # Hypothetical module
            QueueService.publish(queue_name, json.dumps(data))
            logger.info(f"✅ Data successfully sent to queue: {queue_name}")
            return {"status": "success", "message": f"Data sent to queue {queue_name}"}
        except Exception as e:
            logger.error(f"❌ Messaging queue send failed: {str(e)}")
            return {"status": "error", "message": str(e)}