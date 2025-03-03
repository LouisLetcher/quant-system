import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserService:
    """Manages user accounts and authentication."""

    def __init__(self):
        self.users = {}  # Simulated user storage

    def register_user(self, username: str, password: str):
        """Registers a new user."""
        if username in self.users:
            logger.warning(f"⚠️ User {username} already exists.")
            return False

        self.users[username] = {"password": password, "api_keys": {}}
        logger.info(f"✅ User {username} registered.")
        return True

    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticates a user."""
        if username in self.users and self.users[username]["password"] == password:
            logger.info(f"✅ User {username} authenticated.")
            return True
        else:
            logger.warning(f"❌ Authentication failed for {username}.")
            return False

    def store_api_key(self, username: str, platform: str, api_key: str):
        """Stores API keys securely."""
        if username not in self.users:
            logger.warning(f"⚠️ User {username} not found.")
            return False

        self.users[username]["api_keys"][platform] = api_key
        logger.info(f"✅ API Key stored for {username} on {platform}.")
        return True