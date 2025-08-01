import os
import redis
import logging

logger = logging.getLogger(__name__)

class RedisStorage:
    def __init__(self):
        self.redis_url = os.getenv('REDIS_URL')
        if not self.redis_url:
            raise Exception("REDIS_URL environment variable is not set")
        
        try:
            self.redis = redis.from_url(self.redis_url)
            # Test connection
            self.redis.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Error connecting to Redis: {str(e)}")
            raise
    
    async def add_bot_user(self, user_id: int):
        """Add user to the bot users list for future announcements"""
        try:
            key = "bot_users"
            # Add user_id to the set if it doesn't exist yet
            self.redis.sadd(key, user_id)
            logger.info(f"Added user {user_id} to bot users")
            return True
        except Exception as e:
            logger.error(f"Error adding user to bot users: {str(e)}")
            return False
    
    async def get_all_bot_users(self):
        """Get all users who have started the bot"""
        try:
            key = "bot_users"
            users = self.redis.smembers(key)
            # Convert bytes to integers
            return [int(user_id) for user_id in users]
        except Exception as e:
            logger.error(f"Error getting bot users: {str(e)}")
            return []
            
    async def save_user_rules(self, user_id: int, rules: str):
        """Save user rules for future conversations"""
        try:
            key = f"user_rules:{user_id}"
            self.redis.set(key, rules)
            logger.info(f"Saved rules for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving user rules: {str(e)}")
            return False
            
    async def get_user_rules(self, user_id: int):
        """Get user rules for conversations"""
        try:
            key = f"user_rules:{user_id}"
            rules = self.redis.get(key)
            # Convert bytes to string if exists
            return rules.decode('utf-8') if rules else None
        except Exception as e:
            logger.error(f"Error getting user rules: {str(e)}")
            return None
            
    async def delete_user_rules(self, user_id: int):
        """Delete user rules"""
        try:
            key = f"user_rules:{user_id}"
            self.redis.delete(key)
            logger.info(f"Deleted rules for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting user rules: {str(e)}")
            return False
