import json
import time
import logging
import redis.asyncio as redis_async
from fastapi_limiter import FastAPILimiter
from app.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB, REDIS_TTL, CACHE_TTL

logger = logging.getLogger("NyayaGPT-API")

class RedisService:
    def __init__(self):
        self.client = None
    
    async def init_redis(self):
        """Initialize Redis connection with improved error handling for GCP"""
        try:
            # Create Redis connection URL
            if REDIS_PASSWORD:
                redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
            else:
                redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
            
            self.client = redis_async.from_url(
                redis_url, 
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.client.ping()
            
            # Initialize rate limiter only if Redis is working
            await FastAPILimiter.init(self.client)
            
            logger.info("Redis connection established")
            return self.client
        except Exception as e:
            logger.error(f"Redis initialization error: {str(e)}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")
    
    async def get_conversation(self, conversation_id):
        """Get conversation history from Redis with error handling"""
        if not self.client:
            logger.warning("Redis client not initialized - returning empty conversation history")
            return []
        
        try:
            conversation_data = await self.client.get(f"conv:{conversation_id}")
            if conversation_data:
                return json.loads(conversation_data)
            return []
        except Exception as e:
            logger.error(f"Error retrieving conversation: {str(e)}")
            return []

    async def save_message_to_conversation(self, conversation_id, message):
        """Save a single message to the conversation history with error handling"""
        if not self.client:
            logger.warning("Redis client not initialized - skipping message save")
            return
        
        try:
            conversation = await self.get_conversation(conversation_id)
            
            if "timestamp" not in message:
                message["timestamp"] = time.time()
            
            conversation.append(message)
            
            await self.client.setex(
                f"conv:{conversation_id}", 
                REDIS_TTL, 
                json.dumps(conversation)
            )
        except Exception as e:
            logger.error(f"Error saving message to conversation: {str(e)}")
    
    async def delete_conversation(self, conversation_id):
        """Delete a conversation by ID"""
        if not self.client:
            raise Exception("Redis client not initialized")
        
        try:
            deleted = await self.client.delete(f"conv:{conversation_id}")
            return deleted > 0
        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            raise
    
    async def get_cached_response(self, query: str, model_name: str, strategy: str):
        """Get cached response if available with error handling"""
        if not self.client:
            return None
        
        try:
            cache_key = f"cache:{hash(f'{query}:{model_name}:{strategy}')}"
            cached = await self.client.get(cache_key)
            
            if cached:
                logger.info(f"Cache hit for query: {query[:30]}...")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    async def cache_response(self, query: str, model_name: str, strategy: str, response_data: dict):
        """Cache response for future use with error handling"""
        if not self.client:
            return
        
        try:
            cache_key = f"cache:{hash(f'{query}:{model_name}:{strategy}')}"
            await self.client.setex(
                cache_key,
                CACHE_TTL,
                json.dumps(response_data)
            )
            logger.info(f"Cached response for query: {query[:30]}...")
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}")
    
    async def clear_cache(self):
        """Clear the response cache"""
        if not self.client:
            raise Exception("Redis client not initialized")
        
        try:
            cursor = 0
            deleted_count = 0
            
            while True:
                cursor, keys = await self.client.scan(cursor, match="cache:*")
                if keys:
                    deleted = await self.client.delete(*keys)
                    deleted_count += deleted
                
                if cursor == 0:
                    break
            
            return deleted_count
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise

# Global Redis service instance
redis_service = RedisService()