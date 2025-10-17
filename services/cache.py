import redis.asyncio as redis
import json
from typing import Any, Optional
import os

class RedisCache:
    def __init__(self):
        self.redis_client = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0)),
                decode_responses=True
            )
            # Test connection
            await self.redis_client.ping()
            print("Redis connected successfully")
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        if not self.redis_client:
            await self.initialize()
            if not self.redis_client:
                return None
        
        try:
            data = await self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None
    
    async def set(self, key: str, value: Any, expire: int = 3600) -> None:
        if not self.redis_client:
            await self.initialize()
            if not self.redis_client:
                return
        
        try:
            await self.redis_client.setex(
                key, expire, json.dumps(value)
            )
        except Exception as e:
            print(f"Redis set error: {e}")
    
    async def cleanup(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()