import time
from typing import Dict

class RateLimiter:
    def __init__(self):
        self.requests: Dict[str, list] = {}
    
    async def check_limit(self, user_id: str, max_requests: int = 100, window_seconds: int = 3600) -> bool:
        """Simple in-memory rate limiter"""
        current_time = time.time()
        
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Remove old requests outside the window
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if current_time - req_time < window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[user_id]) < max_requests:
            self.requests[user_id].append(current_time)
            return True
        
        return False