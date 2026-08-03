import time
import redis
from backend.app.config import settings

# Attempt to connect to Redis for Rate Limiting
try:
    redis_client = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)
    redis_client.ping()
except Exception:
    redis_client = None
    # Fallback to in-memory dict for local testing if redis is unavailable
    _in_memory_cache = {}

class RateLimiter:
    """
    Sliding window rate limiter: Max 1 alert per machine per 10 minutes (600 seconds)
    """
    def __init__(self, window_seconds: int = 600):
        self.window_seconds = window_seconds

    def check_and_record(self, machine_id: str, fault_type: str = "general") -> int:
        """Returns the duplicate count (1 means first time, >1 means deduplicated)."""
        key = f"alert_dedup:{machine_id}:{fault_type}"
        
        if redis_client:
            count = redis_client.incr(key)
            if count == 1:
                redis_client.expire(key, self.window_seconds)
            return count
        else:
            current_time = int(time.time())
            # Simple in-memory fallback for local testing
            record = _in_memory_cache.get(key)
            if record and (current_time - record['time']) < self.window_seconds:
                record['count'] += 1
                return record['count']
            _in_memory_cache[key] = {'time': current_time, 'count': 1}
            return 1

class ActionGuard:
    @staticmethod
    def require_human_confirmation(severity: str, human_confirmed: bool) -> bool:
        """
        Critical severity actions MUST have a human-in-the-loop confirmation.
        Returns True if allowed, False if blocked.
        """
        if severity.lower() == "critical" and not human_confirmed:
            return False
        return True
