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

    def check_and_record(self, machine_id: str) -> bool:
        """Returns True if action is allowed, False if rate limited."""
        key = f"alert_rate_limit:{machine_id}"
        current_time = int(time.time())
        
        if redis_client:
            last_alert = redis_client.get(key)
            if last_alert and (current_time - int(last_alert)) < self.window_seconds:
                return False
            redis_client.set(key, current_time, ex=self.window_seconds)
            return True
        else:
            last_alert = _in_memory_cache.get(key)
            if last_alert and (current_time - int(last_alert)) < self.window_seconds:
                return False
            _in_memory_cache[key] = current_time
            return True

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
