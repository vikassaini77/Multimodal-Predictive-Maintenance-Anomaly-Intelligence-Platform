import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_rate_limit_health_endpoint():
    # The /health endpoint has a limit of 1000/minute. 
    # To test this quickly without making 1000 requests, we can just verify that 
    # it doesn't limit us initially and headers are set.
    # However, we can mock the limiter or just hit it a few times to check headers.
    
    response = client.get("/health")
    assert response.status_code == 200
    # slowapi sets headers like X-RateLimit-Limit, X-RateLimit-Remaining
    assert "X-RateLimit-Limit" in response.headers
    assert response.headers["X-RateLimit-Limit"] == "1000"
    
def test_rate_limit_predict_endpoint_unauthorized():
    # Because predict requires auth, we should get 401. 
    # But slowapi might limit before auth if middleware order is different, 
    # or after. FastAPI dependencies usually run before the endpoint body.
    # But @limiter.limit intercepts it. Let's see if we get the rate limit headers.
    
    response = client.post("/graph/predict", json={"nodes": [], "edges": []})
    # We expect a 401 because we didn't provide a JWT
    assert response.status_code == 401

def test_rate_limit_exceeded_custom_429(monkeypatch):
    # Let's forcibly trigger a 429 by mocking the limiter's check method
    from backend.app.core.security import limiter
    
    def mock_check(*args, **kwargs):
        from slowapi.errors import RateLimitExceeded
        from limits import RateLimitItemPerMinute
        raise RateLimitExceeded(RateLimitItemPerMinute(100))
        
    monkeypatch.setattr(limiter, "_check_request_limit", mock_check)
    
    response = client.get("/health")
    assert response.status_code == 429
    assert "Retry-After" in response.headers
    assert "Rate limit exceeded" in response.json()["error"]
