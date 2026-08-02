import pytest
import time
from unittest.mock import MagicMock
import sqlalchemy.schema
sqlalchemy.schema.MetaData.create_all = MagicMock() # Prevent module level DB connection
import redis
redis.Redis.ping = MagicMock()

from fastapi.testclient import TestClient
from backend.app.main import app
from sqlalchemy.exc import OperationalError
from backend.app.core.breaker import db_circuit_breaker
from backend.app.db.session import _ping_db
import torch

client = TestClient(app)

def test_db_circuit_breaker(monkeypatch):
    # Reset circuit breaker
    db_circuit_breaker.state = "CLOSED"
    db_circuit_breaker.failures = 0
    
    # Mock _ping_db to simulate DB down
    def mock_ping_db(*args, **kwargs):
        raise OperationalError("Simulated DB connection error", None, None)
        
    monkeypatch.setattr("backend.app.db.session._ping_db", mock_ping_db)
    
    # Send requests to an endpoint that requires DB (e.g. audit logs)
    # The first 3 requests should fail with 503 OperationalError (since threshold is 3)
    # The 4th request should fail with 503 Circuit Breaker OPEN
    headers = {"Authorization": "Bearer TEST_TOKEN"}
    
    # First request
    response = client.get("/audit/logs", headers=headers)
    assert response.status_code == 503
    assert "error" in response.text.lower()
    
    # Second request
    response = client.get("/audit/logs", headers=headers)
    assert response.status_code == 503
    
    # Third request
    response = client.get("/audit/logs", headers=headers)
    assert response.status_code == 503
    
    # Fourth request (Circuit breaker is OPEN)
    response = client.get("/audit/logs", headers=headers)
    assert response.status_code == 503
    assert "Circuit Breaker OPEN" in response.text
    
def test_gpu_oom_fallback(monkeypatch):
    # Mock inference predict to throw CUDA out of memory
    from backend.app.api.dependencies import get_model_container
    models = get_model_container()
    
    original_run_forward = None
    
    def mock_forward(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")
        
    # We will just patch the inner _run_forward if possible, or patch the model's forward
    # The easiest is to mock the gnn_model.forward to raise OOM
    monkeypatch.setattr(models.pipeline.gnn_model, "forward", mock_forward)
    
    payload = {
        "machine_id": "M_TEST",
        "timestamp": time.time(),
        "graph": {
            "nodes": [{"id": "M_TEST", "type": "machine", "features": [0.0] * 64}],
            "edges": []
        }
    }
    
    # Send predict request
    response = client.post("/graph/predict/full", json=payload)
    
    # Because of our fallback, it should eventually return 200 with latency_warning=True
    # Wait, in the fallback, we call _run_forward again. Since the mock always raises, 
    # it might raise again unless we only raise it once.
    # Let's create a side effect that raises once.
    call_count = 0
    def mock_forward_once(*args, **kwargs):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            raise RuntimeError("CUDA out of memory")
        return torch.zeros((1, 256))
        
    monkeypatch.setattr(models.pipeline.gnn_model, "forward", mock_forward_once)
    
    # Also we need to mock scorer_head since we return zeros
    def mock_scorer(*args, **kwargs):
        return torch.tensor([0.5])
    monkeypatch.setattr(models.pipeline.scorer_head, "forward", mock_scorer)
    
    # Send predict request
    response = client.post("/graph/predict/full", json=payload)
    assert response.status_code == 200
    
    # We need to wait for async job or use the eager celery app
    data = response.json()
    job_id = data["job_id"]
    status_response = client.get(f"/graph/jobs/{job_id}")
    assert status_response.status_code == 200
    result_data = status_response.json()["result"]
    assert result_data["latency_warning"] == True
    
def test_redis_down_limiter_fails_open(monkeypatch):
    # Mock redis ping or get to raise connection error
    import redis
    def mock_ping(*args, **kwargs):
        raise redis.ConnectionError("Simulated Redis connection error")
        
    monkeypatch.setattr("redis.Redis.ping", mock_ping)
    monkeypatch.setattr("redis.Redis.get", mock_ping)
    monkeypatch.setattr("redis.Redis.set", mock_ping)
    monkeypatch.setattr("redis.Redis.setex", mock_ping)
    
    # Send request to a rate-limited endpoint
    # The limiter should swallow the error and allow the request
    # Since we mocked redis globally, token verification might fail if it checks blacklist.
    # Let's mock is_token_blacklisted too.
    monkeypatch.setattr("backend.app.core.security.is_token_blacklisted", lambda *args: False)
    
    # We use /health which has a limit
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
