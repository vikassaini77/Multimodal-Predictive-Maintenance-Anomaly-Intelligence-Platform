import pytest
from fastapi.testclient import TestClient
import time
from backend.app.main import app

client = TestClient(app)

def test_predict_full_pipeline_success():
    # Mock full request
    payload = {
        "machine_id": "M1",
        "timestamp": time.time(),
        "sensor_data": [[0.5] * 10 for _ in range(14)], # 14 channels, 10 sequence steps
        "visual_data": [[[0.1] * 224 for _ in range(224)] for _ in range(3)], # 3x224x224
        "graph": {
            "nodes": [
                {"id": "M1", "type": "machine", "features": [0.0] * 64},
                {"id": "C1", "type": "conveyor", "features": [0.0] * 64}
            ],
            "edges": [
                {"source": "M1", "target": "C1", "type": "feeds_into"}
            ]
        }
    }
    
    response = client.post("/graph/predict/full", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["machine_id"] == "M1"
    assert "anomaly_score" in data
    assert "is_anomaly" in data
    assert "threshold" in data
    assert "cache_hit" in data
    assert data["cache_hit"] == False # First time should be false

def test_predict_full_pipeline_caching():
    # Sending same payload twice should trigger cache hit
    timestamp = time.time()
    payload = {
        "machine_id": "M2",
        "timestamp": timestamp,
        "sensor_data": [[0.5] * 10 for _ in range(14)], 
        "visual_data": [[[0.1] * 224 for _ in range(224)] for _ in range(3)], 
        "graph": {
            "nodes": [
                {"id": "M2", "type": "machine", "features": [0.0] * 64}
            ],
            "edges": []
        }
    }
    
    # First request
    r1 = client.post("/graph/predict/full", json=payload)
    assert r1.status_code == 200
    assert r1.json()["cache_hit"] == False
    
    # Second request
    r2 = client.post("/graph/predict/full", json=payload)
    assert r2.status_code == 200
    assert r2.json()["cache_hit"] == True
    
    # Third request with different timestamp window
    payload["timestamp"] = timestamp + 1000.0 # Way outside window
    r3 = client.post("/graph/predict/full", json=payload)
    assert r3.status_code == 200
    assert r3.json()["cache_hit"] == False

def test_predict_full_pipeline_missing_modalities():
    # Only sending visual data
    payload = {
        "machine_id": "M3",
        "timestamp": time.time(),
        "sensor_data": None, 
        "visual_data": [[[0.1] * 224 for _ in range(224)] for _ in range(3)], 
        "graph": {
            "nodes": [
                {"id": "M3", "type": "machine", "features": [0.0] * 64}
            ],
            "edges": []
        }
    }
    
    response = client.post("/graph/predict/full", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "anomaly_score" in data
