import pytest
from fastapi.testclient import TestClient
import time
import json
import redis
from backend.app.main import app
from backend.app.celery_app import celery_app

celery_app.conf.update(
    task_always_eager=True, 
    task_store_eager_result=True,
    task_eager_propagates=False
)

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
    assert "job_id" in data
    job_id = data["job_id"]
    
    # Poll job status
    status_response = client.get(f"/graph/jobs/{job_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    
    assert status_data["status"] == "success"
    assert status_data["result"] is not None
    assert status_data["result"]["machine_id"] == "M1"
    assert "anomaly_score" in status_data["result"]
    assert "is_anomaly" in status_data["result"]
    assert "threshold" in status_data["result"]
    assert status_data["result"]["cache_hit"] == False # First time should be false

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
    job_id1 = r1.json()["job_id"]
    res1 = client.get(f"/graph/jobs/{job_id1}").json()
    assert res1["result"]["cache_hit"] == False
    
    # Second request
    r2 = client.post("/graph/predict/full", json=payload)
    assert r2.status_code == 200
    job_id2 = r2.json()["job_id"]
    res2 = client.get(f"/graph/jobs/{job_id2}").json()
    assert res2["result"]["cache_hit"] == True
    
    # Third request with different timestamp window
    payload["timestamp"] = timestamp + 1000.0 # Way outside window
    r3 = client.post("/graph/predict/full", json=payload)
    assert r3.status_code == 200
    job_id3 = r3.json()["job_id"]
    res3 = client.get(f"/graph/jobs/{job_id3}").json()
    assert res3["result"]["cache_hit"] == False

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
    job_id = response.json()["job_id"]
    
    status_response = client.get(f"/graph/jobs/{job_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] == "success"
    assert "anomaly_score" in status_data["result"]

def test_predict_full_pipeline_failure_dlq(monkeypatch):
    # Empty redis dlq list first
    r = redis.Redis.from_url(celery_app.conf.broker_url)
    r.delete("dlq")

    # Force a failure in the pipeline
    def mock_predict(*args, **kwargs):
        raise Exception("Simulated worker crash")

    from backend.app.api.dependencies import get_model_container
    models = get_model_container()
    monkeypatch.setattr(models.pipeline, "predict", mock_predict)

    payload = {
        "machine_id": "M_FAIL",
        "timestamp": time.time(),
        "graph": {
            "nodes": [{"id": "M_FAIL", "type": "machine", "features": [0.0] * 64}],
            "edges": []
        }
    }

    # First request
    response = client.post("/graph/predict/full", json=payload)
    assert response.status_code == 200

    # The celery task should have failed (and retried 3 times in eager mode, then run on_failure)
    # Check DLQ endpoint
    dlq_response = client.get("/graph/jobs/failed")
    assert dlq_response.status_code == 200
    dlq_data = dlq_response.json()
    assert dlq_data["status"] == "success"
    
    # We should have exactly 1 failed job in DLQ since it exhausts retries
    assert len(dlq_data["failed_jobs"]) >= 1
    failed_job = dlq_data["failed_jobs"][0]
    assert "Simulated worker crash" in failed_job["exc"]
    assert failed_job["args"][0]["machine_id"] == "M_FAIL"
