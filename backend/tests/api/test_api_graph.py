import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_graph_faults():
    payload = {
        "nodes": [
            {"id": "m1", "type": "machine", "features": [0.1] * 256},
            {"id": "m2", "type": "machine", "features": [0.5] * 256},
            {"id": "c1", "type": "conveyor", "features": [0.2] * 256},
            {"id": "s1", "type": "sensor", "features": [0.3] * 256}
        ],
        "edges": [
            {"source": "m1", "target": "c1", "type": "feeds_into"},
            {"source": "c1", "target": "m2", "type": "feeds_into"},
            {"source": "s1", "target": "m1", "type": "monitors"}
        ]
    }
    
    response = client.post("/graph/predict", json=payload)
    if response.status_code != 200:
        print(response.json())
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["node_id"] == "m1"
    assert data[0]["node_type"] == "machine"
    assert "fault_probability" in data[0]
