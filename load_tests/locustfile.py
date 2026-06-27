import time
import random
from locust import HttpUser, task, between

class PipelineUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict_full_pipeline(self):
        machine_id = f"M{random.randint(1, 100)}"
        
        payload = {
            "machine_id": machine_id,
            "timestamp": time.time(),
            "sensor_data": [[0.5] * 10 for _ in range(14)], # 14 channels, 10 sequence steps
            "visual_data": [[[0.1] * 224 for _ in range(224)] for _ in range(3)], # 3x224x224
            "graph": {
                "nodes": [
                    {"id": machine_id, "type": "machine", "features": [0.0] * 64}
                ],
                "edges": []
            }
        }
        
        self.client.post("/graph/predict/full", json=payload)
