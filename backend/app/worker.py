import networkx as nx
import torch
import json
import redis
import time
from celery import Task
from backend.app.celery_app import celery_app
from backend.app.api.dependencies import get_model_container
from backend.app.data.factory_graph import convert_to_pyg_heterodata

redis_client = redis.Redis.from_url(celery_app.conf.broker_url)

class DLQTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        # Push job details to the dead letter queue (dlq)
        job_info = {
            "task_id": task_id,
            "failed_at": time.time(),
            "args": args,
            "kwargs": kwargs,
            "exc": str(exc),
            "traceback": str(einfo.traceback) if einfo else None
        }
        job_json = json.dumps(job_info)
        redis_client.lpush("dlq", job_json)
        
        # Publish alert to WebSocket listeners
        alert = {
            "type": "dlq_alert",
            "task_id": task_id,
            "error": str(exc)
        }
        redis_client.publish("dlq_alerts", json.dumps(alert))
        
        super().on_failure(exc, task_id, args, kwargs, einfo)

@celery_app.task(
    name="predict_full_pipeline_task", 
    bind=True, 
    base=DLQTask,
    autoretry_for=(Exception,), 
    max_retries=3, 
    retry_backoff=True
)
def predict_full_pipeline_task(self, payload_dict: dict):
    # Retrieve the singleton model container
    models = get_model_container()
    
    # 1. Parse Graph
    nx_graph = nx.DiGraph()
    graph_data = payload_dict.get("graph", {})
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    for node in nodes:
        nx_graph.add_node(node["id"], type=node["type"])
    for edge in edges:
        source_type = nx_graph.nodes[edge["source"]]["type"]
        target_type = nx_graph.nodes[edge["target"]]["type"]
        edge_tuple = (source_type, edge["type"], target_type)
        nx_graph.add_edge(edge["source"], edge["target"], type=edge_tuple)
        
    hetero_graph = convert_to_pyg_heterodata(nx_graph)
    
    # 2. Parse Raw Data
    sensor_tensor = None
    if payload_dict.get("sensor_data"):
        sensor_tensor = torch.tensor(payload_dict["sensor_data"], dtype=torch.float32)
        
    visual_tensor = None
    if payload_dict.get("visual_data"):
        visual_tensor = torch.tensor(payload_dict["visual_data"], dtype=torch.float32)
        
    # 3. Run Pipeline
    result = models.pipeline.predict(
        machine_id=payload_dict["machine_id"],
        timestamp=payload_dict["timestamp"],
        sensor_data=sensor_tensor,
        visual_data=visual_tensor,
        hetero_graph=hetero_graph
    )
    
    # Return serializable dict
    return {
        "machine_id": result["machine_id"],
        "timestamp": result["timestamp"],
        "anomaly_score": result["anomaly_score"],
        "is_anomaly": result["is_anomaly"],
        "threshold": result["threshold"],
        "cache_hit": result["cache_hit"]
    }
