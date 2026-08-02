from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import time
import math

class NodeInput(BaseModel):
    id: str
    type: str = Field(..., description="Type of the node: 'machine', 'conveyor', or 'sensor'")
    features: List[float] = Field(..., description="Node feature embedding (e.g., 256-dimensional vector)")

class EdgeInput(BaseModel):
    source: str
    target: str
    type: str = Field(..., description="Edge relation type: e.g., 'feeds_into', 'monitors'")

class GraphInput(BaseModel):
    nodes: List[NodeInput]
    edges: List[EdgeInput]

class NodeExplanation(BaseModel):
    node_id: str
    node_type: str
    importance_score: float

class RiskPrediction(BaseModel):
    node_id: str
    node_type: str
    fault_probability: float
    is_fault: bool
    top_contributing_neighbors: Optional[List[NodeExplanation]] = None

class FullPredictRequest(BaseModel):
    machine_id: str
    timestamp: float
    sensor_data: Optional[List[List[float]]] = Field(None, description="Raw sensor data sequence [channels, seq_len]")
    visual_data: Optional[List[List[List[float]]]] = Field(None, description="Raw visual data [channels, H, W]")
    graph: GraphInput = Field(..., description="Local neighborhood graph topology")

    @field_validator("timestamp")
    def validate_timestamp(cls, v):
        current_time = time.time()
        # Allow a small buffer (e.g., 5 seconds) for slight clock drifts
        if v > current_time + 5.0:
            raise ValueError(f"Timestamp {v} is in the future (current time: {current_time})")
        return v

    @field_validator("sensor_data")
    def validate_sensor_data(cls, v):
        if v is None:
            return v
        if not v:
            raise ValueError("Sensor data cannot be empty if provided")
            
        for i, channel in enumerate(v):
            if not channel:
                raise ValueError(f"Sensor channel {i} is empty")
            for j, val in enumerate(channel):
                if math.isnan(val):
                    raise ValueError(f"NaN value detected at channel {i}, step {j}")
                # Assuming reasonable normalized ranges, or physical bounds
                if val < -10000.0 or val > 10000.0:
                    raise ValueError(f"Value {val} out of range at channel {i}, step {j}")
        return v

class FullPredictResponse(BaseModel):
    machine_id: str
    timestamp: float
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    cache_hit: bool
    explanations: Optional[List[NodeExplanation]] = None
    lower_confidence: bool = False
    latency_warning: bool = False

class AsyncPredictResponse(BaseModel):
    job_id: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[FullPredictResponse] = None
