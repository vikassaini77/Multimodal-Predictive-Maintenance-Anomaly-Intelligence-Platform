from pydantic import BaseModel, Field
from typing import List, Optional

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

class FullPredictResponse(BaseModel):
    machine_id: str
    timestamp: float
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    cache_hit: bool
    explanations: Optional[List[NodeExplanation]] = None
