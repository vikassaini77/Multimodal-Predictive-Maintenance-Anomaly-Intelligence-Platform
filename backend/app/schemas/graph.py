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
