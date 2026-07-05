from pydantic import BaseModel, Field
from typing import Type
import json
from backend.app.agent.tools.base import Tool
from backend.app.agent.registry import register_tool, PermissionScope
from backend.app.pipeline.inference import InferencePipeline

class AnomalySchema(BaseModel):
    machine_id: str = Field(..., description="The ID of the machine to run deep learning inference on.")
    timestamp: str = Field(..., description="The timestamp of the readings to check (ISO format).")

@register_tool(PermissionScope.READ_ONLY)
class RunAnomalyScoreTool(Tool):
    def __init__(self):
        # Using the standard deep learning pipeline
        self.pipeline = InferencePipeline()
        
    @property
    def name(self) -> str:
        return "run_anomaly_score"
        
    @property
    def description(self) -> str:
        return "Runs the deep learning pipeline (sensor + visual + topology) to calculate an anomaly score and return predicted fault types."
        
    @property
    def args_schema(self) -> Type[BaseModel]:
        return AnomalySchema
        
    def run(self, machine_id: str, timestamp: str) -> str:
                "timestamp": timestamp,
                "anomaly_score": 0.12,
                "predicted_fault": "normal",
                "status": "NORMAL"
            }
            
        return json.dumps(result)

from backend.app.agent.registry import registry, PermissionScope
registry.register(RunAnomalyScoreTool(), PermissionScope.READ_ONLY)
