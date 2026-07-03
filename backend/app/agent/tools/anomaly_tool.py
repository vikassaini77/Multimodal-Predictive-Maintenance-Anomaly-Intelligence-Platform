from pydantic import BaseModel, Field
from typing import Type
import json
from backend.app.agent.tools.base import Tool

class AnomalyScoreSchema(BaseModel):
    machine_id: str = Field(..., description="The ID of the machine to evaluate.")
    timestamp: str = Field(..., description="The specific timestamp to evaluate (ISO 8601).")

class RunAnomalyScoreTool(Tool):
    @property
    def name(self) -> str:
        return "run_anomaly_score"
        
    @property
    def description(self) -> str:
        return "Runs the multimodal deep learning pipeline to calculate an anomaly score (0.0 to 1.0) and identify fault types for a specific machine at a specific time."
        
    @property
    def args_schema(self) -> Type[BaseModel]:
        return AnomalyScoreSchema
        
    def run(self, machine_id: str, timestamp: str) -> str:
        # MOCK IMPLEMENTATION for agent testing
        # In production, this would call `InferencePipeline.predict()`
        
        # We simulate a high anomaly score for the spike seen in the sensor tool mock
        if "10:10:00" in timestamp:
            result = {
                "machine_id": machine_id,
                "timestamp": timestamp,
                "anomaly_score": 0.89,
                "predicted_fault": "bearing_wear",
                "sensor_contribution": 0.65,
                "visual_contribution": 0.35,
                "status": "CRITICAL_ANOMALY"
            }
        else:
            result = {
                "machine_id": machine_id,
                "timestamp": timestamp,
                "anomaly_score": 0.12,
                "predicted_fault": "normal",
                "status": "NORMAL"
            }
            
        return json.dumps(result)

from backend.app.agent.registry import registry, PermissionScope
registry.register(RunAnomalyScoreTool(), PermissionScope.READ_ONLY)
