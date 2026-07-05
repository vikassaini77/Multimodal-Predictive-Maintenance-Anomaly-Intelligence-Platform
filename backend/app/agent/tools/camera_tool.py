from pydantic import BaseModel, Field
from typing import Type
import json
from backend.app.agent.tools.base import Tool
from backend.app.agent.registry import register_tool, PermissionScope

class CameraSchema(BaseModel):
    machine_id: str = Field(..., description="The ID of the machine to fetch a snapshot for.")

@register_tool(PermissionScope.ACTION)
class FetchCameraSnapshotTool(Tool):
    @property
    def name(self) -> str:
        return "fetch_camera_snapshot"
        
    @property
    def description(self) -> str:
        return "Fetches the latest visual snapshot from a machine's camera to check for visible defects (like a conveyor jam)."
        
    @property
    def args_schema(self) -> Type[BaseModel]:
        return CameraSchema
        
    def run(self, machine_id: str) -> str:
        # Mock visual response for the agent to parse
        return json.dumps({
            "machine_id": machine_id,
            "timestamp": "2023-10-27T10:15:00Z",
            "camera_status": "online",
            "visual_analysis": "Detected object obstructing conveyor belt path. Large metallic debris visible near primary roller."
        })
