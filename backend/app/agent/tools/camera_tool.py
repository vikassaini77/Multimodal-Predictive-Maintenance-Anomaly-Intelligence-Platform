from pydantic import BaseModel, Field
from typing import Type
import json
from backend.app.agent.tools.base import Tool
from backend.app.agent.registry import registry, PermissionScope

class FetchImageSchema(BaseModel):
    machine_id: str = Field(..., description="The ID of the machine to fetch the image snapshot for.")

class FetchImageSnapshotTool(Tool):
    @property
    def name(self) -> str:
        return "fetch_image_snapshot"
        
    @property
    def description(self) -> str:
        return "Fetches the latest visual frame (camera snapshot) metadata for a given machine to assess physical conditions."
        
    @property
    def args_schema(self) -> Type[BaseModel]:
        return FetchImageSchema
        
    def run(self, machine_id: str) -> str:
        # Mock Implementation returning visual metadata
        return json.dumps({
            "machine_id": machine_id,
            "timestamp": "2023-10-27T10:10:05Z",
            "visual_assessment": "Visible debris on the outer bearing casing. No smoke detected.",
            "status": "success"
        })

# Register tool as READ_ONLY
registry.register(FetchImageSnapshotTool(), PermissionScope.READ_ONLY)
