from pydantic import BaseModel, Field
from typing import Type
import json
from backend.app.agent.tools.base import Tool
from backend.app.agent.registry import register_tool, PermissionScope

class SensorSchema(BaseModel):
    machine_id: str = Field(..., description="The ID of the machine to query (e.g., 'machine_001').")
    limit: int = Field(5, description="Number of recent readings to fetch.")

@register_tool(PermissionScope.READ_ONLY)
class QuerySensorDBTool(Tool):
    @property
    def name(self) -> str:
        return "query_sensor_db"
        
    @property
    def description(self) -> str:
        return "Fetches the most recent raw sensor readings (vibration, temperature, pressure) for a specific machine."
        
    @property
    def args_schema(self) -> Type[BaseModel]:
        return SensorSchema
        
    def run(self, machine_id: str, limit: int = 5) -> str:
        # Mock Implementation for the Agent Demo
        # In a real app, this would query TimescaleDB/Postgres
        
        # We simulate a problematic reading for machine_001 to trigger anomaly detection
        if machine_id == "machine_001":
            data = [
                {"timestamp": "2023-10-27T10:00:00Z", "vibration_x": 0.05, "temperature": 45.2},
                {"timestamp": "2023-10-27T10:05:00Z", "vibration_x": 0.06, "temperature": 45.5},
                {"timestamp": "2023-10-27T10:10:00Z", "vibration_x": 1.25, "temperature": 58.9}, # Spike!
            ]
        else:
            data = [
                {"timestamp": "2023-10-27T10:00:00Z", "vibration_x": 0.01, "temperature": 40.0},
                {"timestamp": "2023-10-27T10:05:00Z", "vibration_x": 0.02, "temperature": 40.1},
            ]
            
        return json.dumps({"status": "success", "machine_id": machine_id, "recent_readings": data[-limit:]})
