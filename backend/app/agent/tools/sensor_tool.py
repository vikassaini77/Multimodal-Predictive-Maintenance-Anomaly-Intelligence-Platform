from pydantic import BaseModel, Field
from typing import Dict, Any, Type
import json
from backend.app.agent.tools.base import Tool

class QuerySensorSchema(BaseModel):
    machine_id: str = Field(..., description="The ID of the machine to query (e.g., 'machine_001').")
    limit: int = Field(5, description="Number of recent readings to fetch.")

class QuerySensorDBTool(Tool):
    @property
    def name(self) -> str:
        return "query_sensor_db"
        
    @property
    def description(self) -> str:
        return "Fetches recent raw sensor readings (temperature, vibration, acoustic) from the database for a given machine."
        
    @property
    def args_schema(self) -> Type[BaseModel]:
        return QuerySensorSchema
        
    def run(self, machine_id: str, limit: int = 5) -> str:
        # MOCK IMPLEMENTATION for agent testing
        # In production, this would query TimescaleDB/Postgres.
        mock_data = [
            {"timestamp": "2023-10-27T10:00:00Z", "temperature": 85.5, "vibration": 1.2, "acoustic": 45.0},
            {"timestamp": "2023-10-27T10:05:00Z", "temperature": 86.2, "vibration": 1.5, "acoustic": 47.1},
            {"timestamp": "2023-10-27T10:10:00Z", "temperature": 92.1, "vibration": 3.8, "acoustic": 55.4} # Spike
        ]
        
        return json.dumps({
            "machine_id": machine_id,
            "status": "success",
            "recent_readings": mock_data[:limit]
        })
