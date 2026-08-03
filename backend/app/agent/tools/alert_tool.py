from pydantic import BaseModel, Field
from typing import Type
import json
from datetime import datetime
import uuid

from backend.app.agent.tools.base import Tool
from backend.app.agent.registry import register_tool, PermissionScope
from backend.app.agent.guardrails import RateLimiter, ActionGuard, redis_client
from backend.app.db.session import SessionLocal
from backend.app.db.models import MaintenanceWindow
from backend.app.core.email import send_critical_alert_email

class AlertSchema(BaseModel):
    machine_id: str = Field(..., description="The ID of the machine to alert on.")
    severity: str = Field(..., description="Severity level: 'info', 'warning', or 'critical'.")
    message: str = Field(..., description="The diagnosis and recommended action.")
    human_confirmed: bool = Field(False, description="Set to True if a human operator has explicitly authorized this critical alert.")

@register_tool(PermissionScope.ACTION)
class DispatchAlertTool(Tool):
    def __init__(self):
        self.rate_limiter = RateLimiter(window_seconds=600)
        
    @property
    def name(self) -> str:
        return "dispatch_alert"
        
    @property
    def description(self) -> str:
        return "Dispatches an email/webhook alert to the maintenance team. If severity is 'critical', 'human_confirmed' must be True."
        
    @property
    def args_schema(self) -> Type[BaseModel]:
        return AlertSchema

    def _get_zone(self, machine_id: str) -> str:
        if machine_id.startswith("CNC"):
            return "Milling Zone"
        elif machine_id.startswith("STAMP"):
            return "Stamping Zone"
        elif machine_id.startswith("WELD"):
            return "Welding Zone"
        return "General Zone"
        
    def run(self, machine_id: str, severity: str, message: str, human_confirmed: bool = False) -> str:
        # 1. Guardrail: Action Confirmation for Critical
        if not ActionGuard.require_human_confirmation(severity, human_confirmed):
            return json.dumps({
                "status": "blocked", 
                "error": "Guardrail triggered: Critical alerts require explicit human confirmation. Please ask the operator for approval before proceeding."
            })
            
        # 2. Check Maintenance Window
        zone = self._get_zone(machine_id)
        db = SessionLocal()
        now = datetime.now()
        current_day = now.weekday()
        current_time = now.time()
        
        try:
            # Check if any window is active for this zone
            active_windows = db.query(MaintenanceWindow).filter(
                MaintenanceWindow.zone == zone,
                MaintenanceWindow.day_of_week == current_day,
                MaintenanceWindow.start_time <= current_time,
                MaintenanceWindow.end_time >= current_time
            ).all()
            if active_windows:
                return json.dumps({
                    "status": "suppressed",
                    "message": f"Alert suppressed because {zone} is currently in a scheduled maintenance window."
                })
        finally:
            db.close()
            
        # 3. Deduplication (10 minute TTL)
        count = self.rate_limiter.check_and_record(machine_id, fault_type=message)
        
        if count == 1:
            # First occurrence
            if severity.lower() == "critical":
                send_critical_alert_email(machine_id, message, score=1.0)
                
            if redis_client:
                alert_payload = {
                    "type": "alert",
                    "id": str(uuid.uuid4()),
                    "machineId": machine_id,
                    "zone": zone,
                    "score": 1.0 if severity.lower() == "critical" else (0.6 if severity.lower() == "warning" else 0.1),
                    "message": message,
                    "duplicateCount": 1
                }
                redis_client.publish("edge_alerts", json.dumps(alert_payload))
                
            return json.dumps({
                "status": "success",
                "message": f"Alert dispatched successfully to maintenance team for {machine_id}."
            })
        else:
            # Duplicated occurrence
            if redis_client:
                increment_payload = {
                    "type": "alert_increment",
                    "machineId": machine_id,
                    "message": message,
                    "duplicateCount": count
                }
                redis_client.publish("edge_alerts", json.dumps(increment_payload))
            
            return json.dumps({
                "status": "deduplicated",
                "message": f"Alert deduplicated (seen {count} times). Counter incremented in UI."
            })
