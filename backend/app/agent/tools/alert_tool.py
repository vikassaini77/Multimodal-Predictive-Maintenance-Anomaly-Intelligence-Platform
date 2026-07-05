from pydantic import BaseModel, Field
from typing import Type
import json
from backend.app.agent.tools.base import Tool
from backend.app.agent.registry import register_tool, PermissionScope
from backend.app.agent.guardrails import RateLimiter, ActionGuard

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
        
    def run(self, machine_id: str, severity: str, message: str, human_confirmed: bool = False) -> str:
        # 1. Guardrail: Action Confirmation for Critical
        if not ActionGuard.require_human_confirmation(severity, human_confirmed):
            return json.dumps({
                "status": "blocked", 
                "error": "Guardrail triggered: Critical alerts require explicit human confirmation. Please ask the operator for approval before proceeding."
            })
            
        # 2. Guardrail: Rate Limiting
        if not self.rate_limiter.check_and_record(machine_id):
            return json.dumps({
                "status": "blocked",
                "error": "Guardrail triggered: Rate limit exceeded. Only 1 alert per machine is allowed every 10 minutes to prevent spam."
            })
            
        # 3. Execute Action (Mock)
        return json.dumps({
            "status": "success",
            "message": f"Alert successfully dispatched to maintenance team for {machine_id} (Severity: {severity})."
        })
