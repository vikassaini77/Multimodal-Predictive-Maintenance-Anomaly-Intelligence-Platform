import pytest
from unittest.mock import patch
from backend.app.agent.registry import ToolRegistry, PermissionScope
from backend.app.agent.tools.base import Tool
from pydantic import BaseModel
from backend.app.agent.tools.alert_tool import DispatchAlertTool
from backend.app.agent.tools.camera_tool import FetchImageSnapshotTool

class DummyActionTool(Tool):
    @property
    def name(self): return "dummy_action"
    @property
    def description(self): return "action"
    @property
    def args_schema(self): return BaseModel
    def run(self): return "action done"

def test_tool_registry_scopes():
    registry = ToolRegistry()
    registry.register(FetchImageSnapshotTool(), PermissionScope.READ_ONLY)
    registry.register(DummyActionTool(), PermissionScope.ACTION)
    
    # Read Only Agent
    ro_tools = registry.get_allowed_tools(PermissionScope.READ_ONLY)
    assert len(ro_tools) == 1
    assert ro_tools[0].name == "fetch_image_snapshot"
    
    # Action Agent
    action_tools = registry.get_allowed_tools(PermissionScope.ACTION)
    assert len(action_tools) == 2
    
def test_rate_limiter():
    tool = DispatchAlertTool()
    
    # First alert should succeed
    res1 = tool.run("machine_99", "warning", "test")
    assert "success" in res1
    
    # Second alert immediately should fail
    res2 = tool.run("machine_99", "warning", "test 2")
    assert "blocked" in res2
    assert "Rate limit exceeded" in res2

def test_action_guard_critical():
    tool = DispatchAlertTool()
    # Provide a new machine ID so rate limit doesn't block it
    res_blocked = tool.run("machine_100", "critical", "test")
    assert "blocked" in res_blocked
    assert "explicit human confirmation" in res_blocked
    
    res_success = tool.run("machine_100", "critical", "test", human_confirmed=True)
    assert "success" in res_success
