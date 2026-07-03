from enum import Enum
from typing import List, Type
from backend.app.agent.tools.base import Tool

class PermissionScope(Enum):
    READ_ONLY = "read_only"
    ACTION = "action"
    ADMIN = "admin"

class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self._scopes = {}

    def register(self, tool: Tool, scope: PermissionScope):
        self._tools[tool.name] = tool
        self._scopes[tool.name] = scope

    def get_allowed_tools(self, agent_scope: PermissionScope) -> List[Tool]:
        """
        Returns a list of tools the agent is permitted to use based on its scope.
        READ_ONLY can only use READ_ONLY tools.
        ACTION can use READ_ONLY and ACTION tools.
        ADMIN can use everything.
        """
        allowed = []
        for name, tool in self._tools.items():
            tool_scope = self._scopes[name]
            
            if agent_scope == PermissionScope.ADMIN:
                allowed.append(tool)
            elif agent_scope == PermissionScope.ACTION and tool_scope in [PermissionScope.READ_ONLY, PermissionScope.ACTION]:
                allowed.append(tool)
            elif agent_scope == PermissionScope.READ_ONLY and tool_scope == PermissionScope.READ_ONLY:
                allowed.append(tool)
                
        return allowed

# Global Registry Instance
registry = ToolRegistry()
