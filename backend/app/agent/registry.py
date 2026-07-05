import importlib
import pkgutil
from enum import Enum
from typing import List, Type, Callable, Dict
from backend.app.agent.tools.base import Tool

class PermissionScope(Enum):
    READ_ONLY = "read_only"
    ACTION = "action"
    ADMIN = "admin"

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._scopes: Dict[str, PermissionScope] = {}

    def register(self, tool: Tool, scope: PermissionScope):
        self._tools[tool.name] = tool
        self._scopes[tool.name] = scope

    def get_allowed_tools(self, agent_scope: PermissionScope) -> List[Tool]:
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

def register_tool(scope: PermissionScope) -> Callable:
    """Decorator to register a tool class in the global registry."""
    def decorator(cls: Type[Tool]):
        # Instantiate the tool and register it
        instance = cls()
        registry.register(instance, scope)
        return cls
    return decorator

def auto_discover_tools(package_name: str = "backend.app.agent.tools"):
    """
    Dynamically imports all modules in the given package so that 
    their @register_tool decorators are executed.
    """
    package = importlib.import_module(package_name)
    if not hasattr(package, "__path__"):
        return
        
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        full_module_name = f"{package_name}.{module_name}"
        if full_module_name != "backend.app.agent.tools.base":
            importlib.import_module(full_module_name)
