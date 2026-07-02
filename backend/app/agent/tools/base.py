from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Type, Any, Dict

class Tool(ABC):
    """
    Abstract interface for all Agent Tools.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool to be passed to the LLM."""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed description of when and how the LLM should use this tool."""
        pass
        
    @property
    @abstractmethod
    def args_schema(self) -> Type[BaseModel]:
        """Pydantic schema defining the arguments this tool expects."""
        pass
        
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool with the provided arguments and return the observation."""
        pass
        
    def to_gemini_function(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation suitable for Gemini's function calling API.
        We will rely on google-genai's native integration where possible, 
        but this provides a schema fallback.
        """
        schema = self.args_schema.model_json_schema()
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }
