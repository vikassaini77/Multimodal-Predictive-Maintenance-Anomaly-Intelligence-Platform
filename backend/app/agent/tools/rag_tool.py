from pydantic import BaseModel, Field
from typing import Type
import json
from backend.app.agent.tools.base import Tool
from backend.app.agent.registry import register_tool, PermissionScope
from backend.app.rag.pipeline import RAGPipeline

class RAGSchema(BaseModel):
    query: str = Field(..., description="The natural language query to search the maintenance manuals for.")
    machine_id: str = Field(None, description="Optional. The specific machine ID to filter the manuals by.")

@register_tool(PermissionScope.READ_ONLY)
class SearchManualRAGTool(Tool):
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        
    @property
    def name(self) -> str:
        return "search_maintenance_manuals"
        
    @property
    def description(self) -> str:
        return "Searches internal maintenance manuals, OEM guides, and historical repair logs to find troubleshooting steps or specs."
        
    @property
    def args_schema(self) -> Type[BaseModel]:
        return RAGSchema
        
    def run(self, query: str, machine_id: str = None) -> str:
        # We call our real RAG pipeline from Week 2!
        # In a test environment without a loaded DB, this might return empty results,
        # but the agent loop will handle parsing the answer.
        try:
            answer, context_docs = self.rag_pipeline.query(query, top_k=3)
            return json.dumps({
                "status": "success",
                "rag_answer": answer,
                "sources_retrieved": len(context_docs)
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to query manuals: {str(e)}"
            })
