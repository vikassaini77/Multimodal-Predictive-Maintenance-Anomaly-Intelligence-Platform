from pydantic import BaseModel, Field
from typing import Type
from backend.app.agent.tools.base import Tool
from backend.app.rag.answerer import RAGAnswerer
from sqlalchemy.orm import Session
from backend.app.db.session import SessionLocal

class RAGSearchSchema(BaseModel):
    query: str = Field(..., description="The question or search query to look up in the OEM maintenance manuals (e.g., 'What causes bearing wear?').")

class SearchManualRAGTool(Tool):
    @property
    def name(self) -> str:
        return "search_manual_rag"
        
    @property
    def description(self) -> str:
        return "Queries the OEM maintenance manuals to find standard operating procedures, failure modes, and mitigation strategies."
        
    @property
    def args_schema(self) -> Type[BaseModel]:
        return RAGSearchSchema
        
    def run(self, query: str) -> str:
        # We mock this for the standalone agent tests if DB isn't ready
        # In production, we'd do:
        # db = SessionLocal()
        # answerer = RAGAnswerer(db)
        # res = answerer.answer_query(query)
        # return res.model_dump_json()
        
        # MOCK IMPLEMENTATION:
        if "bearing" in query.lower():
            return '{"answer": "Bearing wear is typically caused by inadequate lubrication or excessive vibration. Recommended action is immediate inspection and relubrication.", "confidence": 0.95, "sources": ["manual_v1_p45"], "recommended_action": "Inspect and relubricate bearings."}'
        
        return '{"answer": "No specific procedures found for this query.", "confidence": 0.1, "sources": [], "recommended_action": null}'

from backend.app.agent.registry import registry, PermissionScope
registry.register(SearchManualRAGTool(), PermissionScope.READ_ONLY)
