from pydantic import BaseModel, Field
from typing import List, Optional

class RAGAnswer(BaseModel):
    answer: str = Field(..., description="The synthesized answer to the user query.")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0 based on the available context.")
    sources: List[str] = Field(..., description="List of chunk IDs that were used to formulate the answer. Empty if no relevant context.")
    recommended_action: Optional[str] = Field(None, description="Any actionable recommendation derived from the maintenance manuals.")
