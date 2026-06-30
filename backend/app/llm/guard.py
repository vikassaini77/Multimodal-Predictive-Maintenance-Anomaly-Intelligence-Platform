from typing import List, Dict, Any
from backend.app.llm.schemas import RAGAnswer

class HallucinationGuard:
    @staticmethod
    def verify_citations(answer: RAGAnswer, retrieved_chunks: List[Dict[str, Any]]) -> RAGAnswer:
        """
        Post-hoc check to reject answers citing chunks that weren't retrieved.
        """
        valid_ids = {str(chunk.get("id")) for chunk in retrieved_chunks}
        
        # Check if any cited source is not in the valid IDs
        invalid_sources = [source for source in answer.sources if str(source) not in valid_ids]
        
        if invalid_sources:
            # Modify the answer to strip invalid sources and lower confidence
            answer.sources = [source for source in answer.sources if str(source) in valid_ids]
            answer.confidence = max(0.0, answer.confidence - 0.5)
            # Alternatively, we could raise an exception to reject it completely, 
            # but graceful degradation is usually better.
            
        return answer
