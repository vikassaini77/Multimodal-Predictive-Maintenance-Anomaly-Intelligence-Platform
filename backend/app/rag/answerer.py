import json
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from backend.app.rag.retriever_hybrid import HybridRetriever
from backend.app.rag.reranker import CrossEncoderReranker
from backend.app.llm.client import GeminiClient
from backend.app.llm.schemas import RAGAnswer
from backend.app.llm.prompts import build_rag_prompt
from backend.app.llm.guard import HallucinationGuard

class RAGAnswerer:
    def __init__(self, db: Session):
        self.retriever = HybridRetriever(db)
        self.reranker = CrossEncoderReranker()
        self.llm = GeminiClient()
        self.guard = HallucinationGuard()
        
    def answer_query(self, query: str) -> RAGAnswer:
        """
        End-to-end RAG pipeline: retrieve -> rerank -> prompt -> generate -> guard
        """
        # 1. Retrieve candidates
        candidates = self.retriever.retrieve(query, top_k=20)
        
        # 2. Rerank top candidates
        top_chunks = self.reranker.rerank(query, candidates, top_k=5)
        
        # 3. Build Prompt
        prompt = build_rag_prompt(query, top_chunks)
        
        # 4. Generate Structured Answer via Gemini
        raw_json_str = self.llm.generate(prompt=prompt, response_schema=RAGAnswer)
        
        # 5. Parse and Validate
        try:
            # We strip markdown JSON blocks if present
            clean_str = raw_json_str.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
                
            parsed_dict = json.loads(clean_str.strip())
            answer_obj = RAGAnswer(**parsed_dict)
            
        except Exception as e:
            # Fallback if generation fails validation
            answer_obj = RAGAnswer(
                answer=f"Failed to generate a structured response: {str(e)}",
                confidence=0.0,
                sources=[]
            )
            
        # 6. Guard against hallucinations
        guarded_answer = self.guard.verify_citations(answer_obj, top_chunks)
        
        return guarded_answer
