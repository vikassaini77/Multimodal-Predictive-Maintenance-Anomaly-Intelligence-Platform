from typing import List, Dict, Any
from sqlalchemy.orm import Session
from backend.app.rag.retriever_dense import DenseRetriever
from backend.app.rag.retriever_bm25 import BM25Retriever

class QueryExpander:
    def __init__(self):
        # A simple synonym map for industrial terms
        self.synonyms = {
            "bearing failure": ["bearing wear", "bearing degradation", "bearing damage"],
            "motor overheat": ["high temperature", "thermal fault", "overheating"],
            "vibration": ["oscillation", "shaking", "resonance"],
            "pump cavitation": ["fluid cavitation", "pressure drop", "intake blockage"]
        }
        
    def expand(self, query: str) -> str:
        expanded_terms = [query]
        query_lower = query.lower()
        for key, syns in self.synonyms.items():
            if key in query_lower:
                expanded_terms.extend(syns)
        # Join unique terms
        return " ".join(list(dict.fromkeys(expanded_terms)))

class HybridRetriever:
    def __init__(self, db: Session):
        self.dense = DenseRetriever(db)
        self.bm25 = BM25Retriever(db)
        self.expander = QueryExpander()
        
    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves candidates using both BM25 and Dense search, 
        and fuses them using Reciprocal Rank Fusion (RRF).
        """
        expanded_query = self.expander.expand(query)
        
        # Get candidates from both retrievers
        dense_results = self.dense.retrieve(expanded_query, top_k=top_k)
        bm25_results = self.bm25.retrieve(expanded_query, top_k=top_k)
        
        return self._rrf_fusion(dense_results, bm25_results, top_k)
        
    def _rrf_fusion(self, dense_results: List[Dict], bm25_results: List[Dict], top_k: int, k_param: int = 60) -> List[Dict]:
        """
        Implements Reciprocal Rank Fusion: score = 1 / (k_param + rank)
        """
        scores = {}
        docs = {}
        
        # Process Dense results
        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            docs[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k_param + rank + 1)
            
        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            if doc_id not in docs:
                docs[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k_param + rank + 1)
            
        # Sort by fused score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_docs = []
        for doc_id, score in sorted_results[:top_k]:
            doc_copy = dict(docs[doc_id])
            doc_copy["score"] = score
            fused_docs.append(doc_copy)
            
        return fused_docs
