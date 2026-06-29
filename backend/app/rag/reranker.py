from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate documents using a Cross-Encoder.
        """
        if not documents:
            return []
            
        # CrossEncoder expects pairs of (query, document_text)
        pairs = [(query, doc["content"]) for doc in documents]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Attach new scores and sort
        doc_scores = []
        for i, doc in enumerate(documents):
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = float(scores[i])
            doc_scores.append(doc_copy)
            
        doc_scores.sort(key=lambda x: x["rerank_score"], reverse=True)
        return doc_scores[:top_k]
