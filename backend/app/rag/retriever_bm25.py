import string
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from rank_bm25 import BM25Okapi
from backend.app.db.models import Document

class BM25Retriever:
    def __init__(self, db: Session):
        self.db = db
        self.corpus = []
        self.bm25 = None
        self._build_index()
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer tuned for technical/industrial terms.
        Lowercases, removes punctuation, and splits by whitespace.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def _build_index(self):
        """
        Fetches all documents from Postgres and builds the in-memory BM25 index.
        """
        docs = self.db.query(Document).all()
        self.corpus = [
            {
                "id": str(doc.id),
                "content": doc.content,
                "metadata": doc.metadata_
            }
            for doc in docs
        ]
        
        tokenized_corpus = [self._tokenize(doc["content"]) for doc in self.corpus]
        
        # If corpus is empty, bm25 will raise an error on query, so handle gracefully
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top_k documents using BM25.
        """
        if not self.bm25 or not self.corpus:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair documents with their scores and sort descending
        doc_scores = [(self.corpus[i], score) for i, score in enumerate(scores)]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        retrieved_docs = []
        for doc, score in doc_scores[:top_k]:
            retrieved_docs.append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": float(score)
            })
            
        return retrieved_docs
