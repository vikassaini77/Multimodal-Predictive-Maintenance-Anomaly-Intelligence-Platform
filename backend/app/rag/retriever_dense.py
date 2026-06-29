from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.app.rag.embeddings import EmbeddingGenerator

class DenseRetriever:
    def __init__(self, db: Session, embedder: EmbeddingGenerator = None):
        self.db = db
        self.embedder = embedder or EmbeddingGenerator()
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top_k documents using pgvector cosine distance (<=>).
        """
        query_vector = self.embedder.embed_query(query)
        
        # We use raw SQL with SQLAlchemy to leverage pgvector's <=> operator
        sql = text("""
            SELECT id, content, metadata, 1 - (embedding <=> :vector) AS cosine_similarity
            FROM documents
            ORDER BY embedding <=> :vector
            LIMIT :top_k
        """)
        
        # Convert vector to Postgres array format string
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"
        
        results = self.db.execute(sql, {"vector": vector_str, "top_k": top_k}).fetchall()
        
        retrieved_docs = []
        for row in results:
            retrieved_docs.append({
                "id": str(row.id),
                "content": row.content,
                "metadata": row.metadata,
                "score": float(row.cosine_similarity)
            })
            
        return retrieved_docs
