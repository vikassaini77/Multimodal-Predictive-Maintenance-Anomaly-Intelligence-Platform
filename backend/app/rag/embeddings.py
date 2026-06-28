from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts.
        Returns a list of 384-dimensional float arrays.
        """
        # encode returns a numpy array, we convert to list of floats for pgvector
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
        
    def embed_query(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]
