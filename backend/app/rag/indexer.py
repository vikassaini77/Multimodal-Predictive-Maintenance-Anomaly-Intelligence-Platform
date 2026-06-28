import hashlib
from typing import List, Dict
from sqlalchemy.orm import Session
from backend.app.db.models import Document
from backend.app.rag.chunker import TextChunker
from backend.app.rag.embeddings import EmbeddingGenerator
from backend.app.utils.logger import logger

class DocumentIndexer:
    def __init__(self, db: Session):
        self.db = db
        self.chunker = TextChunker(chunk_size=512, overlap=50)
        self.embedder = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
        
    def _generate_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def index_documents(self, documents: List[Dict[str, str]]):
        """
        Chunks, embeds, and inserts documents into the database.
        Uses content hash to prevent duplicate insertions.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            
        if not all_chunks:
            return
            
        logger.info(f"Generated {len(all_chunks)} chunks from {len(documents)} documents. Generating embeddings...")
        
        texts = [chunk["content"] for chunk in all_chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        records_to_insert = []
        for chunk, emb in zip(all_chunks, embeddings):
            content_hash = self._generate_hash(chunk["content"])
            
            # Check if exists
            exists = self.db.query(Document).filter(Document.content_hash == content_hash).first()
            if not exists:
                doc_record = Document(
                    content=chunk["content"],
                    embedding=emb,
                    metadata_=chunk.get("metadata", {}),
                    content_hash=content_hash
                )
                records_to_insert.append(doc_record)
                
        if records_to_insert:
            self.db.add_all(records_to_insert)
            self.db.commit()
            logger.info(f"Successfully inserted {len(records_to_insert)} new chunks into pgvector.")
        else:
            logger.info("No new chunks to insert (all existing hashes).")
