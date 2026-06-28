import pytest
from unittest.mock import MagicMock
from backend.app.rag.chunker import TextChunker
from backend.app.rag.embeddings import EmbeddingGenerator
from backend.app.rag.indexer import DocumentIndexer
from backend.app.rag.corpus import generate_synthetic_manuals

def test_chunker_boundary_cases():
    chunker = TextChunker(chunk_size=10, overlap=2)
    doc = {"title": "Test", "content": "This is a very short test document for chunking."}
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) > 0
    assert "metadata" in chunks[0]
    assert chunks[0]["metadata"]["title"] == "Test"

def test_embedding_shape():
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_batch(["Test sentence"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384

def test_upsert_idempotency():
    mock_db = MagicMock()
    # Mocking the query to return None the first time, and an object the second time
    mock_db.query().filter().first.side_effect = [None, MagicMock()]
    
    indexer = DocumentIndexer(db=mock_db)
    docs = generate_synthetic_manuals()[:1]
    
    # First index (should insert)
    indexer.index_documents(docs)
    assert mock_db.add_all.call_count == 1
    
    # Second index (should skip due to mock returning an existing object, but we mock the query side effect)
    # Re-setup the mock to simulate existing records
    mock_db.query().filter().first.side_effect = [MagicMock()] * 10 
    indexer.index_documents(docs)
    
    # add_all should still have been called only once
    assert mock_db.add_all.call_count == 1
