import pytest
from unittest.mock import MagicMock
from backend.app.rag.retriever_bm25 import BM25Retriever
from backend.app.rag.retriever_hybrid import HybridRetriever
from backend.app.rag.reranker import CrossEncoderReranker
from backend.app.db.models import Document

def test_bm25_retriever():
    mock_db = MagicMock()
    # Create some mock documents
    doc1 = Document(id="uuid-1", content="The bearing failure was caused by excessive vibration.", metadata_={"title": "Doc 1"})
    doc2 = Document(id="uuid-2", content="Hydraulic pump pressure drop due to cavitation.", metadata_={"title": "Doc 2"})
    
    mock_db.query().all.return_value = [doc1, doc2]
    
    retriever = BM25Retriever(db=mock_db)
    results = retriever.retrieve("vibration")
    
    assert len(results) > 0
    assert results[0]["id"] == "uuid-1"

def test_hybrid_retriever_query_expansion():
    mock_db = MagicMock()
    retriever = HybridRetriever(db=mock_db)
    
    expanded = retriever.expander.expand("bearing failure")
    assert "bearing failure" in expanded
    assert "bearing wear" in expanded
    assert "bearing degradation" in expanded

def test_reranker_shape():
    reranker = CrossEncoderReranker()
    docs = [
        {"content": "Irrelevant text about something else.", "id": "1"},
        {"content": "This text is highly relevant to the pump cavitation issue.", "id": "2"}
    ]
    
    results = reranker.rerank("pump cavitation", docs, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "2"
