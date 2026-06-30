import pytest
from unittest.mock import MagicMock, patch
from backend.app.llm.client import GeminiClient
from backend.app.llm.schemas import RAGAnswer
from backend.app.llm.guard import HallucinationGuard

@patch("backend.app.llm.client.genai.Client")
def test_gemini_client_retry(mock_genai_client):
    # Setup mock to fail twice with an Exception, then succeed
    mock_instance = MagicMock()
    
    # We mock the generate_content to raise an exception on first 2 calls
    class RateLimitException(Exception): pass
    
    mock_response = MagicMock()
    mock_response.text = '{"answer": "Test", "confidence": 0.9, "sources": []}'
    
    mock_instance.models.generate_content.side_effect = [
        RateLimitException("429 Too Many Requests"),
        RateLimitException("429 Too Many Requests"),
        mock_response
    ]
    
    mock_genai_client.return_value = mock_instance
    
    client = GeminiClient()
    # It should succeed after 2 retries
    res = client.generate("Test prompt")
    assert "Test" in res
    assert mock_instance.models.generate_content.call_count == 3

def test_hallucination_guard():
    guard = HallucinationGuard()
    
    retrieved_chunks = [
        {"id": "chunk_1", "content": "valid"},
        {"id": "chunk_2", "content": "valid"}
    ]
    
    # Test valid answer
    ans_valid = RAGAnswer(answer="OK", confidence=0.9, sources=["chunk_1"])
    res = guard.verify_citations(ans_valid, retrieved_chunks)
    assert res.confidence == 0.9
    assert len(res.sources) == 1
    
    # Test hallucinated answer
    ans_fake = RAGAnswer(answer="Fake", confidence=0.9, sources=["chunk_1", "fake_chunk"])
    res2 = guard.verify_citations(ans_fake, retrieved_chunks)
    assert res2.confidence == 0.4  # Penalized by 0.5
    assert "fake_chunk" not in res2.sources
    assert "chunk_1" in res2.sources
