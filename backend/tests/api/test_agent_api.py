import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.api.agent_router import memory
from unittest.mock import patch

client = TestClient(app)

def test_session_lifecycle():
    # 1. Create session
    response = client.post("/agent/session/new")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    session_id = data["session_id"]
    
    # 2. Check memory is empty
    history = memory.get_history(session_id)
    assert len(history) == 0
    
    # 3. Add to memory
    memory.save_turn(session_id, "Hello", "Hi there")
    history = memory.get_history(session_id)
    assert len(history) == 2 # 1 user, 1 model
    
    # 4. End session
    response = client.post(f"/agent/session/{session_id}/end")
    assert response.status_code == 200
    
    # 5. Check memory cleared
    history = memory.get_history(session_id)
    assert len(history) == 0

@patch("backend.app.agent.memory.genai.Client")
def test_conversation_summarization(mock_client):
    session_id = "test_summarize_session"
    memory.max_turns = 2 # Set artificially low to trigger quickly (4 messages max)
    
    # Mock the summarize response
    mock_response = mock_client.return_value.models.generate_content.return_value
    mock_response.text = "Mock Summary"
    
    memory.clear(session_id)
    
    # Turn 1
    memory.save_turn(session_id, "Q1", "A1")
    # Turn 2
    memory.save_turn(session_id, "Q2", "A2")
    
    history = memory.get_history(session_id)
    assert len(history) == 4 # 2 turns
    
    # Turn 3 (Should trigger summarization: drop older, add summary + recent 2 turns = 5 messages)
    memory.save_turn(session_id, "Q3", "A3")
    
    history = memory.get_history(session_id)
    assert len(history) == 5
    # First message should be the summary
    assert "Mock Summary" in history[0].parts[0].text
    
    memory.clear(session_id)

def test_websocket_endpoint():
    # TestClient supports websocket testing
    response = client.post("/agent/session/new")
    session_id = response.json()["session_id"]
    
    with patch("backend.app.api.agent_router.ReActAgent") as mock_agent:
        mock_instance = mock_agent.return_value
        mock_instance.run.return_value = {
            "status": "success",
            "answer": "This is a mocked final answer",
            "trace": [
                "--- Iteration 1 ---",
                "ACTION: Calling `dummy_tool` with {}",
                "OBSERVATION: dummy result",
                "FINAL ANSWER: This is a mocked final answer"
            ]
        }
        
        with client.websocket_connect(f"/agent/chat/{session_id}") as websocket:
            websocket.send_text("Hello agent")
            
            # 1. Ack
            data = websocket.receive_json()
            assert data["type"] == "status"
            
            # 2. Trace step 1 (iteration + action)
            data = websocket.receive_json()
            assert data["type"] == "trace_step"
            assert "iteration" in data["content"]
            
            # 3. Trace step 2 (observation)
            data = websocket.receive_json()
            assert data["type"] == "trace_step"
            
            # 4. Final Answer Trace Step
            data = websocket.receive_json()
            assert data["type"] == "trace_step"
            assert "final_answer" in data["content"]
            
            # 5. Final Answer WebSocket message
            data = websocket.receive_json()
            assert data["type"] == "final_answer"
            assert data["content"] == "This is a mocked final answer"
            
    memory.clear(session_id)
