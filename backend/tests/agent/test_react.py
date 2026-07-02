import pytest
from unittest.mock import MagicMock, patch
from backend.app.agent.react import ReActAgent
from backend.app.agent.tools.sensor_tool import QuerySensorDBTool
from backend.app.agent.tools.anomaly_tool import RunAnomalyScoreTool

@patch("backend.app.agent.react.genai.Client")
def test_react_agent_max_iterations(mock_genai_client):
    # Setup mock to constantly return function calls (infinite loop scenario)
    mock_instance = MagicMock()
    
    mock_response = MagicMock()
    mock_response.text = ""
    # Create a mock function call
    mock_func_call = MagicMock()
    mock_func_call.name = "query_sensor_db"
    mock_func_call.args = {"machine_id": "m1"}
    
    mock_part = MagicMock()
    mock_part.function_call = mock_func_call
    
    mock_content = MagicMock()
    mock_content.parts = [mock_part]
    
    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    
    mock_response.candidates = [mock_candidate]
    
    mock_instance.models.generate_content.return_value = mock_response
    mock_genai_client.return_value = mock_instance
    
    tools = [QuerySensorDBTool()]
    agent = ReActAgent(tools=tools, max_iterations=2)
    
    res = agent.run("Check machine m1")
    
    assert res["status"] == "timeout"
    assert "timeout" in res["answer"].lower() or "time" in res["answer"].lower()
    # verify generate_content was called exactly max_iterations times
    assert mock_instance.models.generate_content.call_count == 2
    
def test_tool_dispatch():
    tool = QuerySensorDBTool()
    res = tool.run(machine_id="m1", limit=1)
    assert "m1" in res
    assert "status" in res

def test_anomaly_tool():
    tool = RunAnomalyScoreTool()
    res = tool.run(machine_id="m1", timestamp="2023-10-27T10:10:00Z")
    assert "CRITICAL_ANOMALY" in res
