import pytest
import json
from datetime import datetime, time
from unittest.mock import patch, MagicMock
from backend.app.agent.tools.alert_tool import DispatchAlertTool
from backend.app.agent.guardrails import RateLimiter, _in_memory_cache
from backend.app.db.models import MaintenanceWindow

@pytest.fixture(autouse=True)
def clean_cache():
    _in_memory_cache.clear()
    yield
    _in_memory_cache.clear()

@pytest.fixture
def mock_db_session():
    with patch("backend.app.agent.tools.alert_tool.SessionLocal") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        yield mock_session

def test_deduplication():
    tool = DispatchAlertTool()
    
    with patch("backend.app.agent.tools.alert_tool.send_critical_alert_email") as mock_email:
        # First call should succeed and send email
        res1 = json.loads(tool.run("CNC-01", "critical", "Test Fault", True))
        assert res1["status"] == "success"
        mock_email.assert_called_once()
        
        # Second call should be deduplicated
        res2 = json.loads(tool.run("CNC-01", "critical", "Test Fault", True))
        assert res2["status"] == "deduplicated"
        assert "seen 2 times" in res2["message"]
        # Email shouldn't be called again
        mock_email.assert_called_once()

        # 50th call should be deduplicated
        for _ in range(3, 51):
            tool.run("CNC-01", "critical", "Test Fault", True)
            
        res50 = json.loads(tool.run("CNC-01", "critical", "Test Fault", True))
        assert res50["status"] == "deduplicated"
        assert "seen 51 times" in res50["message"]
        mock_email.assert_called_once()

def test_maintenance_window_suppression(mock_db_session):
    # Setup active maintenance window in mock
    now = datetime.now()
    window = MaintenanceWindow(
        zone="Milling Zone",
        day_of_week=now.weekday(),
        start_time=time(0, 0),
        end_time=time(23, 59)
    )
    mock_db_session.query().filter().all.return_value = [window]
    
    tool = DispatchAlertTool()
    
    with patch("backend.app.agent.tools.alert_tool.send_critical_alert_email") as mock_email:
        res = json.loads(tool.run("CNC-01", "critical", "Test Fault", True))
        assert res["status"] == "suppressed"
        assert "scheduled maintenance window" in res["message"]
        
        # Should not send email
        mock_email.assert_not_called()
