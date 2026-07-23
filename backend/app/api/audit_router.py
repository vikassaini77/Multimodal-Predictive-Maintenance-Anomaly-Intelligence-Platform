from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
import datetime

from backend.app.db.session import get_db
from backend.app.db.models import AuditLog
from backend.app.core.security import get_current_user

router = APIRouter()

@router.get("/logs")
def get_audit_logs(
    machine_id: Optional[str] = Query(None, description="Filter by machine ID"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    date: Optional[datetime.date] = Query(None, description="Filter by specific date (YYYY-MM-DD)"),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Retrieve audit logs for agent tool calls and alerts.
    Requires authentication.
    """
    query = db.query(AuditLog)
    
    if machine_id:
        query = query.filter(AuditLog.machine_id == machine_id)
    
    if severity:
        query = query.filter(AuditLog.severity == severity)
        
    if date:
        # Filter for the entire day (from 00:00:00 to 23:59:59)
        start_dt = datetime.datetime.combine(date, datetime.time.min).replace(tzinfo=datetime.timezone.utc)
        end_dt = datetime.datetime.combine(date, datetime.time.max).replace(tzinfo=datetime.timezone.utc)
        query = query.filter(AuditLog.timestamp >= start_dt, AuditLog.timestamp <= end_dt)
        
    logs = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    
    return [
        {
            "id": str(log.id),
            "timestamp": log.timestamp.isoformat(),
            "actor": log.actor,
            "action": log.action,
            "machine_id": log.machine_id,
            "severity": log.severity,
            "inputs": log.inputs,
            "outputs": log.outputs,
            "outcome": log.outcome
        }
        for log in logs
    ]
