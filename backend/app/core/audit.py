import datetime
import json
from sqlalchemy.orm import Session
from backend.app.db.models import AuditLog
from backend.app.db.session import SessionLocal

def log_audit_event(actor: str, action: str, inputs: dict, outputs: dict, outcome: str, machine_id: str = None, severity: str = None):
    """
    Logs an action to the audit logs table.
    Used for tracking ReAct agent tool calls, alerts, and other critical actions.
    """
    db: Session = SessionLocal()
    try:
        # Convert Pydantic objects to dicts if needed
        if hasattr(inputs, "model_dump"):
            inputs = inputs.model_dump()
        if hasattr(outputs, "model_dump"):
            outputs = outputs.model_dump()
            
        # Try converting outputs to a dict if it is a JSON string
        if isinstance(outputs, str):
            try:
                outputs = json.loads(outputs)
            except Exception:
                outputs = {"raw_output": outputs}

        audit_log = AuditLog(
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            actor=actor,
            action=action,
            machine_id=machine_id,
            severity=severity,
            inputs=inputs,
            outputs=outputs,
            outcome=outcome
        )
        db.add(audit_log)
        db.commit()
    except Exception as e:
        db.rollback()
        # In a robust production environment, you might log this error to a different channel, 
        # but you wouldn't necessarily crash the core process.
        print(f"Failed to write audit log: {e}")
    finally:
        db.close()
