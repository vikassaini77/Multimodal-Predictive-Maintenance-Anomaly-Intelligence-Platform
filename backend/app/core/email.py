import os
import resend
from backend.app.utils.logger import logger

def send_critical_alert_email(machine_id: str, message: str, score: float):
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        logger.warning(f"[Email Mock] Critical Alert for {machine_id}: {message}")
        return
        
    resend.api_key = api_key
    
    html_content = f"""
    <h2>Critical Machine Alert: {machine_id}</h2>
    <p><strong>Diagnosis:</strong> {message}</p>
    <p><strong>Anomaly Score:</strong> {score:.3f}</p>
    <br/>
    <p>Please check the Multimodal Predictive Maintenance Dashboard immediately.</p>
    """
    
    try:
        params = {
            "from": "alerts@resend.dev",
            "to": "oncall@industrial-ai.local",
            "subject": f"CRITICAL: {machine_id} Failing",
            "html": html_content
        }
        r = resend.Emails.send(params)
        logger.info(f"Dispatched critical email via Resend: {r}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
