import os
import resend

def send_critical_alert_email(machine_id: str, message: str, score: float):
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        print(f"[Email Mock] Critical Alert for {machine_id}: {message}")
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
        r = resend.Emails.send({
            "from": "alerts@resend.dev",
            "to": "oncall@industrial-ai.local",
            "subject": f"CRITICAL: {machine_id} Failing",
            "html": html_content
        })
        print(f"Dispatched critical email via Resend: {r}")
    except Exception as e:
        print(f"Failed to send email: {e}")
