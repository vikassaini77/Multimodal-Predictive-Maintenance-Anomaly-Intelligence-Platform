# Product Documentation: Multimodal Predictive Maintenance

## Alert Deduplication Strategy
To prevent alert fatigue and operator overload, the system implements a robust deduplication strategy for all edge anomalies detected.

### 1. Redis-Backed Deduplication Window
Every time the `DispatchAlertTool` attempts to dispatch an alert, it records the event in Redis using a 10-minute Time-To-Live (TTL) key (`alert_dedup:{machine_id}:{fault_type}`). 
- **First Occurrence**: The alert is dispatched via WebSocket to the Edge Dashboard, and if the severity is Critical, an email is sent via the Resend API to the on-call maintenance team.
- **Subsequent Occurrences**: The system increments the Redis counter but **suppresses** duplicate emails and disruptive full-screen notifications. Instead, a lightweight `alert_increment` WebSocket event is dispatched to update the UI counter (e.g., "Same fault seen 47 times") in real-time.

### 2. Maintenance Window Suppression
The platform allows operators to configure weekly recurring Maintenance Windows per Machine Zone (e.g., Milling Zone, Stamping Zone, Welding Zone). 
If an anomaly is detected while a zone is in an active maintenance window, the `DispatchAlertTool` intercepts and completely suppresses the alert to avoid false positives during routine servicing.

### 3. Zone Grouping
Alerts displayed in the `AlertFeed` are grouped by their respective physical factory zones. This spatial awareness allows operators to quickly identify cascading failures across interconnected machines within the same physical sector.
