# Runbook: Dead Letter Queue (DLQ) Management

This runbook outlines how to monitor, investigate, and mitigate failed background jobs in the Multimodal Predictive Maintenance Platform. We use Celery with a Redis broker, where permanently failed tasks (after exhausting 3 retries) are routed to a Dead Letter Queue (`dlq`).

## Alerts and Monitoring

1. **Dashboard UI:** The React dashboard contains a "Dead Letter Queue" panel that automatically appears when jobs fail. It lists the task ID, failure time, and an expandable stack trace.
2. **Toast Notifications:** A real-time toast alert will slide in from the top right of the dashboard whenever a job exhausts its retries and is moved to the DLQ.
3. **Prometheus / Grafana:** We run a `celery-exporter` service (port 9808) which exposes Celery metrics. Check the Grafana dashboard for:
   - `celery_task_failed_total`: High spike indicates systemic issues (e.g., DB down).
   - `celery_queue_length`: If the main queue is backed up, workers might be starved.

## Investigating Failed Jobs

1. **View the DLQ Panel:** Open the Edge Monitoring Dashboard and look for the DLQ panel. Click "View Error" to expand the Python traceback.
2. **Common Failure Modes:**
   - **GPU OOM (Out of Memory):** Transient OOMs should be caught by the 3 automatic retries. If they end up in the DLQ, the batch size is too large or the GPU is permanently hung.
   - **Network/DB Timeout:** If Postgres or Qdrant are unreachable.
   - **Data Validation Errors:** A machine payload may have missing sensor channels or wrong visual dimensions.

## Resolving and Requeueing

Currently, the DLQ is stored as a Redis List (`dlq`). 

To inspect the queue via CLI:
```bash
docker exec -it <redis_container_id> redis-cli
LRANGE dlq 0 -1
```

To clear the queue manually (if jobs are unrecoverable):
```bash
docker exec -it <redis_container_id> redis-cli DEL dlq
```

*Future Enhancement:* We plan to add a "Requeue" button next to each job in the UI which will POST to a new `/jobs/failed/{task_id}/retry` endpoint, popping the item from the `dlq` list and pushing it back into Celery.

## Known Chaos Failure Modes & Fallbacks

1. **GPU Out of Memory (OOM):**
   - **Behavior:** The inference pipeline automatically detects `CUDA out of memory`. It clears the GPU cache, transfers models and data to the CPU, and re-runs inference.
   - **Client Impact:** The API responds successfully (200 OK), but the payload includes `"latency_warning": true`.
   - **Action:** If this occurs frequently, monitor batch sizes and consider scaling up GPU resources or adding a dedicated queue limit.

2. **Database (PostgreSQL) Unreachable:**
   - **Behavior:** The `db_circuit_breaker` trips after 3 consecutive connection failures. Subsequent requests fail fast without waiting for timeouts.
   - **Client Impact:** The API immediately returns a `503 Service Unavailable` with a message "Circuit Breaker OPEN".
   - **Action:** Check PostgreSQL container health and network connectivity. The circuit breaker automatically tests recovery (half-open) after 30 seconds.

3. **Redis Unreachable:**
   - **Behavior:** `slowapi` rate limiter is configured to swallow errors. If Redis drops, the rate limiter fails open.
   - **Client Impact:** All requests pass without false `429 Too Many Requests` errors.
   - **Action:** Restart Redis container. Note that during downtime, the API is vulnerable to abuse as rate limits are inactive.
