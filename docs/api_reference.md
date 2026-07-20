# API Reference

This document outlines the REST and WebSocket endpoints for interacting with the Multimodal Predictive Maintenance Platform.

## Core Endpoints

### `POST /graph/predict`
Accepts a heterogeneous graph state (machines, sensors, conveyors) and returns fault probabilities along with GNN explanations for the top contributing neighbors.

### `POST /graph/predict/full`
End-to-End smoke test pipeline. Sends data through the Two-Tower fusion model, feeds it into the GraphSAGE model for topological context, and outputs a final calibrated anomaly score.
**Note:** This endpoint has been updated to run asynchronously via Celery to prevent API blocking.

**Response:**
```json
{
  "job_id": "uuid-string"
}
```

### `GET /graph/jobs/{job_id}`
Check the status of an async prediction job.

**Response (Pending):**
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "result": null
}
```

**Response (Success):**
```json
{
  "job_id": "uuid-string",
  "status": "success",
  "result": {
    "machine_id": "M1",
    "timestamp": 1234567890.1,
    "anomaly_score": 0.85,
    "is_anomaly": true,
    "threshold": 0.5,
    "cache_hit": false
  }
}
```

#### Async Polling Pattern Sequence Diagram

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Celery Worker
    participant Redis Broker

    Client->>API: POST /graph/predict/full (Payload)
    API->>Redis Broker: Enqueue Task
    API-->>Client: 200 OK { "job_id": "uuid" }
    
    Celery Worker->>Redis Broker: Pull Task
    Celery Worker->>Celery Worker: Run GNN & Fusion ML models (Long running)
    
    loop Every 1s
        Client->>API: GET /graph/jobs/uuid
        API->>Redis Broker: Check Status
        API-->>Client: 200 OK { "status": "pending" }
    end
    
    Celery Worker->>Redis Broker: Store Result (SUCCESS)
    
    Client->>API: GET /graph/jobs/uuid
    API->>Redis Broker: Check Status & Result
    API-->>Client: 200 OK { "status": "success", "result": {...} }
```

## Agent Endpoints (Day 20)

### `POST /agent/session/new`
Creates a new conversation session, returning a unique `session_id`.
**Response:**
```json
{
  "session_id": "uuid-string",
  "message": "Session created."
}
```

### `POST /agent/session/{session_id}/end`
Terminates an active session and clears the conversation history from Redis.

### `WebSocket /agent/chat/{session_id}`
Real-time conversational endpoint for interacting with the diagnostic agent.

#### Protocol
1. **Client** sends plain text message: `"Check machine_001"`
2. **Server** sends acknowledgment:
   ```json
   {"type": "status", "content": "Thinking..."}
   ```
3. **Server** streams agent thought process (Trace Steps):
   ```json
   {"type": "trace_step", "content": {"iteration": "1"}}
   {"type": "trace_step", "content": {"action": "query_sensor_db", "action_args": "{'machine_id': 'machine_001'}"}}
   {"type": "trace_step", "content": {"observation": {"status": "success", "recent_readings": [...]}}}
   ```
4. **Server** sends final answer:
   ```json
   {"type": "final_answer", "content": "The machine looks fine, no alerts found."}
   ```
