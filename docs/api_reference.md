# API Reference

This document outlines the REST and WebSocket endpoints for interacting with the Multimodal Predictive Maintenance Platform.

## Core Endpoints

### `POST /graph/predict`
Accepts a heterogeneous graph state (machines, sensors, conveyors) and returns fault probabilities along with GNN explanations for the top contributing neighbors.

### `POST /graph/predict/full`
End-to-End smoke test pipeline. Sends data through the Two-Tower fusion model, feeds it into the GraphSAGE model for topological context, and outputs a final calibrated anomaly score.

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
