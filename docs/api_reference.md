# API Reference: Graph Prediction Endpoint

The Multimodal Predictive Maintenance Platform exposes a FastAPI serving layer to score heterogeneous graph states for potential faults. 

## `POST /graph/predict`

Calculates fault risk probabilities per machine node using a pre-trained GraphSAGE network, and returns subgraph explanations detailing which connected equipment influenced the prediction.

### Request Body Schema
Expects a JSON payload matching the `GraphInput` schema.

```json
{
  "nodes": [
    {
      "id": "machine_01",
      "type": "machine",
      "features": [0.12, -0.45, ...] // 256-dim feature array
    },
    {
      "id": "conveyor_A",
      "type": "conveyor",
      "features": [0.88, 0.03, ...]
    }
  ],
  "edges": [
    {
      "source": "machine_01",
      "target": "conveyor_A",
      "type": "feeds_into"
    }
  ]
}
```

### Response Schema
Returns a list of `RiskPrediction` objects for every `machine` in the request graph. 

```json
[
  {
    "node_id": "machine_01",
    "node_type": "machine",
    "fault_probability": 0.87,
    "is_fault": true,
    "top_contributing_neighbors": [
      {
        "node_id": "conveyor_A",
        "node_type": "conveyor",
        "importance_score": 0.45
      }
    ]
  }
]
```

### cURL Example
```bash
curl -X 'POST' \
  'http://localhost:8000/graph/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "nodes": [
    {"id": "m1", "type": "machine", "features": [0.0]}
  ],
  "edges": []
}'
```
