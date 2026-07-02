# Autonomous Diagnostic Agent (ReAct)

The ReAct Agent sits on top of the `InferencePipeline` and `RAGAnswerer`, acting as the brain of the platform. Using Gemini 2.5 Flash's function-calling capabilities, it can autonomously query sensors, trigger deep learning anomaly models, and look up maintenance procedures.

## ReAct Architecture (Thought -> Action -> Observation)

1. **User Request**: The operator asks a question (e.g., "Machine 5 is vibrating, what's wrong?").
2. **Thought**: The LLM reasons about what information it needs.
3. **Action**: The LLM outputs a function call (e.g., `run_anomaly_score(machine_5)`).
4. **Observation**: The system executes the Python tool and returns the JSON result to the LLM context window.
5. **Repeat**: The LLM loops until it has sufficient evidence to make a final diagnosis and recommendation.

## Tool Catalog

### `query_sensor_db`
- **Description**: Fetches recent raw sensor readings (temperature, vibration, acoustic).
- **Arguments**: `machine_id` (str), `limit` (int)

### `run_anomaly_score`
- **Description**: Runs the multimodal deep learning pipeline to calculate an anomaly score and identify faults.
- **Arguments**: `machine_id` (str), `timestamp` (str)

### `search_manual_rag`
- **Description**: Queries the OEM maintenance manuals (using Day 17's RAG Pipeline).
- **Arguments**: `query` (str)
