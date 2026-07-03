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

### `fetch_image_snapshot`
- **Description**: Fetches the latest visual frame metadata for a machine.
- **Arguments**: `machine_id` (str)

### `dispatch_alert` (ACTION TOOL)
- **Description**: Dispatches an email/webhook alert to the maintenance team.
- **Arguments**: `machine_id` (str), `severity` (str), `message` (str), `human_confirmed` (bool)

## Safety Guardrails & Registry

To prevent autonomous chaos, the agent operates under strict safety layers:

1. **Permission Scopes (`ToolRegistry`)**:
   - Tools are categorized as `READ_ONLY` or `ACTION`.
   - The agent is instantiated with a specific scope. By default, it runs as `READ_ONLY` and literally cannot see or invoke action tools.

2. **Rate Limiting**:
   - A Redis-backed sliding window rate limiter prevents the agent from spamming actions.
   - For example, `dispatch_alert` is limited to max 1 alert per machine per 10 minutes.

3. **Human-in-the-Loop Confirmation**:
   - High-severity actions (e.g., `critical` alerts or automated shutdowns) are intercepted by the `ActionGuard`.
   - They require an explicit `human_confirmed=True` flag. If the agent attempts a critical action without this flag, the tool returns a "blocked" error message, forcing the agent to stop and ask the user for authorization in its final response.
