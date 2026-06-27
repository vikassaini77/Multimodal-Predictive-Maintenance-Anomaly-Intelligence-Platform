# Week 3 Roadmap: RAG Pipeline & ReAct Agent

In Week 3, we transition from the core ML inference engine to building the AI Agent layer that will consume these predictions and provide actionable maintenance intelligence.

## Goals
1. Implement a Retrieval-Augmented Generation (RAG) pipeline to query OEM maintenance manuals and past incident reports.
2. Build a ReAct (Reasoning and Acting) Agent using Gemini 2.5 Flash to orchestrate diagnostics.
3. Integrate the Agent with the FastAPI backend and external alerting systems (e.g., Slack/Email).

## Day-by-Day Breakdown

- **Day 15**: Setup `pgvector` database and chunking strategies for PDF maintenance manuals.
- **Day 16**: Implement hybrid search (dense + sparse) for the RAG retriever.
- **Day 17**: Develop the ReAct Agent core loop and prompt templates.
- **Day 18**: Define Tool schemas for the Agent (e.g., `get_graph_prediction`, `search_manuals`, `get_sensor_history`).
- **Day 19**: Integrate Agent orchestration within the FastAPI backend (new `/agent/chat` endpoint).
- **Day 20**: Implement streaming responses and UI integration for the Agent chat interface.
- **Day 21**: Week 3 Review, E2E Agent testing, and deployment prep.
