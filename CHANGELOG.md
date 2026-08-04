# Changelog

All notable changes to the **Multimodal Predictive Maintenance & Anomaly Intelligence Platform** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-08-04

### Added
- **Async Jobs**: Improved background processing for model drift evaluation and long-running analytics.
- **Authentication**: JWT-based authentication across all API endpoints with refresh token rotation.
- **Rate Limiting**: Redis-backed rate limiting per IP address across all public and authenticated routes.
- **Audit Logs**: Comprehensive event tracking for user actions (CRUD) with persistent Redis queue.
- **Alert Deduplication**: Edge feed deduplication using Redis `INCR` to prevent redundant failure notifications for the same machine ID and timestamp window.
- **Chaos Tests**: Introduced the `AgentBenchmark` chaos suite to evaluate agent responses under severe fault cascades.
- **Structured Logging**: Migrated all standard print statements to `structlog` for JSON-formatted, context-rich logging (includes trace IDs, machine IDs, user IDs).
- **Connection Pooling**: Improved DB latency and reliability by adding SQLAlchemy connection pooling (size=20, max_overflow=10).
- **Maintenance Windows**: New CRUD endpoints for maintenance scheduling to suppress alerts on machines during active maintenance.

### Changed
- Refactored middleware into a strict sequential stack (`Auth -> Rate Limit -> Audit Log -> Request ID`).
- Redesigned `AlertFeed` on frontend to group anomalies by physical factory zone with badge counters for duplicate events.

### Fixed
- Fixed database connection exhaustion under heavy concurrent load (Locust testing verified 0% error rate).
- Fixed missing `X-Trace-ID` in internal agent tool executions.

## [1.0.0] - 2026-07-14

### Added
- **Two-Tower Multimodal AI Model**: Merged 1D sensor telemetry and 2D visual anomaly models.
- **GraphSAGE Topology Learning**: Predicts cascading failures across interconnected factory nodes.
- **ReAct Autonomous Agent**: Powered by Gemini 2.5 Flash, capable of querying sensors, viewing heatmaps, and retrieving OEM manuals.
- **Pgvector RAG Pipeline**: Lightning-fast vector retrieval of chunked maintenance manuals.
- **React Frontend Dashboard**: Live updating WebSockets, SVG factory map, and Agent Chat UI.
- **Edge Deployment**: TensorRT + ONNX quantized inference service exposed via gRPC.
- **Observability Stack**: Prometheus + Grafana metrics, JSON structured logging for Loki.
- **Docker Compose Orchestration**: Single command boot-up sequence with internal health checks.
- **GitHub Actions CI/CD**: Automatic builds and smoke testing on merge to `main`.
- **Demo Mode**: 30-minute factory simulation script showcasing sensor drift and agentic resolution.

### Changed
- Refactored `logger.py` to use `python-json-logger`.
- Transitioned initial simple graph models to full GraphSAGE architecture.
- Upgraded TailwindCSS to v4.0 native `@theme` integration.
- Switched from HTTP polling to Asyncio WebSockets for UI telemetry.

### Fixed
- Addressed memory leaks in PyTorch DataLoader during extended testing.
- Resolved race conditions in gRPC Edge initialization.
- Fixed layout shifts in React Dashboard alert feed.

### Security
- Integrated authentication headers (`X-API-Key`) for Agent tooling queries.
- Human-in-the-loop guardrails (approve/deny) for critical agent actions.
