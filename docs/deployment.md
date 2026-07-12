# Deployment & Orchestration Guide

This platform is designed to be deployed using Docker Compose for simple, robust orchestration across all 8 services.

## Architecture Orchestration
The `docker-compose.yml` spins up:
1. **postgres** - pgvector database
2. **redis** - caching and pub/sub
3. **qdrant** - vector search for RAG
4. **api** - FastAPI backend
5. **edge** - gRPC edge inference service
6. **frontend** - React Dashboard (Vite + Nginx)
7. **prometheus** - Time-series metrics
8. **grafana** - Visual dashboards

## Quick Start (Local & Staging)

1. Clone the repository and navigate to the root directory.
2. Ensure you have Docker and Docker Compose installed.
3. Export your Gemini API Key:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
4. Build and start the entire stack:
   ```bash
   docker compose up --build -d
   ```

## Verification & Observability

- **Frontend Dashboard**: http://localhost:80
- **API Swagger Docs**: http://localhost:8000/docs
- **Grafana Dashboards**: http://localhost:3000 (admin/admin)
- **API Metrics**: http://localhost:8000/metrics
- **API Health**: http://localhost:8000/health
- **Edge Health**: http://localhost:8080/health

### CI/CD
This repository is configured with GitHub Actions. On every push to `main`, the CI pipeline will automatically test the `docker compose build` and run a smoke test to ensure all container health checks pass within 60 seconds.
