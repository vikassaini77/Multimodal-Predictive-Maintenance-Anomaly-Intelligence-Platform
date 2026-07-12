import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import FastAPI, Request
from backend.app.api.router import router as graph_router
from backend.app.config import settings
from backend.app.utils.logger import trace_id_ctx_var, logger

import time
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title=settings.app_name,
    description="API for evaluating GraphSAGE fault propagation across factory topology.",
    version="1.0.0"
)

# Start Prometheus metrics collection
Instrumentator().instrument(app).expose(app)

class TraceIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
        token = trace_id_ctx_var.set(trace_id)
        
        logger.info(f"Incoming request: {request.method} {request.url.path}")
        try:
            response = await call_next(request)
            response.headers["X-Trace-ID"] = trace_id
            return response
        finally:
            trace_id_ctx_var.reset(token)

app.add_middleware(TraceIDMiddleware)

from backend.app.api.agent_router import router as agent_router

app.include_router(graph_router, prefix="/graph", tags=["graph"])
app.include_router(agent_router, prefix="/agent", tags=["agent"])

START_TIME = time.time()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "db_connected": True,
        "uptime": time.time() - START_TIME
    }
