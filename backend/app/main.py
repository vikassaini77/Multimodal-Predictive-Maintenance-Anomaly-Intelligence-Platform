import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import FastAPI, Request, Depends
from backend.app.api.router import router as graph_router
from backend.app.config import settings
from backend.app.utils.logger import trace_id_ctx_var, logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from backend.app.core.security import limiter

import time
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title=settings.app_name,
    description="API for evaluating GraphSAGE fault propagation across factory topology.",
    version="1.0.0"
)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    response = JSONResponse(
        {"error": f"Rate limit exceeded: {exc.detail}"}, status_code=429
    )
    # _rate_limit_exceeded_handler already adds Retry-After, but we enforce it here manually
    response.headers["Retry-After"] = str(exc.headers.get("Retry-After", 60)) if hasattr(exc, "headers") else "60"
    return response

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
from backend.app.api.auth import router as auth_router
from backend.app.api.audit_router import router as audit_router
from backend.app.core.security import get_current_user

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(graph_router, prefix="/graph", tags=["graph"], dependencies=[Depends(get_current_user)])
app.include_router(agent_router, prefix="/agent", tags=["agent"], dependencies=[Depends(get_current_user)])
app.include_router(audit_router, prefix="/audit", tags=["audit"], dependencies=[Depends(get_current_user)])

START_TIME = time.time()

@app.get("/health")
@limiter.limit("1000/minute")
def health_check(request: Request):
    return {
        "status": "healthy",
        "model_loaded": True,
        "db_connected": True,
        "uptime": time.time() - START_TIME
    }
