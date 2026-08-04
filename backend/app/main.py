import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import FastAPI, Request, Depends
from backend.app.api.router import router as graph_router
from backend.app.config import settings
from backend.app.utils.logger import trace_id_ctx_var, logger
from backend.app.db.session import init_db
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from backend.app.core.security import limiter
from backend.app.core.middleware import CoreMiddlewareStack

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

@app.exception_handler(RequestValidationError)
def custom_validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error.get("loc", []))
        errors.append({
            "field": field,
            "error": error.get("msg")
        })
    return JSONResponse(
        status_code=422,
        content={"message": "Validation failed", "details": errors}
    )

# Start Prometheus metrics collection
Instrumentator().instrument(app).expose(app)

app.add_middleware(CoreMiddlewareStack)

from backend.app.api.agent_router import router as agent_router
from backend.app.api.auth import router as auth_router
from backend.app.api.audit_router import router as audit_router
from backend.app.api.maintenance_router import router as maintenance_router

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(graph_router, prefix="/graph", tags=["graph"])
app.include_router(agent_router, prefix="/agent", tags=["agent"])
app.include_router(audit_router, prefix="/audit", tags=["audit"])
app.include_router(maintenance_router, prefix="/maintenance", tags=["maintenance"])

init_db()

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

from fastapi import WebSocket, WebSocketDisconnect
import redis
import asyncio

@app.websocket("/ws/edge-feed")
async def edge_feed_websocket(websocket: WebSocket):
    await websocket.accept()
    r = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)
    pubsub = r.pubsub()
    pubsub.subscribe("edge_alerts")
    
    try:
        while True:
            message = pubsub.get_message(ignore_subscribe_messages=True)
            if message and message["type"] == "message":
                await websocket.send_text(message["data"])
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pubsub.unsubscribe("edge_alerts")
    except Exception as e:
        pubsub.unsubscribe("edge_alerts")
        logger.error(f"Edge Feed WebSocket error: {e}")
