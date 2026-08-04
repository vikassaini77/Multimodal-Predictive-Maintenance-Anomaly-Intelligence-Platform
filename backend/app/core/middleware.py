import uuid
import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, status
from fastapi.responses import JSONResponse
from backend.app.utils.logger import trace_id_ctx_var, logger
from backend.app.core.security import verify_token, limiter
from slowapi.errors import RateLimitExceeded

class CoreMiddlewareStack(BaseHTTPMiddleware):
    def __init__(self, app, public_paths=None):
        super().__init__(app)
        self.public_paths = public_paths or ["/auth/login", "/auth/refresh", "/health"]

    async def dispatch(self, request: Request, call_next):
        # 1. Request ID (Generate first so all logs have it)
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
        token = trace_id_ctx_var.set(trace_id)
        
        try:
            # 2. Auth (Check early to reject unauthenticated)
            path = request.url.path
            user_id = "anonymous"
            if not any(path.startswith(p) for p in self.public_paths) and not path.startswith("/ws/"):
                auth_token = request.cookies.get("access_token")
                if not auth_token:
                    auth_header = request.headers.get("Authorization")
                    if auth_header and auth_header.startswith("Bearer "):
                        auth_token = auth_header.split(" ")[1]
                
                if not auth_token:
                    return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
                
                try:
                    payload = verify_token(auth_token, expected_type="access")
                    user_id = payload.get("sub", "unknown")
                    request.state.user = payload
                except Exception as e:
                    return JSONResponse(status_code=401, content={"detail": "Invalid token"})

            # 3. Rate Limit (Enforce limits after knowing user is authenticated)
            # We skip manual rate limit enforcement here if we rely on route decorators, 
            # but to ensure strict ordering, SlowAPI typically runs before route handlers.
            
            # 4. Audit Log
            start_time = time.time()
            logger.info(f"AUDIT LOG: Incoming {request.method} {path} user={user_id}")
            
            response = await call_next(request)
            
            process_time = time.time() - start_time
            logger.info(f"AUDIT LOG: Completed {request.method} {path} status={response.status_code} in {process_time:.4f}s user={user_id}")
            
            response.headers["X-Trace-ID"] = trace_id
            return response
            
        finally:
            trace_id_ctx_var.reset(token)
