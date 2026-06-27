import logging
import uuid
import contextvars
from logging import LogRecord
from backend.app.config import settings

# Context variable to hold the trace ID
trace_id_ctx_var = contextvars.ContextVar("trace_id", default="NO-TRACE-ID")

class TraceIDFilter(logging.Filter):
    """
    Injects the trace_id from the context variable into the log record.
    """
    def filter(self, record: LogRecord) -> bool:
        record.trace_id = trace_id_ctx_var.get()
        return True

def setup_logger(name: str = "industrial_mind") -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Structured log format including trace_id
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s] - %(message)s'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.addFilter(TraceIDFilter())
        
        # Prevent propagation to the root logger to avoid double-logging
        logger.propagate = False
        
    return logger

logger = setup_logger()
