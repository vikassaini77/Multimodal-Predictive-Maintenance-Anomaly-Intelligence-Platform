import logging
import uuid
import contextvars
import structlog
from backend.app.config import settings

# Context variable to hold the trace ID
trace_id_ctx_var = contextvars.ContextVar("trace_id", default="NO-TRACE-ID")

def inject_context_vars(logger, log_method, event_dict):
    """
    Injects context variables into the structlog event dictionary.
    """
    event_dict["trace_id"] = trace_id_ctx_var.get()
    return event_dict

def setup_logger(name: str = "industrial_mind"):
    # Configure standard logging to be captured by structlog
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            inject_context_vars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger(name)

logger = setup_logger()
