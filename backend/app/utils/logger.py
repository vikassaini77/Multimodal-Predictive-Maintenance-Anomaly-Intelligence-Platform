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
        
        from pythonjsonlogger import jsonlogger
        
        class CustomJsonFormatter(jsonlogger.JsonFormatter):
            def add_fields(self, log_record, record, message_dict):
                super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
                if not log_record.get('timestamp'):
                    log_record['timestamp'] = self.formatTime(record, self.datefmt)
                if log_record.get('level'):
                    log_record['level'] = log_record['level'].upper()
                else:
                    log_record['level'] = record.levelname

        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s %(trace_id)s'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.addFilter(TraceIDFilter())
        
        # Prevent propagation to the root logger to avoid double-logging
        logger.propagate = False
        
    return logger

logger = setup_logger()
