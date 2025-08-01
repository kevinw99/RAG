"""Centralized logging configuration for RAG system.

Provides structured logging with different levels, formatters, and outputs.
Supports both development and production configurations.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

from ..config.settings import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self):
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add file and line info in debug mode
        if record.levelno >= logging.DEBUG:
            log_obj.update({
                'file': record.filename,
                'line': record.lineno,
                'function': record.funcName,
            })
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread',
                          'threadName', 'processName', 'process', 'getMessage',
                          'exc_info', 'exc_text', 'stack_info']:
                log_obj[key] = value
        
        return json.dumps(log_obj, default=str)


class ContextualFormatter(logging.Formatter):
    """Human-readable formatter with context information."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with additional context."""
        # Add context information
        if hasattr(record, 'request_id'):
            record.name = f"{record.name}[{record.request_id}]"
        
        if hasattr(record, 'user_id'):
            record.message = f"[user:{record.user_id}] {record.getMessage()}"
        else:
            record.message = record.getMessage()
        
        return super().format(record)


class RAGLogger:
    """Centralized logger configuration for RAG system."""
    
    def __init__(self):
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self.configured = False
    
    def configure_logging(self, 
                         log_level: str = "INFO",
                         log_format: str = "contextual",
                         log_file: Optional[str] = None,
                         max_file_size: int = 10 * 1024 * 1024,  # 10MB
                         backup_count: int = 5,
                         enable_console: bool = True,
                         enable_file: bool = True) -> None:
        """Configure logging system.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Format type (contextual, json)
            log_file: Log file path (optional)
            max_file_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
            enable_console: Enable console logging
            enable_file: Enable file logging
        """
        if self.configured:
            return
        
        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Choose formatter
        if log_format.lower() == "json":
            formatter = JSONFormatter()
        else:
            formatter = ContextualFormatter()
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            root_logger.addHandler(console_handler)
            self.handlers['console'] = console_handler
        
        # File handler
        if enable_file:
            if not log_file:
                log_file = "logs/rag_system.log"
            
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            root_logger.addHandler(file_handler)
            self.handlers['file'] = file_handler
        
        # Configure specific loggers with appropriate levels
        self._configure_specific_loggers()
        
        self.configured = True
        
        # Log configuration
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured: level={log_level}, format={log_format}, "
                   f"console={enable_console}, file={enable_file}")
    
    def _configure_specific_loggers(self):
        """Configure specific loggers with appropriate levels."""
        # RAG system loggers
        rag_loggers = [
            'rag_system.core',
            'rag_system.storage',
            'rag_system.retrieval', 
            'rag_system.generation',
            'rag_system.api',
            'rag_system.monitoring'
        ]
        
        for logger_name in rag_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)  # Default to INFO for RAG components
        
        # External library loggers (reduce verbosity)
        external_loggers = {
            'chromadb': logging.WARNING,
            'sentence_transformers': logging.WARNING,
            'transformers': logging.WARNING,
            'urllib3': logging.WARNING,
            'httpx': logging.WARNING,
            'fastapi': logging.INFO,
            'uvicorn': logging.INFO,
        }
        
        for logger_name, level in external_loggers.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get configured logger for a component.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        if not self.configured:
            self.configure_logging()
        
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        
        return self.loggers[name]
    
    def add_request_context(self, logger: logging.Logger, request_id: str, 
                           user_id: Optional[str] = None) -> logging.LoggerAdapter:
        """Add request context to logger.
        
        Args:
            logger: Base logger
            request_id: Request ID
            user_id: Optional user ID
            
        Returns:
            Logger adapter with context
        """
        extra = {'request_id': request_id}
        if user_id:
            extra['user_id'] = user_id
        
        return logging.LoggerAdapter(logger, extra)
    
    def create_component_logger(self, component_name: str, 
                               log_level: Optional[str] = None) -> logging.Logger:
        """Create logger for a specific component.
        
        Args:
            component_name: Name of the component
            log_level: Optional specific log level
            
        Returns:
            Component logger
        """
        logger = self.get_logger(f"rag_system.{component_name}")
        
        if log_level:
            logger.setLevel(getattr(logging, log_level.upper()))
        
        return logger


# Global logger instance
rag_logger = RAGLogger()


# Convenience functions
def configure_logging(**kwargs):
    """Configure logging system."""
    rag_logger.configure_logging(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """Get logger for component."""
    return rag_logger.get_logger(name)


def get_component_logger(component_name: str, log_level: Optional[str] = None) -> logging.Logger:
    """Get component-specific logger."""
    return rag_logger.create_component_logger(component_name, log_level)


def add_request_context(logger: logging.Logger, request_id: str, 
                       user_id: Optional[str] = None) -> logging.LoggerAdapter:
    """Add request context to logger."""
    return rag_logger.add_request_context(logger, request_id, user_id)


# Component loggers
def get_api_logger() -> logging.Logger:
    """Get API logger."""
    return get_component_logger("api")


def get_core_logger() -> logging.Logger:
    """Get core system logger."""
    return get_component_logger("core")


def get_storage_logger() -> logging.Logger:
    """Get storage logger."""
    return get_component_logger("storage")


def get_retrieval_logger() -> logging.Logger:
    """Get retrieval logger."""
    return get_component_logger("retrieval")


def get_generation_logger() -> logging.Logger:
    """Get generation logger."""
    return get_component_logger("generation")


def get_monitoring_logger() -> logging.Logger:
    """Get monitoring logger."""
    return get_component_logger("monitoring")


# Auto-configure logging on import if not already configured
def auto_configure():
    """Auto-configure logging with default settings."""
    if not rag_logger.configured:
        # Use settings if available
        log_level = getattr(settings, 'log_level', 'INFO')
        log_format = getattr(settings, 'log_format', 'contextual')
        
        # Check if we're in production
        is_production = getattr(settings, 'environment', 'development') == 'production'
        
        rag_logger.configure_logging(
            log_level=log_level,
            log_format='json' if is_production else log_format,
            enable_console=True,
            enable_file=True,
            log_file="logs/rag_system.log"
        )


# Auto-configure on import
auto_configure()