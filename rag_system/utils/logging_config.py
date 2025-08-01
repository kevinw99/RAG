"""
Logging configuration for RAG system.

Provides structured logging with JSON format and correlation IDs.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format: str = "json",
    log_file: Optional[str] = None
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level
        format: Log format ("json" or "text")
        log_file: Optional log file path
    """
    if format == "json":
        log_format = (
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
