#!/usr/bin/env python3
"""Server startup script for RAG API.

Production-ready server launcher with configuration options.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rag_system.api.server import start_server
from rag_system.config.settings import settings

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_api.log')
        ]
    )

def main():
    """Main server startup function."""
    parser = argparse.ArgumentParser(description="RAG API Server")
    
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--config-file",
        help="Path to configuration file (optional)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load config file if provided
    if args.config_file:
        config_path = Path(args.config_file)
        if config_path.exists():
            logger.info(f"Loading configuration from: {config_path}")
            # Could implement config file loading here
        else:
            logger.warning(f"Configuration file not found: {config_path}")
    
    # Display startup info
    logger.info("=" * 60)
    logger.info("RAG API Server Starting")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"Vector Store Path: {settings.vector_store_path}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info("=" * 60)
    
    # Check environment
    if not settings.openai_api_key and settings.llm_provider.lower() == "openai":
        logger.warning("OpenAI API key not set - query endpoint will fail")
    
    try:
        # Start server
        start_server(
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()