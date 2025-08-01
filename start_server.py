#!/usr/bin/env python3
"""Simple script to start the RAG API server."""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and start server
from rag_system.api.server import start_server

if __name__ == "__main__":
    print("🚀 Starting RAG API Server...")
    print("=" * 50)
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("=" * 50)
    
    try:
        start_server(
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            workers=1
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed: {e}")