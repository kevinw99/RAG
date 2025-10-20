#!/usr/bin/env python3
"""
Start ChromaDB HTTP server for web interface access.
This exposes your existing ChromaDB database via HTTP API.
"""

import uvicorn
import chromadb
from chromadb.config import Settings
import os
from pathlib import Path

def start_server():
    """Start ChromaDB server on port 8000."""
    
    # Path to your existing ChromaDB database
    db_path = "/Users/kweng/AI/RAG/data/indices/chroma_db"
    
    print(f"🚀 Starting ChromaDB Server...")
    print(f"📁 Database path: {db_path}")
    print(f"🌐 Server URL: http://localhost:8000")
    print(f"🔍 Collection: rag_documents")
    print(f"📊 Total chunks: 13,565")
    print()
    print("💡 To connect from web interface:")
    print("   - ChromaDB URL: http://localhost:8000")
    print("   - Collection: rag_documents")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Configure ChromaDB for HTTP server mode
    os.environ["CHROMA_DB_IMPL"] = "chromadb.db.duckdb.DuckDB"
    os.environ["CHROMA_PERSIST_DIRECTORY"] = db_path
    
    # Start the server
    try:
        # Import chromadb server components
        from chromadb.api.fastapi import app
        
        # Run with uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        
    except ImportError:
        print("❌ ChromaDB server components not available.")
        print("📦 Installing chromadb server dependencies...")
        
        import subprocess
        subprocess.run(["pip", "install", "chromadb[server]"], check=True)
        
        print("✅ Dependencies installed. Please restart this script.")
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print()
        print("🔧 Alternative: Manual ChromaDB server startup")
        print("   Run this command in terminal:")
        print(f"   chroma run --path {db_path} --port 8000")

if __name__ == "__main__":
    start_server()