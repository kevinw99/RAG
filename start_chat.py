#!/usr/bin/env python3
"""
Startup script for RAG Chat System

Starts both the FastAPI backend and Streamlit frontend for a complete chat experience.
"""

import subprocess
import sys
import time
import requests
import os
import signal
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import requests
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("💡 Install requirements: pip install -r requirements_streamlit.txt")
        return False

def is_port_in_use(port):
    """Check if a port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_api(url="http://localhost:8000/health", timeout=30):
    """Wait for the FastAPI server to be ready."""
    print("⏳ Waiting for RAG API to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print("✅ RAG API is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    
    print("❌ RAG API failed to start within timeout")
    return False

def start_fastapi_server():
    """Start the FastAPI server in the background."""
    if is_port_in_use(8000):
        print("ℹ️  FastAPI server already running on port 8000")
        return None
    
    print("🚀 Starting RAG FastAPI server...")
    
    # Start the server
    process = subprocess.Popen([
        sys.executable, "start_server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    return process

def start_streamlit():
    """Start the Streamlit chat interface."""
    print("🎨 Starting Streamlit chat interface...")
    
    # Change to the directory containing streamlit_chat.py
    os.chdir(Path(__file__).parent)
    
    # Start Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_chat.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])

def cleanup_processes(fastapi_process):
    """Clean up background processes."""
    if fastapi_process:
        print("\n🧹 Cleaning up FastAPI server...")
        fastapi_process.terminate()
        try:
            fastapi_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            fastapi_process.kill()

def main():
    """Main function to start the complete chat system."""
    print("🤖 RAG Chat System Startup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    fastapi_process = None
    
    try:
        # Start FastAPI server
        fastapi_process = start_fastapi_server()
        
        # Wait for API to be ready
        if not wait_for_api():
            print("❌ Could not start RAG API server")
            sys.exit(1)
        
        print("\n🎉 System ready!")
        print("📝 FastAPI server: http://localhost:8000")
        print("💬 Streamlit chat: http://localhost:8501")
        print("\n🚀 Starting Streamlit interface...")
        print("Press Ctrl+C to stop both servers")
        
        # Start Streamlit (this will block until user closes)
        start_streamlit()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down chat system...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cleanup_processes(fastapi_process)
        print("✅ Shutdown complete")

if __name__ == "__main__":
    main()