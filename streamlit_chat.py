#!/usr/bin/env python3
"""
Streamlit ChatGPT-like Interface for RAG System

A modern chat interface that connects to your FastAPI RAG backend,
providing a seamless conversational experience for querying documents.
"""

import streamlit as st
import requests
import json
import time
import pickle
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        color: #333;
        padding: 0.75rem 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border-left: 4px solid #28a745;
    }
    
    .source-item {
        background-color: #e9ecef;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        border-left: 3px solid #17a2b8;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    
    .stats-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"
DEFAULT_K = 5
DEFAULT_RERANK = True
DEFAULT_TEMPLATE = "default"

# Persistence configuration
CHAT_HISTORY_DIR = Path("chat_history")
CHAT_HISTORY_FILE = CHAT_HISTORY_DIR / "conversation_state.json"
BACKUP_HISTORY_FILE = CHAT_HISTORY_DIR / "conversation_backup.json"

class RAGChatInterface:
    """Main chat interface class for RAG system."""
    
    def __init__(self):
        """Initialize the chat interface."""
        self.api_url = f"{API_BASE_URL}/query"
        self.health_url = f"{API_BASE_URL}/health"
        
        # Ensure chat history directory exists
        CHAT_HISTORY_DIR.mkdir(exist_ok=True)
        
        # Initialize session state with persistence
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "conversation_stats" not in st.session_state:
            st.session_state.conversation_stats = {
                "total_queries": 0,
                "total_response_time": 0.0,
                "avg_confidence": 0.0,
                "sources_used": set()
            }
        if "conversation_loaded" not in st.session_state:
            st.session_state.conversation_loaded = False
        
        # Load conversation history on first run
        if not st.session_state.conversation_loaded:
            self.load_conversation_history()
            st.session_state.conversation_loaded = True
    
    def save_conversation_history(self) -> None:
        """Save conversation history and stats to disk."""
        try:
            # Create backup of existing file
            if CHAT_HISTORY_FILE.exists():
                import shutil
                shutil.copy2(CHAT_HISTORY_FILE, BACKUP_HISTORY_FILE)
            
            # Prepare data for serialization
            conversation_data = {
                "messages": st.session_state.messages,
                "conversation_stats": {
                    "total_queries": st.session_state.conversation_stats["total_queries"],
                    "total_response_time": st.session_state.conversation_stats["total_response_time"],
                    "avg_confidence": st.session_state.conversation_stats["avg_confidence"],
                    "sources_used": list(st.session_state.conversation_stats["sources_used"])  # Convert set to list
                },
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            # Save to file
            with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            st.error(f"⚠️ Failed to save conversation history: {e}")
    
    def load_conversation_history(self) -> None:
        """Load conversation history and stats from disk."""
        try:
            if CHAT_HISTORY_FILE.exists():
                with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    conversation_data = json.load(f)
                
                # Load messages
                if "messages" in conversation_data:
                    st.session_state.messages = conversation_data["messages"]
                
                # Load stats
                if "conversation_stats" in conversation_data:
                    stats = conversation_data["conversation_stats"]
                    st.session_state.conversation_stats = {
                        "total_queries": stats.get("total_queries", 0),
                        "total_response_time": stats.get("total_response_time", 0.0),
                        "avg_confidence": stats.get("avg_confidence", 0.0),
                        "sources_used": set(stats.get("sources_used", []))  # Convert list back to set
                    }
                
                # Show success message (only once per session)
                if len(st.session_state.messages) > 0:
                    # Use a more subtle notification that doesn't persist
                    if "load_notification_shown" not in st.session_state:
                        st.session_state.load_notification_shown = True
                        st.toast(f"✅ Loaded {len(st.session_state.messages)} messages from previous conversation", icon="💬")
                    
        except Exception as e:
            st.warning(f"⚠️ Could not load conversation history: {e}")
            # Try to load from backup
            self.load_from_backup()
    
    def load_from_backup(self) -> None:
        """Load conversation history from backup file."""
        try:
            if BACKUP_HISTORY_FILE.exists():
                with open(BACKUP_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    conversation_data = json.load(f)
                
                if "messages" in conversation_data:
                    st.session_state.messages = conversation_data["messages"]
                    st.info("📦 Loaded conversation from backup file")
                    
        except Exception as e:
            st.warning(f"⚠️ Could not load from backup: {e}")
    
    def auto_save_conversation(self) -> None:
        """Automatically save conversation after each interaction."""
        self.save_conversation_history()
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history both from memory and disk."""
        try:
            # Clear session state
            st.session_state.messages = []
            st.session_state.conversation_stats = {
                "total_queries": 0,
                "total_response_time": 0.0,
                "avg_confidence": 0.0,
                "sources_used": set()
            }
            
            # Remove saved files
            if CHAT_HISTORY_FILE.exists():
                CHAT_HISTORY_FILE.unlink()
            if BACKUP_HISTORY_FILE.exists():
                BACKUP_HISTORY_FILE.unlink()
                
            st.success("🗑️ Conversation history cleared completely")
            
        except Exception as e:
            st.error(f"❌ Failed to clear conversation history: {e}")

    def check_api_health(self) -> bool:
        """Check if the RAG API is healthy and accessible."""
        try:
            response = requests.get(self.health_url, timeout=30)  # Increased timeout
            return response.status_code == 200
        except requests.exceptions.Timeout:
            st.warning("⏳ RAG API is slow to respond. Server may be initializing...")
            st.info("💡 This can happen on first startup while loading models. Please wait and refresh.")
            return False
        except Exception as e:
            st.error(f"❌ Cannot connect to RAG API: {e}")
            st.info("💡 Make sure your FastAPI server is running: `python start_server.py`")
            return False
    
    def query_rag_system(self, 
                        query: str, 
                        k: int = DEFAULT_K, 
                        rerank: bool = DEFAULT_RERANK,
                        template_type: str = DEFAULT_TEMPLATE) -> Optional[Dict[str, Any]]:
        """Send query to RAG system and return response."""
        try:
            payload = {
                "query": query,
                "k": k,
                "rerank": rerank,
                "template_type": template_type
            }
            
            with st.spinner("🤔 Thinking..."):
                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=120  # Increased to 2 minutes for complex queries
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Query timed out after 2 minutes")
            st.info("💡 This can happen with:")
            st.info("- Complex queries requiring multiple reasoning steps")
            st.info("- Large document retrieval and processing")  
            st.info("- First query after server restart (model loading)")
            st.info("🔄 Try a simpler query or wait for server to fully initialize")
            return None
        except Exception as e:
            st.error(f"❌ Error querying RAG system: {e}")
            return None
    
    def format_confidence_score(self, confidence: float) -> str:
        """Format confidence score with appropriate styling."""
        percentage = confidence * 100
        
        if confidence >= 0.8:
            return f'<span class="confidence-high">🟢 {percentage:.1f}% (High)</span>'
        elif confidence >= 0.5:
            return f'<span class="confidence-medium">🟡 {percentage:.1f}% (Medium)</span>'
        else:
            return f'<span class="confidence-low">🔴 {percentage:.1f}% (Low)</span>'
    
    def display_sources(self, sources: List[Dict[str, Any]]) -> None:
        """Display source documents in an expandable section."""
        if not sources:
            return
        
        with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
            for i, source in enumerate(sources, 1):
                filename = source.get('filename', 'Unknown')
                chunk_id = source.get('chunk_id', 'N/A')
                start_char = source.get('start_char', 0)
                end_char = source.get('end_char', 0)
                
                st.markdown(f"""
                <div class="source-item">
                    <strong>📄 {i}. {filename}</strong><br>
                    <small>Chunk: {chunk_id} | Characters: {start_char}-{end_char}</small>
                </div>
                """, unsafe_allow_html=True)
    
    def update_conversation_stats(self, response_data: Dict[str, Any]) -> None:
        """Update conversation statistics."""
        stats = st.session_state.conversation_stats
        
        stats["total_queries"] += 1
        stats["total_response_time"] += response_data.get("response_time", 0)
        
        # Update average confidence
        current_confidence = response_data.get("confidence_score", 0)
        total_confidence = stats["avg_confidence"] * (stats["total_queries"] - 1) + current_confidence
        stats["avg_confidence"] = total_confidence / stats["total_queries"]
        
        # Track unique sources
        for source in response_data.get("sources", []):
            filename = source.get("filename")
            if filename:
                stats["sources_used"].add(filename)
    
    def display_conversation_stats(self) -> None:
        """Display conversation statistics in sidebar."""
        stats = st.session_state.conversation_stats
        
        if stats["total_queries"] > 0:
            avg_response_time = stats["total_response_time"] / stats["total_queries"]
            
            st.markdown("### 📊 Conversation Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Queries", stats["total_queries"])
                st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
            
            with col2:
                st.metric("Avg Confidence", f"{stats['avg_confidence']*100:.1f}%")
                st.metric("Sources Used", len(stats["sources_used"]))
            
            if len(stats["sources_used"]) > 0:
                with st.expander("📚 Documents Referenced"):
                    for filename in sorted(stats["sources_used"]):
                        st.write(f"• {filename}")
    
    def export_conversation(self) -> None:
        """Export conversation history as JSON."""
        if st.session_state.messages:
            conversation_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "stats": {
                    "total_queries": st.session_state.conversation_stats["total_queries"],
                    "avg_response_time": st.session_state.conversation_stats["total_response_time"] / max(1, st.session_state.conversation_stats["total_queries"]),
                    "avg_confidence": st.session_state.conversation_stats["avg_confidence"],
                    "unique_sources": len(st.session_state.conversation_stats["sources_used"])
                }
            }
            
            json_str = json.dumps(conversation_data, indent=2)
            
            st.download_button(
                label="💾 Export Conversation",
                data=json_str,
                file_name=f"rag_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def clear_conversation(self) -> None:
        """Clear conversation history and stats."""
        if st.button("🗑️ Clear Conversation"):
            self.clear_conversation_history()
            st.rerun()
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar with settings and stats."""
        st.sidebar.markdown("### ⚙️ Query Settings")
        
        # Query parameters
        k = st.sidebar.slider(
            "📊 Number of sources (k)",
            min_value=1,
            max_value=20,
            value=DEFAULT_K,
            help="Number of document chunks to retrieve"
        )
        
        rerank = st.sidebar.checkbox(
            "🔄 Enable reranking",
            value=DEFAULT_RERANK,
            help="Use cross-encoder reranking for better results"
        )
        
        template_type = st.sidebar.selectbox(
            "📝 Response template",
            options=["default", "citation", "summary", "comparison"],
            index=0,
            help="Choose response formatting style"
        )
        
        st.sidebar.markdown("---")
        
        # Display stats
        self.display_conversation_stats()
        
        st.sidebar.markdown("---")
        
        # Persistence status
        st.sidebar.markdown("### 💾 Conversation Persistence")
        
        # Show persistence status
        if CHAT_HISTORY_FILE.exists():
            file_size = CHAT_HISTORY_FILE.stat().st_size
            last_modified = datetime.fromtimestamp(CHAT_HISTORY_FILE.stat().st_mtime)
            st.sidebar.success(f"✅ Auto-saved ({file_size} bytes)")
            st.sidebar.caption(f"Last saved: {last_modified.strftime('%H:%M:%S')}")
        else:
            st.sidebar.info("💾 No saved conversation yet")
        
        # Manual save button
        if st.sidebar.button("💾 Save Now"):
            self.auto_save_conversation()
            st.sidebar.success("✅ Conversation saved!")
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Export and clear buttons
        self.export_conversation()
        self.clear_conversation()
        
        return {
            "k": k,
            "rerank": rerank,
            "template_type": template_type
        }
    
    def render_main_interface(self, query_params: Dict[str, Any]) -> None:
        """Render the main chat interface."""
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🤖 RAG Chat Assistant</h1>
            <p>Ask questions about your documents and get intelligent answers with sources</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display conversation history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        st.markdown(message["content"])
                    else:
                        # Assistant message with metadata
                        st.markdown(message["content"])
                        
                        # Display confidence score
                        if "confidence_score" in message:
                            confidence_html = self.format_confidence_score(message["confidence_score"])
                            st.markdown(
                                f'**Confidence:** {confidence_html}',
                                unsafe_allow_html=True
                            )
                        
                        # Display response time
                        if "response_time" in message:
                            st.caption(f"⏱️ Response time: {message['response_time']:.2f}s")
                        
                        # Display sources
                        if "sources" in message:
                            self.display_sources(message["sources"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Auto-save after user input
            self.auto_save_conversation()
            
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Query RAG system
            response_data = self.query_rag_system(
                query=prompt,
                k=query_params["k"],
                rerank=query_params["rerank"],
                template_type=query_params["template_type"]
            )
            
            if response_data:
                # Add assistant response to chat history
                assistant_message = {
                    "role": "assistant",
                    "content": response_data["answer"],
                    "confidence_score": response_data.get("confidence_score", 0),
                    "response_time": response_data.get("response_time", 0),
                    "sources": response_data.get("sources", []),
                    "timestamp": datetime.now().isoformat()
                }
                
                st.session_state.messages.append(assistant_message)
                
                # Update conversation stats
                self.update_conversation_stats(response_data)
                
                # Auto-save conversation after each interaction
                self.auto_save_conversation()
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response_data["answer"])
                    
                    # Display confidence score
                    confidence_html = self.format_confidence_score(response_data["confidence_score"])
                    st.markdown(
                        f'**Confidence:** {confidence_html}',
                        unsafe_allow_html=True
                    )
                    
                    # Display response time
                    st.caption(f"⏱️ Response time: {response_data['response_time']:.2f}s")
                    
                    # Display sources
                    self.display_sources(response_data["sources"])
                
                # Rerun to update the display
                st.rerun()
    
    def run(self) -> None:
        """Main application entry point."""
        # Check API health first
        if not self.check_api_health():
            st.stop()
        
        # Render sidebar and get query parameters
        query_params = self.render_sidebar()
        
        # Render main chat interface
        self.render_main_interface(query_params)


def main():
    """Main function to run the Streamlit app."""
    try:
        # Create and run the chat interface
        chat_interface = RAGChatInterface()
        chat_interface.run()
        
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.info("Please check the console for more details and ensure your RAG API is running.")


if __name__ == "__main__":
    main()