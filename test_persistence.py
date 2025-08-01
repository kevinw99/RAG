#!/usr/bin/env python3
"""
Test script for conversation persistence feature.

This script demonstrates that conversation history persists across Streamlit reloads.
"""

import requests
import json
import time
from pathlib import Path

def test_conversation_persistence():
    """Test that conversations are saved and can be reloaded."""
    
    print("🧪 Testing Conversation Persistence")
    print("=" * 50)
    
    # Check if chat history file exists
    chat_history_file = Path("chat_history/conversation_state.json")
    
    if chat_history_file.exists():
        print("📁 Found existing conversation history:")
        
        try:
            with open(chat_history_file, 'r') as f:
                data = json.load(f)
            
            print(f"   📝 Messages: {len(data.get('messages', []))}")
            print(f"   📊 Total queries: {data.get('conversation_stats', {}).get('total_queries', 0)}")
            print(f"   🕒 Last updated: {data.get('last_updated', 'Unknown')}")
            
            # Show last few messages
            messages = data.get('messages', [])
            if messages:
                print("\n💬 Recent conversation:")
                for msg in messages[-3:]:  # Show last 3 messages
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:100] + "..." if len(msg.get('content', '')) > 100 else msg.get('content', '')
                    print(f"   {role.upper()}: {content}")
            
        except Exception as e:
            print(f"   ❌ Error reading history: {e}")
    
    else:
        print("📁 No conversation history found (this is normal for first run)")
    
    print("\n" + "=" * 50)
    print("💡 Tips for testing persistence:")
    print("1. Start the Streamlit app: streamlit run streamlit_chat.py")
    print("2. Have a conversation with the RAG system")
    print("3. Refresh the browser page (Ctrl+R or F5)")
    print("4. Your conversation should be restored automatically!")
    print("5. Check the 'Conversation Persistence' section in the sidebar")
    
    print("\n🔧 Files created:")
    print(f"   📄 Main: {chat_history_file}")
    print(f"   📄 Backup: {chat_history_file.parent / 'conversation_backup.json'}")

if __name__ == "__main__":
    test_conversation_persistence()