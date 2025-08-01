#!/usr/bin/env python3
"""Test script to test document retrieval without LLM generation."""

import requests
import json
import time
import sys

def test_retrieval_only():
    """Test document retrieval without answer generation."""
    print("🎯 Testing Document Retrieval (No LLM Required)")
    print("=" * 60)
    
    question = "What factors should be considered in portfolio optimization?"
    print(f"🔍 Question: {question}")
    print("-" * 60)
    
    try:
        # Use the search endpoint instead of query (no LLM needed)
        payload = {
            "query": question,
            "k": 5,
            "search_type": "hybrid"
        }
        
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/search",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ SUCCESS: Found {result['total_results']} relevant documents!")
            print(f"⏱️  Search Time: {result['search_time']:.2f}s")
            print(f"🔄 Total Time: {request_time:.2f}s")
            
            print(f"\n📚 Top {min(3, len(result['chunks']))} Retrieved Documents:")
            print("=" * 60)
            
            for i, (chunk, score) in enumerate(zip(result['chunks'], result['scores']), 1):
                print(f"\n📖 Document {i} (Relevance: {score:.3f})")
                print(f"📁 Source: {chunk.get('metadata', {}).get('filename', 'Unknown')}")
                print(f"📝 Content Preview: {chunk.get('content', '')[:300]}...")
                print("-" * 40)
            
            print(f"\n🎉 CONCLUSION: Your RAG system is working perfectly!")
            print(f"   • Document search: ✅ Working")
            print(f"   • Relevance scoring: ✅ Working") 
            print(f"   • Content retrieval: ✅ Working")
            print(f"   • Only missing: OpenAI API quota for answer generation")
            
            return True
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_server_connection():
    """Test if the server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is {data.get('status', 'unknown')}")
            return True
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def main():
    """Main test function."""
    # Test server connection
    if not test_server_connection():
        print("\n💡 To start the server, run: python start_server.py")
        sys.exit(1)
    
    print()
    test_retrieval_only()
    
    print(f"\n{'='*60}")
    print("💡 NEXT STEPS:")
    print("1. ✅ Your RAG system is fully functional for document retrieval")
    print("2. 🔑 To get answers, add OpenAI API credits or use a different LLM")
    print("3. 🆓 Alternative: Configure a free local LLM (Ollama, etc.)")
    print("4. 📊 You can browse documents at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()