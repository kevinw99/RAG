#!/usr/bin/env python3
"""Test script to ask questions to the RAG API server."""

import requests
import json
import time
import sys

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

def ask_question(question, k=5, rerank=True, template_type="default"):
    """Ask a question to the RAG system."""
    print(f"\n🔍 Question: {question}")
    print("-" * 60)
    
    try:
        # Prepare request
        payload = {
            "query": question,
            "k": k,
            "rerank": rerank,
            "template_type": template_type
        }
        
        # Send request
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/query",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"📝 Answer: {result['answer'][:400]}{'...' if len(result['answer']) > 400 else ''}")
            print(f"🎯 Confidence: {result['confidence_score']:.1%}")
            print(f"⏱️  Total Time: {request_time:.2f}s")
            print(f"🔄 Response Time: {result['response_time']:.2f}s")
            print(f"📚 Sources: {len(result['sources'])} documents")
            
            if result['sources']:
                print("\n📖 Top Sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    filename = source.get('filename', 'Unknown')
                    print(f"   {i}. {filename}")
            
            return True
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Main test function."""
    print("🎯 RAG System Question Tester")
    print("=" * 60)
    
    # Test server connection
    if not test_server_connection():
        print("\n💡 To start the server, run:")
        print("   python start_server.py")
        print("   # or")
        print("   python -m rag_system.api.cli serve --port 8000")
        sys.exit(1)
    
    # Test questions
    questions = [
        "What factors should be considered in portfolio optimization?"
    ]
    
    print(f"\n🚀 Testing with {len(questions)} questions...")
    
    successful = 0
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(questions)}")
        
        if ask_question(question, k=3, rerank=False):  # Disable rerank for faster response
            successful += 1
        
        # Brief pause between questions
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"📊 Results: {successful}/{len(questions)} questions answered successfully")
    
    if successful == len(questions):
        print("🎉 All tests passed! Your RAG system is working perfectly!")
    elif successful > 0:
        print("⚠️  Some tests passed. The system is partially working.")
    else:
        print("❌ No tests passed. Check server logs for issues.")

if __name__ == "__main__":
    main()