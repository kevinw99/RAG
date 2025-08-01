#!/usr/bin/env python3
"""Test script specifically for Anthropic Claude API."""

import requests
import json
import os
import time

def check_anthropic_config():
    """Check if Anthropic is properly configured."""
    print("🔍 Checking Anthropic Configuration...")
    
    # Check environment variables
    api_key = os.getenv('ANTHROPIC_API_KEY')
    provider = os.getenv('LLM_PROVIDER', 'openai')
    model = os.getenv('LLM_MODEL', 'claude-3-haiku-20240307')
    
    print(f"   API Key: {'✅ Set' if api_key else '❌ Missing'}")
    print(f"   Provider: {provider}")
    print(f"   Model: {model}")
    
    if not api_key:
        print("\n❌ Missing ANTHROPIC_API_KEY!")
        print("💡 Set it up:")
        print("   1. Get your API key from: https://console.anthropic.com/")
        print("   2. Edit the .env file and add your key")
        print("   3. Or run: export ANTHROPIC_API_KEY='your-key-here'")
        return False
    
    if provider != 'anthropic':
        print(f"\n⚠️  LLM_PROVIDER is '{provider}', should be 'anthropic'")
        print("💡 Fix: export LLM_PROVIDER=anthropic")
    
    return True

def test_anthropic_query():
    """Test a query using Anthropic Claude."""
    print("\n🎯 Testing RAG System with Anthropic Claude")
    print("=" * 60)
    
    question = "What factors should be considered in portfolio optimization?"
    print(f"🔍 Question: {question}")
    print("-" * 60)
    
    try:
        payload = {
            "query": question,
            "k": 3,
            "rerank": False,  # Faster without reranking
            "template_type": "default"
        }
        
        print("🔄 Sending request to Claude...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/query",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=45  # Claude might take a bit longer
        )
        
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print("🎉 SUCCESS! Claude Generated an Answer!")
            print("=" * 60)
            print(f"📝 Answer: {result['answer']}")
            print("=" * 60)
            print(f"🎯 Confidence: {result['confidence_score']:.1%}")
            print(f"⏱️  Total Time: {request_time:.2f}s")
            print(f"🔄 Processing Time: {result['response_time']:.2f}s")
            print(f"📚 Sources: {len(result['sources'])} documents")
            
            if result['sources']:
                print(f"\n📖 Source Documents:")
                for i, source in enumerate(result['sources'][:3], 1):
                    filename = source.get('filename', 'Unknown')
                    print(f"   {i}. {filename}")
            
            return True
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            error_detail = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            print(f"Error: {error_detail}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏱️  Request timed out - Claude might be processing...")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Main test function."""
    print("🤖 RAG System + Anthropic Claude Test")
    print("=" * 60)
    
    # Check configuration
    if not check_anthropic_config():
        return
    
    # Check server
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ RAG server is running")
        else:
            print("⚠️  RAG server issue")
            return
    except:
        print("❌ RAG server not running")
        print("💡 Start it: python start_server.py")
        return
    
    # Test query
    success = test_anthropic_query()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 CONGRATULATIONS!")
        print("   Your RAG system is working perfectly with Claude!")
        print("   You can now ask any questions about your documents.")
    else:
        print("💡 TROUBLESHOOTING:")
        print("   1. Make sure ANTHROPIC_API_KEY is set correctly")
        print("   2. Check your Anthropic account has credits")
        print("   3. Verify LLM_PROVIDER=anthropic")

if __name__ == "__main__":
    main()