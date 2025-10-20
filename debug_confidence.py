#!/usr/bin/env python3
"""
Debug script to test confidence scoring system.
"""

import requests
import json

def test_confidence_scoring():
    """Test the confidence scoring system by making API calls."""
    
    print("🔍 Testing Confidence Scoring System")
    print("=" * 50)
    
    # Test query
    query = "What is portfolio optimization?"
    
    payload = {
        "query": query,
        "k": 3,
        "rerank": False,
        "template_type": "default"
    }
    
    print(f"📝 Query: {query}")
    print(f"⚙️ Parameters: k={payload['k']}, rerank={payload['rerank']}")
    
    try:
        response = requests.post(
            "http://localhost:8000/query",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ Response received:")
            print(f"🎯 Confidence Score: {result.get('confidence_score', 'N/A')}")
            print(f"📊 Sources: {len(result.get('sources', []))}")
            print(f"⏱️  Response Time: {result.get('response_time', 'N/A'):.2f}s")
            
            # Check retrieval stats
            if 'retrieval_stats' in result:
                print(f"🔍 Retrieval Stats: {result['retrieval_stats']}")
            
            # Show first part of answer
            answer = result.get('answer', '')
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"📖 Answer preview: {preview}")
            
            # Analyze the confidence score
            confidence = result.get('confidence_score', 0)
            if confidence == 0.5:
                print("\n⚠️  WARNING: Confidence score is exactly 0.5")
                print("   This suggests the default fallback value is being used")
                print("   The confidence scorer may be catching an exception")
            elif confidence == 0.0:
                print("\n⚠️  WARNING: Confidence score is 0.0")
                print("   This suggests empty response or chunks")
            else:
                print(f"\n✅ Confidence score looks good: {confidence}")
            
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_health():
    """Test API health to make sure server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is healthy")
            return True
        else:
            print(f"⚠️  API server returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API server: {e}")
        return False

if __name__ == "__main__":
    if test_health():
        test_confidence_scoring()
    else:
        print("💡 Make sure the FastAPI server is running: python start_server.py")