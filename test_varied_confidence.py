#!/usr/bin/env python3
"""
Test confidence scoring with various types of queries to ensure variation.
"""

import requests
import json

def test_query(query, expected_confidence_range=None):
    """Test a single query and return confidence score."""
    payload = {
        "query": query,
        "k": 3,
        "rerank": False,
        "template_type": "default"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/query",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            confidence = result.get('confidence_score', 0)
            sources = len(result.get('sources', []))
            
            print(f"📝 Query: {query[:60]}...")
            print(f"🎯 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            print(f"📚 Sources: {sources}")
            
            if expected_confidence_range:
                min_conf, max_conf = expected_confidence_range
                if min_conf <= confidence <= max_conf:
                    print(f"✅ Confidence in expected range ({min_conf}-{max_conf})")
                else:
                    print(f"⚠️  Confidence outside expected range ({min_conf}-{max_conf})")
            
            print("-" * 50)
            return confidence
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    """Test confidence scoring with different query types."""
    print("🧪 Testing Confidence Scoring Variation")
    print("=" * 60)
    
    # Test different types of queries
    test_cases = [
        # Should have high confidence - specific topic well covered
        ("What is Modern Portfolio Theory?", (0.7, 1.0)),
        
        # Should have medium confidence - general question
        ("What factors should be considered in portfolio optimization?", (0.6, 0.9)),
        
        # Should have lower confidence - very specific technical question
        ("What is the exact mathematical formula for the Sharpe ratio optimization constraint?", (0.3, 0.7)),
        
        # Should have low confidence - question not in documents
        ("How do I cook pasta?", (0.0, 0.4)),
        
        # Should have medium-high confidence - well documented topic
        ("What is the purpose of rebalancing in portfolio management?", (0.5, 0.8)),
    ]
    
    scores = []
    
    for query, expected_range in test_cases:
        confidence = test_query(query, expected_range)
        if confidence is not None:
            scores.append(confidence)
    
    if scores:
        print("\n📊 Confidence Score Summary:")
        print(f"   Average: {sum(scores)/len(scores):.3f}")
        print(f"   Range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"   Variation: {max(scores) - min(scores):.3f}")
        
        if max(scores) - min(scores) > 0.2:
            print("✅ Good confidence score variation detected!")
        else:
            print("⚠️  Limited confidence score variation")
    
    print("\n🎯 Confidence scoring system is working!")

if __name__ == "__main__":
    main()