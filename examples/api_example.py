#!/usr/bin/env python3
"""Example usage of RAG API server and client.

Demonstrates how to use the RAG system via API endpoints.
"""

import asyncio
import json
import time
from pathlib import Path

from rag_system.api.client import RAGClient, SyncRAGClient


async def async_example():
    """Async API client example."""
    print("🚀 RAG API Client Example (Async)")
    print("=" * 60)
    
    # Use async client
    async with RAGClient(base_url="http://localhost:8000") as client:
        try:
            # Check server health
            print("📊 Checking server health...")
            health = await client.health_check()
            print(f"   Status: {health['status']}")
            print(f"   Version: {health.get('version', 'Unknown')}")
            print()
            
            # Get system statistics
            print("📈 Getting system statistics...")
            stats = await client.get_stats()
            if 'vector_store' in stats:
                vs_stats = stats['vector_store']
                print(f"   Total chunks: {vs_stats.get('total_chunks', 0):,}")
                print(f"   Collection: {vs_stats.get('collection_name', 'Unknown')}")
                print(f"   Embedding model: {vs_stats.get('embedding_model', 'Unknown')}")
            print()
            
            # Example queries
            queries = [
                "What are the key retirement planning strategies?",
                "How does social security optimization work?",
                "What are the tax implications of different investment accounts?"
            ]
            
            for i, query in enumerate(queries, 1):
                print(f"🔍 Query {i}: {query}")
                print("-" * 40)
                
                try:
                    # Execute query
                    result = await client.query(
                        query=query,
                        k=5,
                        rerank=True,
                        template_type="citation"
                    )
                    
                    print(f"   Answer: {result['answer'][:200]}...")
                    print(f"   Confidence: {result['confidence_score']:.1%}")
                    print(f"   Response time: {result['response_time']:.3f}s")
                    print(f"   Sources: {len(result['sources'])}")
                    print()
                    
                except Exception as e:
                    print(f"   ❌ Query failed: {e}")
                    print()
            
            # Example search (without answer generation)
            print("🔎 Search example (retrieval only)...")
            search_result = await client.search(
                query="investment portfolio",
                k=10,
                search_type="hybrid"
            )
            
            print(f"   Found {search_result['total_results']} chunks")
            print(f"   Search time: {search_result['search_time']:.3f}s")
            print(f"   Top score: {max(search_result['scores']) if search_result['scores'] else 0:.3f}")
            print()
            
        except Exception as e:
            print(f"❌ API Error: {e}")


def sync_example():
    """Synchronous API client example."""
    print("🚀 RAG API Client Example (Sync)")
    print("=" * 60)
    
    # Use sync client wrapper
    client = SyncRAGClient(base_url="http://localhost:8000")
    
    try:
        # Check server health
        print("📊 Checking server health...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print()
        
        # Simple query
        print("🔍 Simple query example...")
        result = client.query(
            query="What is the most important factor in retirement planning?",
            k=3
        )
        
        print(f"   Answer: {result['answer'][:150]}...")
        print(f"   Confidence: {result['confidence_score']:.1%}")
        print()
        
    except Exception as e:
        print(f"❌ API Error: {e}")
    
    finally:
        client.close()


async def ingestion_example():
    """Example of document ingestion via API."""
    print("📚 Document Ingestion Example")
    print("=" * 60)
    
    async with RAGClient(base_url="http://localhost:8000") as client:
        try:
            # Check if we have documents to ingest
            docs_path = Path("data/SpecificationDocuments")
            if not docs_path.exists():
                print(f"❌ Document directory not found: {docs_path}")
                return
            
            print(f"📂 Starting ingestion from: {docs_path}")
            
            # Start ingestion
            ingestion = await client.ingest_documents(
                directory_path=docs_path,
                recursive=True,
                force_reindex=False
            )
            
            task_id = ingestion['task_id']
            print(f"   Task ID: {task_id}")
            print(f"   Message: {ingestion['message']}")
            print()
            
            # Monitor progress
            print("⏳ Monitoring ingestion progress...")
            start_time = time.time()
            
            while True:
                status = await client.get_ingestion_status(task_id)
                elapsed = time.time() - start_time
                
                print(f"   Status: {status['status']} (elapsed: {elapsed:.1f}s)")
                
                if status['status'] == 'completed':
                    result = status.get('result', {})
                    print(f"   ✅ Completed successfully!")
                    print(f"   Documents processed: {result.get('processed_documents', 0)}")
                    print(f"   Total chunks: {result.get('total_chunks', 0)}")
                    print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
                    break
                elif status['status'] == 'failed':
                    print(f"   ❌ Ingestion failed: {status.get('error', 'Unknown error')}")
                    break
                elif elapsed > 600:  # 10 minute timeout
                    print("   ⏰ Timeout - ingestion taking too long")
                    break
                
                # Wait before next check
                await asyncio.sleep(5)
                
        except Exception as e:
            print(f"❌ Ingestion Error: {e}")


def main():
    """Main example function."""
    print("🎯 RAG API Examples")
    print("=" * 60)
    print("Make sure the API server is running:")
    print("  python -m rag_system.api.cli serve")
    print("  or")
    print("  python rag_system/api/run_server.py")
    print()
    
    # Run examples
    try:
        # Async example
        asyncio.run(async_example())
        
        print("\n" + "=" * 60 + "\n")
        
        # Sync example
        sync_example()
        
        print("\n" + "=" * 60 + "\n")
        
        # Ingestion example (commented out by default)
        # asyncio.run(ingestion_example())
        
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    main()