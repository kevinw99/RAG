#!/usr/bin/env python3
"""
Example usage of the RAG system.

This script demonstrates the core functionality:
1. Document processing and indexing
2. Querying with different methods
3. Confidence scoring and source attribution

Usage:
    python example_usage.py
"""

import asyncio
import logging
from pathlib import Path
from rag_system import RAGPipeline, create_rag_pipeline, quick_rag_query

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_basic_usage():
    """Demonstrate basic RAG pipeline usage."""
    print("🚀 RAG System Demo - Basic Usage")
    print("=" * 50)
    
    try:
        # Create pipeline
        pipeline = RAGPipeline(collection_name="demo_collection")
        
        # Initialize (loads models, sets up vector store)
        print("📝 Initializing RAG pipeline...")
        await pipeline.initialize()
        
        # Create some demo documents
        demo_docs_dir = Path("demo_documents")
        demo_docs_dir.mkdir(exist_ok=True)
        
        # Create sample documents
        doc1_content = """
        # Machine Learning Overview
        
        Machine learning is a subset of artificial intelligence (AI) that focuses on 
        algorithms that can learn from and make predictions or decisions based on data.
        
        ## Types of Machine Learning
        1. Supervised Learning - learns from labeled data
        2. Unsupervised Learning - finds patterns in unlabeled data  
        3. Reinforcement Learning - learns through interaction with environment
        
        ## Key Applications
        - Image recognition and computer vision
        - Natural language processing
        - Recommendation systems
        - Autonomous vehicles
        """
        
        doc2_content = """
        # Vector Databases and RAG
        
        Vector databases are specialized databases designed to store and query 
        high-dimensional vector data efficiently.
        
        ## Popular Vector Databases
        - ChromaDB: Open-source, easy to use
        - Pinecone: Cloud-native vector database
        - Weaviate: Open-source with GraphQL API
        - Qdrant: High-performance vector search engine
        
        ## RAG Applications
        Vector databases are crucial for Retrieval-Augmented Generation (RAG):
        1. Store document embeddings
        2. Enable semantic search
        3. Retrieve relevant context
        4. Augment LLM responses
        """
        
        # Write demo documents
        (demo_docs_dir / "ml_overview.md").write_text(doc1_content)
        (demo_docs_dir / "vector_databases.md").write_text(doc2_content)
        
        # Process documents
        print(f"📚 Processing documents from {demo_docs_dir}...")
        stats = await pipeline.process_documents(demo_docs_dir)
        
        print(f"✅ Processed {stats.processed_documents} documents")
        print(f"✅ Created {stats.total_chunks} chunks")
        print(f"✅ Success rate: {stats.success_rate:.1%}")
        
        # Test different query types
        queries = [
            "What is machine learning?",
            "How do vector databases work with RAG?",
            "What are the types of machine learning?",
            "Which vector databases are mentioned?"
        ]
        
        print("\n🔍 Testing Queries")
        print("-" * 30)
        
        for i, question in enumerate(queries, 1):
            print(f"\n{i}. Question: {question}")
            
            # Query with hybrid retrieval
            response = await pipeline.query(
                question=question,
                top_k=3,
                retrieval_method="hybrid"
            )
            
            print(f"   Answer: {response.answer[:100]}...")
            print(f"   Confidence: {response.confidence_score:.1%}")
            print(f"   Sources: {len(response.sources)}")
            print(f"   Response time: {response.response_time:.3f}s")
        
        # Show pipeline statistics
        print("\n📊 Pipeline Statistics")
        print("-" * 30)
        stats = pipeline.get_pipeline_stats()
        print(f"Documents processed: {stats.get('documents_processed', 0)}")
        print(f"Chunks indexed: {stats.get('chunks_indexed', 0)}")
        print(f"Queries processed: {stats.get('queries_processed', 0)}")
        print(f"Average query time: {stats.get('avg_query_time', 0):.3f}s")
        
        # Health check
        print("\n🏥 System Health Check")
        print("-" * 30)
        health = await pipeline.health_check()
        print(f"Status: {health['status']}")
        
        if 'components' in health:
            for component, status in health['components'].items():
                print(f"{component}: {status}")
        
        print("\n✅ Demo completed successfully!")
        
        # Cleanup
        import shutil
        shutil.rmtree(demo_docs_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        logger.exception("Demo failed")


async def demo_quick_query():
    """Demonstrate quick query functionality."""
    print("\n🚀 RAG System Demo - Quick Query")
    print("=" * 50)
    
    # Create demo document
    demo_docs_dir = Path("quick_demo")
    demo_docs_dir.mkdir(exist_ok=True)
    
    doc_content = """
    # RAG System Features
    
    Our RAG system includes several advanced features:
    
    ## Hybrid Retrieval
    - Combines vector search and keyword search (BM25)
    - Uses reciprocal rank fusion for optimal results
    - Supports different retrieval methods
    
    ## Local Embeddings
    - Uses sentence-transformers for zero API costs
    - Optimized for 612MB document libraries
    - Efficient batch processing
    
    ## Confidence Scoring
    - Multi-factor confidence assessment
    - Context relevance analysis
    - Response quality validation
    """
    
    (demo_docs_dir / "features.md").write_text(doc_content)
    
    try:
        # One-line RAG query
        print("📝 Running quick RAG query...")
        response = await quick_rag_query(
            question="What are the main features of the RAG system?",
            documents_path=demo_docs_dir
        )
        
        print(f"✅ Quick query completed!")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence_score:.1%}")
        print(f"Sources: {len(response.sources)}")
        
    except Exception as e:
        print(f"❌ Error in quick demo: {e}")
        logger.exception("Quick demo failed")
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(demo_docs_dir, ignore_errors=True)


async def main():
    """Main demo function."""
    print("🎯 RAG System Demonstration")
    print("This demo shows the key capabilities of the RAG system")
    print("=" * 60)
    
    # Run demos
    await demo_basic_usage()
    await demo_quick_query()
    
    print("\n🎉 All demos completed!")
    print("\nNext steps:")
    print("1. Install: pip install -e .")
    print("2. Set up environment: Add LLM API keys to .env")
    print("3. Try CLI: rag ingest your_documents/")
    print("4. Query: rag query 'your question here'")


if __name__ == "__main__":
    asyncio.run(main())