"""
RAG System - Production-ready Retrieval-Augmented Generation system.

Optimized for 612MB document library (1,566 documents) with:
- Local embeddings (zero ongoing API costs)
- Hybrid retrieval (vector + keyword search)
- Comprehensive evaluation with RAGAs
- <1s response time, <8GB RAM usage
"""

__version__ = "0.1.0"
__author__ = "RAG System"

# Core exports for external use
from .core.pipeline import RAGPipeline, create_rag_pipeline, quick_rag_query
from .core.document_processor import DocumentProcessor  
from .core.retriever import HybridRetriever
from .core.generator import ResponseGenerator
from .storage.vector_store import VectorStoreManager
from .config.settings import Settings

__all__ = [
    "RAGPipeline",
    "create_rag_pipeline",
    "quick_rag_query",
    "DocumentProcessor", 
    "HybridRetriever",
    "ResponseGenerator",
    "VectorStoreManager",
    "Settings",
    "__version__",
]