"""Main RAG pipeline orchestrating all components.

Provides a unified interface for document processing, indexing, retrieval, and generation.
Optimized for 612MB dataset with comprehensive error handling and monitoring.
"""

import logging
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple

from ..core.data_models import Document, Chunk, RAGResponse, RetrievalResult, ProcessingStats
from ..core.document_processor import DocumentProcessor
from ..storage.vector_store import VectorStoreManager
from ..core.retriever import HybridRetriever
from ..core.generator import ResponseGenerator
from ..config.settings import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline orchestrating all components.
    
    Features:
    - Document processing and chunking
    - Vector indexing with local embeddings
    - Hybrid retrieval (vector + BM25)
    - Response generation with confidence scoring
    - End-to-end error handling and monitoring
    """
    
    def __init__(self, 
                 collection_name: str = None,
                 vector_store_path: Path = None,
                 **component_kwargs):
        """Initialize RAG pipeline.
        
        Args:
            collection_name: Vector store collection name
            vector_store_path: Path for vector storage
            **component_kwargs: Additional arguments for components
        """
        self.collection_name = collection_name or settings.collection_name
        self.vector_store_path = vector_store_path or settings.vector_store_path
        
        # Initialize components
        self.document_processor = None
        self.vector_store = None
        self.retriever = None
        self.generator = None
        
        # Store component kwargs
        self.component_kwargs = component_kwargs
        
        # Pipeline state
        self.is_initialized = False
        self.indexed_chunks = []
        
        # Performance tracking
        self.stats = {
            'documents_processed': 0,
            'chunks_indexed': 0,
            'queries_processed': 0,
            'avg_query_time': 0.0,
            'total_processing_time': 0.0,
            'initialization_time': 0.0
        }
        
        logger.info(f"RAGPipeline created: collection={self.collection_name}")
    
    async def initialize(self, skip_generator: bool = False) -> None:
        """Initialize all pipeline components.
        
        Args:
            skip_generator: Skip generator initialization for ingestion-only tasks
        """
        if self.is_initialized:
            logger.info("Pipeline already initialized")
            return
        
        start_time = time.time()
        logger.info("Initializing RAG pipeline components...")
        
        try:
            # Initialize document processor
            self.document_processor = DocumentProcessor(
                **self.component_kwargs.get('document_processor', {})
            )
            
            # Initialize vector store
            self.vector_store = VectorStoreManager(
                collection_name=self.collection_name,
                persist_directory=self.vector_store_path,
                **self.component_kwargs.get('vector_store', {})
            )
            
            # Initialize response generator (optional for ingestion)
            if not skip_generator:
                try:
                    self.generator = ResponseGenerator(
                        **self.component_kwargs.get('generator', {})
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize generator: {e}")
                    logger.info("Generator initialization skipped - query functionality will be limited")
                    self.generator = None
            else:
                logger.info("Generator initialization skipped for ingestion-only mode")
                self.generator = None
            
            # Initialize retriever if documents exist in vector store
            try:
                vector_stats = self.vector_store.get_collection_stats()
                chunk_count = vector_stats.get('total_chunks', 0)
                
                if chunk_count > 0:
                    logger.info(f"Found {chunk_count} existing chunks, initializing retriever...")
                    
                    # Create retriever without chunks initially (it will load from vector store)
                    self.retriever = HybridRetriever(
                        vector_store=self.vector_store,
                        chunks=[],  # Empty chunks - retriever will use vector store directly
                        **self.component_kwargs.get('retriever', {})
                    )
                    logger.info("Retriever initialized with existing chunks")
                else:
                    logger.info("No existing chunks found - retriever will be initialized after document processing")
                    
            except Exception as e:
                logger.warning(f"Could not check for existing chunks: {e}")
            
            self.is_initialized = True
            initialization_time = time.time() - start_time
            self.stats['initialization_time'] = initialization_time
            
            logger.info(f"Pipeline initialized successfully in {initialization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    async def process_documents(self, 
                               documents_path: Union[str, Path],
                               recursive: bool = True,
                               force_reindex: bool = False) -> ProcessingStats:
        """Process and index documents.
        
        Args:
            documents_path: Path to documents directory
            recursive: Process subdirectories
            force_reindex: Re-index existing documents
            
        Returns:
            Processing statistics
        """
        await self.initialize()
        
        documents_path = Path(documents_path)
        logger.info(f"Processing documents from: {documents_path}")
        
        start_time = time.time()
        
        try:
            # Process documents
            documents = await self.document_processor.process_directory(
                documents_path,
                recursive=recursive
            )
            
            if not documents:
                logger.warning("No documents were processed")
                return self.document_processor.get_processing_stats()
            
            # Create chunks
            all_chunks = []
            for doc in documents:
                chunks = self.document_processor.create_chunks(
                    doc,
                    enable_contextual=True
                )
                all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
            
            # Index chunks in vector store
            self.vector_store.add_chunks(
                all_chunks,
                show_progress=True,
                overwrite=force_reindex
            )
            
            # Initialize retriever with chunks
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                chunks=all_chunks,
                **self.component_kwargs.get('retriever', {})
            )
            
            # Store chunks for future use
            self.indexed_chunks = all_chunks
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats['documents_processed'] += len(documents)
            self.stats['chunks_indexed'] += len(all_chunks)
            self.stats['total_processing_time'] += processing_time
            
            logger.info(f"Document processing complete: {len(documents)} docs, "
                       f"{len(all_chunks)} chunks in {processing_time:.2f}s")
            
            return self.document_processor.get_processing_stats()
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise
    
    async def query(self, 
                   question: str,
                   top_k: int = None,
                   retrieval_method: str = "hybrid",
                   template_type: str = "default",
                   filters: Dict[str, Any] = None) -> RAGResponse:
        """Query the RAG system.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            retrieval_method: "hybrid", "vector", or "bm25"
            template_type: Response template type
            filters: Metadata filters for retrieval
            
        Returns:
            RAG response with answer and sources
        """
        await self.initialize()
        
        if not self.retriever:
            raise ValueError("No documents have been indexed. Call process_documents() first.")
        
        start_time = time.time()
        
        try:
            # Retrieve relevant chunks
            retrieval_result = await self.retriever.retrieve(
                query=question,
                top_k=top_k or settings.retrieval_top_k,
                filters=filters,
                retrieval_method=retrieval_method
            )
            
            if not retrieval_result.chunks:
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    confidence_score=0.0,
                    query=question,
                    response_time=time.time() - start_time,
                    retrieval_result=retrieval_result
                )
            
            # Generate response
            response = await self.generator.generate(
                query=question,
                retrieval_result=retrieval_result,
                template_type=template_type
            )
            
            # Update stats
            query_time = time.time() - start_time
            self.stats['queries_processed'] += 1
            self.stats['avg_query_time'] = (
                (self.stats['avg_query_time'] * (self.stats['queries_processed'] - 1) + query_time) /
                self.stats['queries_processed']
            )
            
            logger.debug(f"Query processed in {query_time:.3f}s: "
                        f"confidence={response.confidence_score:.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{question}': {e}")
            return RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence_score=0.0,
                query=question,
                response_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    async def add_document(self, 
                          document_path: Union[str, Path],
                          force_reindex: bool = False) -> bool:
        """Add a single document to the pipeline.
        
        Args:
            document_path: Path to document file
            force_reindex: Re-index if already exists
            
        Returns:
            Success status
        """
        await self.initialize()
        
        document_path = Path(document_path)
        
        try:
            # Process single document
            document = await self.document_processor.process_file(document_path)
            
            # Create chunks
            chunks = self.document_processor.create_chunks(
                document,
                enable_contextual=True
            )
            
            # Add to vector store
            await self.vector_store.add_chunks(
                chunks,
                show_progress=False,
                force_reindex=force_reindex
            )
            
            # Update retriever if needed
            if self.retriever:
                # Reinitialize retriever with updated chunks
                self.indexed_chunks.extend(chunks)
                self.retriever = HybridRetriever(
                    vector_store=self.vector_store,
                    chunks=self.indexed_chunks,
                    **self.component_kwargs.get('retriever', {})
                )
            
            logger.info(f"Added document: {document_path} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {document_path}: {e}")
            return False
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = dict(self.stats)
        
        # Add component stats
        if self.document_processor:
            stats['document_processor'] = self.document_processor.get_processing_stats().__dict__
        
        if self.vector_store:
            stats['vector_store'] = self.vector_store.get_collection_stats()
        
        if self.retriever:
            stats['retriever'] = self.retriever.get_retrieval_stats()
        
        if self.generator:
            stats['generator'] = self.generator.get_generation_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            'status': 'healthy',
            'initialized': self.is_initialized,
            'components': {},
            'stats': self.get_pipeline_stats()
        }
        
        if not self.is_initialized:
            health['status'] = 'not_initialized'
            return health
        
        try:
            # Check each component
            if self.vector_store:
                vector_health = await self.vector_store.health_check()
                health['components']['vector_store'] = vector_health['status']
                if vector_health['status'] != 'healthy':
                    health['status'] = 'degraded'
            
            if self.retriever:
                retriever_health = await self.retriever.health_check()
                health['components']['retriever'] = retriever_health['status']
                if retriever_health['status'] != 'healthy':
                    health['status'] = 'degraded'
            
            if self.generator:
                generator_health = await self.generator.health_check()
                health['components']['generator'] = generator_health['status']
                if generator_health['status'] != 'healthy':
                    health['status'] = 'degraded'
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health
    
    async def reset(self) -> None:
        """Reset the pipeline and clear all data."""
        logger.warning("Resetting RAG pipeline - all data will be lost!")
        
        try:
            if self.vector_store:
                self.vector_store.reset_collection()
            
            self.retriever = None
            self.indexed_chunks = []
            
            # Reset stats
            self.stats = {
                'documents_processed': 0,
                'chunks_indexed': 0,
                'queries_processed': 0,
                'avg_query_time': 0.0,
                'total_processing_time': 0.0,
                'initialization_time': 0.0
            }
            
            logger.info("Pipeline reset complete")
            
        except Exception as e:
            logger.error(f"Error resetting pipeline: {e}")
            raise
    
    def get_indexed_documents_info(self) -> List[Dict[str, Any]]:
        """Get information about indexed documents."""
        # First check in-memory chunks
        if self.indexed_chunks:
            chunks_to_process = self.indexed_chunks
        else:
            # If no in-memory chunks, check vector store
            try:
                vector_stats = self.vector_store.get_collection_stats()
                if vector_stats.get('total_chunks', 0) > 0:
                    # We have chunks in vector store, return basic info
                    return [{
                        'doc_id': 'vector_store',
                        'filename': 'Vector Store Documents',
                        'file_path': '',
                        'chunk_count': vector_stats.get('total_chunks', 0),
                        'total_chars': 0  # Would need to query chunks to get this
                    }]
                else:
                    return []
            except Exception:
                return []
        
        # Group chunks by document  
        chunks_to_process = self.indexed_chunks
        docs_info = {}
        for chunk in chunks_to_process:
            doc_id = chunk.doc_id
            if doc_id not in docs_info:
                docs_info[doc_id] = {
                    'doc_id': doc_id,
                    'filename': chunk.metadata.get('filename', 'Unknown'),
                    'file_path': chunk.metadata.get('file_path', ''),
                    'chunk_count': 0,
                    'total_chars': 0
                }
            
            docs_info[doc_id]['chunk_count'] += 1
            docs_info[doc_id]['total_chars'] += len(chunk.content)
        
        return list(docs_info.values())
    
    async def search_similar_chunks(self, 
                                   query: str,
                                   top_k: int = 10,
                                   method: str = "hybrid") -> List[Tuple[Chunk, float]]:
        """Search for similar chunks (useful for debugging).
        
        Args:
            query: Search query
            top_k: Number of results
            method: Search method
            
        Returns:
            List of (chunk, score) tuples
        """
        if not self.retriever:
            raise ValueError("Pipeline not initialized or no documents indexed")
        
        retrieval_result = await self.retriever.retrieve(
            query=query,
            top_k=top_k,
            retrieval_method=method
        )
        
        return list(zip(retrieval_result.chunks, retrieval_result.scores))


# Convenience functions
async def create_rag_pipeline(documents_path: Union[str, Path] = None,
                            **kwargs) -> RAGPipeline:
    """Create and optionally initialize RAG pipeline.
    
    Args:
        documents_path: Path to documents to process immediately
        **kwargs: Additional arguments for pipeline components
        
    Returns:
        Initialized RAG pipeline
    """
    pipeline = RAGPipeline(**kwargs)
    await pipeline.initialize()
    
    if documents_path:
        await pipeline.process_documents(documents_path)
    
    return pipeline


async def quick_rag_query(question: str,
                         documents_path: Union[str, Path],
                         **kwargs) -> RAGResponse:
    """Quick RAG query for one-off usage.
    
    Args:
        question: Question to ask
        documents_path: Path to documents
        **kwargs: Additional pipeline arguments
        
    Returns:
        RAG response
    """
    pipeline = await create_rag_pipeline(documents_path, **kwargs)
    response = await pipeline.query(question)
    return response