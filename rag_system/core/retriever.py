"""Hybrid retrieval system combining vector and BM25 search with reranking.

Implements state-of-the-art retrieval patterns:
- Semantic search with local embeddings
- Keyword search with BM25
- Reciprocal Rank Fusion for result combination
- Cross-encoder reranking for optimal relevance
- Optimized for 612MB dataset (400K-600K chunks)
"""

import logging
import time
import asyncio
from typing import List, Dict, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from ..core.data_models import Chunk, RetrievalResult
from ..storage.vector_store import VectorStoreManager
from ..config.settings import settings

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class BM25Retriever:
    """BM25-based keyword retrieval optimized for large datasets."""
    
    def __init__(self, chunks: List[Chunk]):
        """Initialize BM25 retriever with document corpus.
        
        Args:
            chunks: List of chunks to index
        """
        self.chunks = chunks
        self.chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Prepare tokenized corpus
        logger.info(f"Building BM25 index for {len(chunks)} chunks")
        start_time = time.time()
        
        # Tokenize documents
        self.tokenized_corpus = [
            self._tokenize_text(chunk.content) 
            for chunk in chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        build_time = time.time() - start_time
        logger.info(f"BM25 index built in {build_time:.2f}s")
        
        # Cache for stopwords
        self.stop_words = set(stopwords.words('english'))
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25 processing."""
        if not text:
            return []
        
        try:
            # Tokenize and convert to lowercase
            tokens = word_tokenize(text.lower())
            
            # Filter out stopwords and non-alphabetic tokens
            tokens = [
                token for token in tokens 
                if token.isalpha() and token not in self.stop_words
            ]
            
            return tokens
            
        except Exception as e:
            logger.warning(f"Error tokenizing text: {e}")
            return text.lower().split()
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[Chunk, float]]:
        """Search using BM25 algorithm.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        if not query.strip():
            return []
        
        try:
            # Tokenize query
            query_tokens = self._tokenize_text(query)
            
            if not query_tokens:
                return []
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.chunks) and scores[idx] > 0:
                    chunk = self.chunks[idx]
                    score = float(scores[idx])
                    results.append((chunk, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get BM25 retriever statistics."""
        return {
            'total_chunks': len(self.chunks),
            'vocab_size': len(self.bm25.idf) if hasattr(self.bm25, 'idf') else 0,
            'avg_doc_length': np.mean([len(doc) for doc in self.tokenized_corpus]),
        }


class CrossEncoderReranker:
    """Cross-encoder reranker for improving retrieval quality."""
    
    def __init__(self, model_name: str = None):
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name
        """
        self.model_name = model_name or settings.reranker_model
        
        logger.info(f"Loading cross-encoder model: {self.model_name}")
        try:
            self.model = CrossEncoder(self.model_name, device=settings.device)
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            self.model = None
    
    async def rerank(self, query: str, 
                    chunks_with_scores: List[Tuple[Chunk, float]],
                    top_k: int = None) -> List[Tuple[Chunk, float]]:
        """Rerank chunks using cross-encoder.
        
        Args:
            query: Original query
            chunks_with_scores: List of (chunk, score) tuples
            top_k: Number of results to return
            
        Returns:
            Reranked list of (chunk, score) tuples
        """
        if not self.model or not chunks_with_scores:
            return chunks_with_scores
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [(query, chunk.content) for chunk, _ in chunks_with_scores]
            
            # Get cross-encoder scores in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                cross_scores = await loop.run_in_executor(
                    executor,
                    self.model.predict,
                    pairs
                )
            
            # Combine with original chunks
            reranked_results = []
            for i, (chunk, original_score) in enumerate(chunks_with_scores):
                cross_score = float(cross_scores[i])
                reranked_results.append((chunk, cross_score))
            
            # Sort by cross-encoder scores (descending)
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k if specified
            if top_k is not None:
                reranked_results = reranked_results[:top_k]
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return chunks_with_scores


class HybridRetriever:
    """Hybrid retrieval system combining vector and keyword search.
    
    Features:
    - Semantic search with ChromaDB vector store
    - Keyword search with BM25
    - Reciprocal Rank Fusion for result combination
    - Optional cross-encoder reranking
    - Optimized for 612MB dataset
    """
    
    def __init__(self, 
                 vector_store: VectorStoreManager,
                 chunks: List[Chunk] = None,
                 alpha: float = None,
                 enable_reranking: bool = None):
        """Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store manager
            chunks: List of chunks for BM25 (if not provided, will load from vector store)
            alpha: Balance between vector (1.0) and BM25 (0.0) search
            enable_reranking: Enable cross-encoder reranking
        """
        self.vector_store = vector_store
        self.alpha = alpha if alpha is not None else settings.hybrid_search_alpha
        self.enable_reranking = (
            enable_reranking if enable_reranking is not None 
            else settings.enable_reranking
        )
        
        # Initialize BM25 retriever
        if chunks:
            self.bm25_retriever = BM25Retriever(chunks)
        else:
            # Load chunks from vector store (for existing collections)
            self.bm25_retriever = None
            logger.warning("No chunks provided for BM25. Will use vector search only.")
        
        # Initialize reranker if enabled
        if self.enable_reranking:
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None
        
        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0.0,
            'vector_search_time': 0.0,
            'bm25_search_time': 0.0,
            'reranking_time': 0.0,
            'fusion_time': 0.0,
        }
        
        logger.info(f"HybridRetriever initialized: alpha={self.alpha}, "
                   f"reranking={self.enable_reranking}")
    
    async def retrieve(self, query: str, 
                      top_k: int = None,
                      filters: Dict[str, Any] = None,
                      retrieval_method: str = "hybrid") -> RetrievalResult:
        """Retrieve relevant chunks using hybrid approach.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters for vector search
            retrieval_method: "hybrid", "vector", or "bm25"
            
        Returns:
            RetrievalResult with chunks and scores
        """
        start_time = time.time()
        top_k = top_k or settings.retrieval_top_k
        
        if not query.strip():
            logger.warning("Empty query provided")
            return RetrievalResult(
                chunks=[], scores=[], query=query,
                retrieval_time=0.0, total_chunks_searched=0,
                retrieval_method=retrieval_method
            )
        
        try:
            if retrieval_method == "vector":
                # Vector search only
                results = await self._vector_search(query, top_k, filters)
            elif retrieval_method == "bm25":
                # BM25 search only
                results = await self._bm25_search(query, top_k)
            else:
                # Hybrid search (default)
                results = await self._hybrid_search(query, top_k, filters)
            
            # Optional reranking
            if self.enable_reranking and self.reranker and results:
                rerank_start = time.time()
                results = await self.reranker.rerank(query, results, top_k)
                self.stats['reranking_time'] += time.time() - rerank_start
            
            # Extract chunks and scores
            chunks = [chunk for chunk, _ in results]
            scores = [score for _, score in results]
            
            # Update statistics
            retrieval_time = time.time() - start_time
            self.stats['total_queries'] += 1
            self.stats['avg_retrieval_time'] = (
                (self.stats['avg_retrieval_time'] * (self.stats['total_queries'] - 1) + 
                 retrieval_time) / self.stats['total_queries']
            )
            
            logger.debug(f"Retrieved {len(chunks)} chunks in {retrieval_time:.3f}s")
            
            return RetrievalResult(
                chunks=chunks,
                scores=scores,
                query=query,
                retrieval_time=retrieval_time,
                total_chunks_searched=len(results),
                retrieval_method=retrieval_method
            )
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return RetrievalResult(
                chunks=[], scores=[], query=query,
                retrieval_time=time.time() - start_time,
                total_chunks_searched=0,
                retrieval_method=retrieval_method
            )
    
    async def _vector_search(self, query: str, top_k: int,
                           filters: Dict[str, Any] = None) -> List[Tuple[Chunk, float]]:
        """Perform vector similarity search."""
        vector_start = time.time()
        
        try:
            # Run the synchronous similarity_search in a thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                results = await loop.run_in_executor(
                    executor,
                    lambda: self.vector_store.similarity_search(query, top_k, filters)
                )
            
            self.stats['vector_search_time'] += time.time() - vector_start
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    async def _bm25_search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        """Perform BM25 keyword search."""
        if not self.bm25_retriever:
            logger.warning("BM25 retriever not available")
            return []
        
        bm25_start = time.time()
        
        try:
            # Run BM25 in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                results = await loop.run_in_executor(
                    executor,
                    self.bm25_retriever.search,
                    query,
                    top_k
                )
            
            self.stats['bm25_search_time'] += time.time() - bm25_start
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    async def _hybrid_search(self, query: str, top_k: int,
                           filters: Dict[str, Any] = None) -> List[Tuple[Chunk, float]]:
        """Perform hybrid search combining vector and BM25."""
        fusion_start = time.time()
        
        try:
            # Perform both searches concurrently
            vector_task = self._vector_search(query, top_k * 2, filters)  # Get more for fusion
            bm25_task = self._bm25_search(query, top_k * 2)
            
            vector_results, bm25_results = await asyncio.gather(
                vector_task, bm25_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed: {vector_results}")
                vector_results = []
            
            if isinstance(bm25_results, Exception):
                logger.error(f"BM25 search failed: {bm25_results}")
                bm25_results = []
            
            # Combine results using Reciprocal Rank Fusion
            combined_results = self._reciprocal_rank_fusion(
                vector_results, bm25_results, self.alpha
            )
            
            self.stats['fusion_time'] += time.time() - fusion_start
            
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, 
                               vector_results: List[Tuple[Chunk, float]],
                               bm25_results: List[Tuple[Chunk, float]],
                               alpha: float) -> List[Tuple[Chunk, float]]:
        """Combine results using Reciprocal Rank Fusion.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            alpha: Weight for vector vs BM25 (0.0 = pure BM25, 1.0 = pure vector)
            
        Returns:
            Combined and ranked results
        """
        # Create chunk score maps
        vector_scores = {chunk.chunk_id: score for chunk, score in vector_results}
        bm25_scores = {chunk.chunk_id: score for chunk, score in bm25_results}
        
        # Get all unique chunks
        all_chunk_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        chunk_map = {}
        
        # Build chunk map
        for chunk, _ in vector_results + bm25_results:
            chunk_map[chunk.chunk_id] = chunk
        
        # Calculate combined scores using RRF
        combined_scores = {}
        k = 60  # RRF parameter
        
        for chunk_id in all_chunk_ids:
            vector_score = vector_scores.get(chunk_id, 0.0)
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            
            # Reciprocal Rank Fusion formula
            vector_rank = self._get_rank(chunk_id, vector_results)
            bm25_rank = self._get_rank(chunk_id, bm25_results)
            
            rrf_score = 0.0
            if vector_rank > 0:
                rrf_score += alpha / (k + vector_rank)
            if bm25_rank > 0:
                rrf_score += (1 - alpha) / (k + bm25_rank)
            
            combined_scores[chunk_id] = rrf_score
        
        # Sort by combined score
        sorted_chunks = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return as list of (chunk, score) tuples
        results = []
        for chunk_id, score in sorted_chunks:
            if chunk_id in chunk_map:
                results.append((chunk_map[chunk_id], score))
        
        return results
    
    def _get_rank(self, chunk_id: str, 
                  results: List[Tuple[Chunk, float]]) -> int:
        """Get rank of chunk in results (1-indexed, 0 if not found)."""
        for i, (chunk, _) in enumerate(results):
            if chunk.chunk_id == chunk_id:
                return i + 1
        return 0
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        stats = dict(self.stats)
        
        # Add BM25 stats if available
        if self.bm25_retriever:
            stats.update({
                'bm25_stats': self.bm25_retriever.get_stats(),
            })
        
        # Add vector store stats
        if self.vector_store:
            stats.update({
                'vector_store_stats': self.vector_store.get_collection_stats(),
            })
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on retrieval system."""
        health = {
            'status': 'healthy',
            'components': {
                'vector_store': 'unknown',
                'bm25_retriever': 'available' if self.bm25_retriever else 'unavailable',
                'reranker': 'available' if self.reranker and self.reranker.model else 'unavailable'
            }
        }
        
        # Check vector store
        try:
            vector_health = await self.vector_store.health_check()
            health['components']['vector_store'] = vector_health['status']
        except Exception as e:
            health['components']['vector_store'] = f'error: {e}'
            health['status'] = 'degraded'
        
        return health


# Convenience function
async def create_hybrid_retriever(vector_store: VectorStoreManager,
                                chunks: List[Chunk] = None,
                                **kwargs) -> HybridRetriever:
    """Create hybrid retriever instance.
    
    Args:
        vector_store: Initialized vector store
        chunks: Chunks for BM25 indexing
        **kwargs: Additional arguments for HybridRetriever
        
    Returns:
        HybridRetriever instance
    """
    retriever = HybridRetriever(vector_store, chunks, **kwargs)
    return retriever