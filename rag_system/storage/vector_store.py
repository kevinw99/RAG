"""
Vector storage system with ChromaDB backend.

Optimized for 612MB document library (1,566 documents) with:
- ChromaDB persistent client for 400K-600K chunks
- Local sentence-transformers embeddings (zero ongoing costs)
- Memory-efficient batch processing
- Metadata filtering and similarity search
- Progress tracking and error handling
"""

import gc
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import chromadb
import chromadb.errors
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import psutil

# Core imports
from rag_system.config.settings import settings
from rag_system.core.models import Chunk, Document

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Vector storage manager using ChromaDB with local embeddings.
    
    Optimized for large document collections (612MB, 1,566 documents).
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = None,
        embedding_model: str = None,
        batch_size: int = None,
        device: str = "cpu",
    ):
        """Initialize vector store manager.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model name
            batch_size: Batch size for embedding generation
            device: Device for embedding model (cpu/cuda)
        """
        self.persist_directory = str(persist_directory or settings.vector_store_path)
        self.collection_name = collection_name or settings.collection_name
        self.batch_size = batch_size or settings.batch_size
        self.device = device
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Initialize embedding model (local for zero costs)
        embedding_model_name = embedding_model or settings.embedding_model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model with dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Initialize or get collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except (ValueError, chromadb.errors.NotFoundError):
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        # Track statistics
        self.stats = {
            "total_chunks_added": 0,
            "total_embedding_time": 0.0,
            "total_storage_time": 0.0,
            "memory_usage_mb": 0.0,
        }
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        show_progress: bool = True,
        overwrite: bool = False,
    ) -> None:
        """Add chunks to vector store with embeddings.
        
        Optimized for 612MB dataset with memory management.
        
        Args:
            chunks: List of chunks to add
            show_progress: Show progress bar
            overwrite: Overwrite existing chunks
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        start_time = time.time()
        
        # Check if chunks already exist (unless overwriting)
        if not overwrite:
            chunks = self._filter_existing_chunks(chunks)
            if not chunks:
                logger.info("All chunks already exist in vector store")
                return
        
        # Process chunks in batches for memory efficiency
        progress_bar = tqdm(
            total=len(chunks),
            desc="Adding chunks",
            disable=not show_progress
        )
        
        try:
            for batch_start in range(0, len(chunks), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]
                
                # Check memory usage
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                if memory_mb > settings.max_memory_usage_mb * 0.8:
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB, running GC")
                    gc.collect()
                
                # Process batch
                self._add_chunk_batch(batch_chunks)
                
                progress_bar.update(len(batch_chunks))
                
                # Garbage collection after each batch
                if settings.enable_garbage_collection:
                    gc.collect()
        
        finally:
            progress_bar.close()
        
        # Update statistics
        self.stats["total_chunks_added"] += len(chunks)
        total_time = time.time() - start_time
        self.stats["total_storage_time"] += total_time
        self.stats["memory_usage_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
        
        logger.info(
            f"Added {len(chunks)} chunks in {total_time:.2f}s "
            f"(avg: {total_time/len(chunks)*1000:.1f}ms per chunk)"
        )
    
    def _filter_existing_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Filter out chunks that already exist in the vector store."""
        if not chunks:
            return chunks
        
        # Get existing IDs
        try:
            result = self.collection.get(ids=[chunk.chunk_id for chunk in chunks])
            existing_ids = set(result["ids"])
            
            # Filter out existing chunks
            filtered_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
            
            if len(filtered_chunks) < len(chunks):
                logger.info(
                    f"Filtered out {len(chunks) - len(filtered_chunks)} existing chunks"
                )
            
            return filtered_chunks
            
        except Exception as e:
            logger.warning(f"Error checking existing chunks: {e}")
            return chunks
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata for ChromaDB compatibility.
        
        ChromaDB only accepts str, int, float, bool, or None values.
        Convert other types (like datetime) to strings.
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, datetime):
                sanitized[key] = value.isoformat()
            elif isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        return sanitized
    
    def _add_chunk_batch(self, chunks: List[Chunk]) -> None:
        """Add a batch of chunks to the vector store."""
        if not chunks:
            return
        
        # Extract text content for embedding
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embedding_start = time.time()
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=min(self.batch_size, len(texts)),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            embedding_time = time.time() - embedding_start
            self.stats["total_embedding_time"] += embedding_time
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk.metadata) for chunk in chunks]
        documents = texts  # ChromaDB stores the original text
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=documents,
            )
            
        except Exception as e:
            logger.error(f"Error adding batch to ChromaDB: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_distances: bool = True,
    ) -> List[Tuple[Chunk, float]]:
        """Perform similarity search in vector store.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_dict: Metadata filters
            include_distances: Include similarity scores
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to chunks
            chunks_with_scores = []
            
            if results["ids"] and results["ids"][0]:  # Check if we have results
                for i, chunk_id in enumerate(results["ids"][0]):
                    # Create chunk from stored data
                    chunk = Chunk(
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                        chunk_id=chunk_id,
                        doc_id=results["metadatas"][0][i].get("doc_id", ""),
                        start_char=results["metadatas"][0][i].get("start_char", 0),
                        end_char=results["metadatas"][0][i].get("end_char", len(results["documents"][0][i])),
                        chunk_index=results["metadatas"][0][i].get("chunk_index", i),
                    )
                    
                    # Get similarity score (ChromaDB returns distances, convert to similarity)
                    distance = results["distances"][0][i] if include_distances else 0.0
                    similarity = 1.0 - distance  # Convert distance to similarity
                    
                    chunks_with_scores.append((chunk, similarity))
            
            return chunks_with_scores
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk object or None if not found
        """
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"] and results["ids"][0]:
                return Chunk(
                    content=results["documents"][0],
                    metadata=results["metadatas"][0],
                    chunk_id=chunk_id,
                    doc_id=results["metadatas"][0].get("doc_id", ""),
                    start_char=results["metadatas"][0].get("start_char", 0),
                    end_char=results["metadatas"][0].get("end_char", len(results["documents"][0])),
                    chunk_index=results["metadatas"][0].get("chunk_index", 0),
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks from vector store.
        
        Args:
            chunk_ids: List of chunk IDs to delete
        """
        if not chunk_ids:
            return
        
        try:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks")
            
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "embedding_dimension": self.embedding_dim,
                "embedding_model": str(self.embedding_model),
                "persist_directory": self.persist_directory,
                **self.stats,
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> None:
        """Clear all data from the collection.
        
        WARNING: This will delete all stored chunks!
        """
        try:
            # Get all IDs and delete them
            result = self.collection.get()
            if result["ids"]:
                self.collection.delete(ids=result["ids"])
                logger.info(f"Cleared {len(result['ids'])} chunks from collection")
            else:
                logger.info("Collection is already empty")
                
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
    
    def optimize_storage(self) -> None:
        """Optimize storage and run maintenance tasks."""
        logger.info("Running storage optimization")
        
        try:
            # Force garbage collection
            gc.collect()
            
            # Log current memory usage
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Current memory usage: {memory_mb:.1f}MB")
            
            # Update statistics
            self.stats["memory_usage_mb"] = memory_mb
            
        except Exception as e:
            logger.error(f"Error during storage optimization: {e}")
    
    def backup_collection(self, backup_path: str) -> None:
        """Create a backup of the collection.
        
        Args:
            backup_path: Path for backup storage
        """
        # ChromaDB uses persistent storage, so we can copy the directory
        import shutil
        
        try:
            backup_path = Path(backup_path)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy the persistent directory
            shutil.copytree(
                self.persist_directory,
                backup_path / "chroma_backup",
                dirs_exist_ok=True
            )
            
            logger.info(f"Collection backed up to: {backup_path}")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    def close(self) -> None:
        """Close the vector store connection."""
        try:
            # ChromaDB client doesn't need explicit closing
            logger.info("Vector store connection closed")
            
        except Exception as e:
            logger.error(f"Error closing vector store: {e}")


# Convenience functions
def create_vector_store(
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    **kwargs
) -> VectorStoreManager:
    """Create a vector store manager instance.
    
    Args:
        persist_directory: Directory for persistent storage
        collection_name: Name of the collection
        **kwargs: Additional arguments for VectorStoreManager
        
    Returns:
        VectorStoreManager instance
    """
    return VectorStoreManager(
        persist_directory=persist_directory,
        collection_name=collection_name,
        **kwargs
    )


def load_existing_vector_store(
    persist_directory: str,
    collection_name: str,
    **kwargs
) -> VectorStoreManager:
    """Load an existing vector store.
    
    Args:
        persist_directory: Directory containing the vector store
        collection_name: Name of the collection to load
        **kwargs: Additional arguments for VectorStoreManager
        
    Returns:
        VectorStoreManager instance
        
    Raises:
        ValueError: If the vector store doesn't exist
    """
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        raise ValueError(f"Persist directory not found: {persist_directory}")
    
    return VectorStoreManager(
        persist_directory=persist_directory,
        collection_name=collection_name,
        **kwargs
    )
