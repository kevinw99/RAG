"""Configuration management for RAG system using Pydantic BaseSettings.

Optimized for 612MB document library with environment variable support.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """RAG system configuration optimized for 612MB dataset (1,566 documents)."""
    
    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    documents_dir: Path = Field(default_factory=lambda: Path("data/SpecificationDocuments"))
    processed_dir: Path = Field(default_factory=lambda: Path("data/processed"))
    indices_dir: Path = Field(default_factory=lambda: Path("data/indices"))
    
    # Document Processing - Optimized for large dataset
    chunk_size: int = Field(default=1000, description="Optimal chunk size for sentence-transformers")
    chunk_overlap: int = Field(default=100, description="10% overlap for efficiency with 612MB dataset")
    supported_extensions: List[str] = Field(
        default=[".pdf", ".txt", ".md", ".docx", ".doc", ".html"],
        description="Supported document formats"
    )
    max_file_size_mb: int = Field(default=50, description="Maximum file size in MB")
    batch_size: int = Field(default=1000, description="Batch size for processing")
    
    # Embedding Configuration - Local models for zero ongoing costs
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local embedding model - 100x faster than API, zero costs"
    )
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")
    device: str = Field(default="cpu", description="Device for embeddings (cpu/cuda)")
    
    # Vector Storage - ChromaDB optimized for 612MB dataset
    vector_store_type: str = Field(default="chromadb", description="Vector database type")
    vector_store_path: Path = Field(
        default_factory=lambda: Path("data/indices/chroma_db"),
        description="ChromaDB storage path"
    )
    collection_name: str = Field(default="rag_documents", description="ChromaDB collection name")
    
    # Memory Management - Critical for 612MB dataset
    max_memory_usage_mb: int = Field(default=8192, description="Maximum RAM usage in MB")
    max_concurrent_embeddings: int = Field(default=1000, description="Max concurrent embeddings")
    enable_garbage_collection: bool = Field(default=True, description="Enable GC for memory management")
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(default=20, description="Number of chunks to retrieve")
    hybrid_search_alpha: float = Field(
        default=0.5, 
        description="Balance between semantic (1.0) and keyword (0.0) search"
    )
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranking model"
    )
    
    # Generation Configuration
    llm_provider: str = Field(default="openai", description="LLM provider (openai/anthropic/local)")
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    llm_temperature: float = Field(default=0.1, description="Generation temperature")
    max_tokens: int = Field(default=1000, description="Maximum response tokens")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence score")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=False, description="Enable auto-reload in development")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    
    # Evaluation Configuration
    evaluation_metrics: List[str] = Field(
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        description="RAGAs evaluation metrics"
    )
    evaluation_batch_size: int = Field(default=10, description="Evaluation batch size")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    
    # Performance Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    
    # Environment Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @field_validator("data_dir", "documents_dir", "processed_dir", "indices_dir", "vector_store_path")
    def ensure_absolute_paths(cls, v):
        """Ensure all paths are absolute."""
        if not v.is_absolute():
            return Path.cwd() / v
        return v
    
    @field_validator("chunk_overlap")
    def validate_chunk_overlap(cls, v):
        """Ensure chunk overlap is reasonable."""
        # Simple validation - chunk overlap should be reasonable
        if v >= 1000:  # Default chunk size
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    @field_validator("hybrid_search_alpha")
    def validate_alpha(cls, v):
        """Ensure alpha is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("hybrid_search_alpha must be between 0 and 1")
        return v
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for path in [self.data_dir, self.documents_dir, self.processed_dir, 
                     self.indices_dir, self.vector_store_path.parent]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration optimized for 612MB dataset."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "batch_size": self.batch_size,
            "max_memory_usage_mb": self.max_memory_usage_mb,
            "max_concurrent_embeddings": self.max_concurrent_embeddings,
            "enable_garbage_collection": self.enable_garbage_collection,
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for local processing."""
        return {
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "reranker_model": self.reranker_model if self.enable_reranking else None,
        }


# Global settings instance
settings = Settings()