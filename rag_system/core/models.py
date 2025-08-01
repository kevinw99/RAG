"""
Data models for RAG system components.

Type-safe data structures ensuring consistency across the 612MB document processing pipeline.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DocumentType(Enum):
    """Supported document formats."""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Document:
    """Represents a source document in the RAG system."""
    content: str
    metadata: Dict[str, Any]
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_path: Optional[Path] = None
    document_type: DocumentType = DocumentType.UNKNOWN
    file_size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_error: Optional[str] = None
    
    def __post_init__(self):
        """Validate document after initialization."""
        if not self.content.strip():
            raise ValueError("Document content cannot be empty")
        
        # Auto-detect document type from source path if not set
        if self.document_type == DocumentType.UNKNOWN and self.source_path:
            suffix = self.source_path.suffix.lower()
            type_mapping = {
                ".pdf": DocumentType.PDF,
                ".docx": DocumentType.DOCX,
                ".doc": DocumentType.DOC,
                ".txt": DocumentType.TXT,
                ".md": DocumentType.MD,
                ".html": DocumentType.HTML,
                ".htm": DocumentType.HTML,
            }
            self.document_type = type_mapping.get(suffix, DocumentType.UNKNOWN)
    
    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return len(self.content.split())
    
    @property
    def character_count(self) -> int:
        """Get character count."""
        return len(self.content)


@dataclass
class Chunk:
    """Represents a text chunk from document processing."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate chunk after initialization."""
        if not self.content.strip():
            raise ValueError("Chunk content cannot be empty")
        if self.start_char < 0:
            raise ValueError("start_char must be non-negative")
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
    
    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return len(self.content.split())
    
    @property
    def character_count(self) -> int:
        """Get character count."""
        return len(self.content)


@dataclass
class RetrievalResult:
    """Results from hybrid retrieval system."""
    query: str
    chunks: List[Chunk]
    scores: List[float]
    retrieval_time: float
    method: str = "hybrid"  # hybrid, vector, keyword
    total_candidates: int = 0
    reranked: bool = False
    
    def __post_init__(self):
        """Validate retrieval results."""
        if len(self.chunks) != len(self.scores):
            raise ValueError("Number of chunks must match number of scores")
        if any(score < 0 for score in self.scores):
            raise ValueError("All scores must be non-negative")


@dataclass
class RAGResponse:
    """Complete response from RAG system."""
    answer: str
    sources: List[Dict[str, str]]
    confidence_score: float
    query: str
    response_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_results: Optional[RetrievalResult] = None
    evaluation_scores: Optional[Dict[str, float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate RAG response."""
        if not self.answer.strip():
            raise ValueError("Answer cannot be empty")
        # Handle NaN confidence scores
        import math
        if math.isnan(self.confidence_score):
            self.confidence_score = 0.5  # Default to moderate confidence
        elif not 0 <= self.confidence_score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
        if self.response_time < 0:
            raise ValueError("Response time must be non-negative")


@dataclass
class ProcessingStats:
    """Statistics from document processing operations."""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    skipped_documents: int = 0
    total_chunks: int = 0
    total_characters: int = 0
    total_words: int = 0
    processing_time: float = 0.0
    average_chunk_size: float = 0.0
    memory_usage_mb: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate processing success rate."""
        if self.total_documents == 0:
            return 0.0
        return self.processed_documents / self.total_documents
    
    @property
    def failure_rate(self) -> float:
        """Calculate processing failure rate."""
        if self.total_documents == 0:
            return 0.0
        return self.failed_documents / self.total_documents
    
    @property
    def chunks_per_document(self) -> float:
        """Calculate average chunks per document."""
        if self.processed_documents == 0:
            return 0.0
        return self.total_chunks / self.processed_documents


@dataclass
class EvaluationResult:
    """Results from RAG system evaluation."""
    query: str
    response: str
    ground_truth: Optional[str] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    overall_score: Optional[float] = None
    evaluation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_overall_score(self) -> float:
        """Compute overall evaluation score from individual metrics."""
        scores = []
        if self.faithfulness is not None:
            scores.append(self.faithfulness)
        if self.answer_relevancy is not None:
            scores.append(self.answer_relevancy)
        if self.context_precision is not None:
            scores.append(self.context_precision)
        if self.context_recall is not None:
            scores.append(self.context_recall)
        
        if not scores:
            return 0.0
        
        self.overall_score = sum(scores) / len(scores)
        return self.overall_score
