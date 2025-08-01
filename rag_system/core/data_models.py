"""Data models and structures for RAG system.

Type-safe data models ensuring consistency across the system.
Optimized for 612MB document library processing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pathlib import Path


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    TEXT = "txt"
    MARKDOWN = "md"
    DOCX = "docx"
    DOC = "doc"
    HTML = "html"


@dataclass
class Document:
    """Document representation with metadata."""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    source: Union[str, Path]
    doc_type: DocumentType
    created_at: datetime = field(default_factory=datetime.now)
    file_size: Optional[int] = None
    
    def __post_init__(self):
        """Ensure source is a string for serialization."""
        if isinstance(self.source, Path):
            self.source = str(self.source)


@dataclass
class Chunk:
    """Document chunk with embedding and metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    doc_id: str
    start_char: int
    end_char: int
    chunk_index: int
    embedding: Optional[List[float]] = None
    
    @property
    def length(self) -> int:
        """Get chunk length in characters."""
        return len(self.content)


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    chunks: List[Chunk]
    scores: List[float]
    query: str
    retrieval_time: float
    total_chunks_searched: int
    retrieval_method: str = "hybrid"  # hybrid, vector, keyword
    
    def get_top_chunks(self, k: int) -> List[Chunk]:
        """Get top k chunks by score."""
        sorted_chunks = sorted(
            zip(self.chunks, self.scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        return [chunk for chunk, _ in sorted_chunks[:k]]


@dataclass
class RAGResponse:
    """Complete RAG system response."""
    answer: str
    sources: List[Dict[str, str]]
    confidence_score: float
    query: str
    response_time: float
    retrieval_result: Optional[RetrievalResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to response."""
        self.metadata[key] = value


@dataclass
class ProcessingStats:
    """Statistics for document processing."""
    total_documents: int = 0
    processed_documents: int = 0
    total_chunks: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    file_sizes: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate processing success rate."""
        if self.total_documents == 0:
            return 0.0
        return self.processed_documents / self.total_documents
    
    @property
    def average_chunks_per_doc(self) -> float:
        """Calculate average chunks per document."""
        if self.processed_documents == 0:
            return 0.0
        return self.total_chunks / self.processed_documents


@dataclass
class EvaluationResult:
    """Result from RAG evaluation."""
    query: str
    ground_truth: Optional[str]
    generated_answer: str
    retrieved_contexts: List[str]
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    overall_score: Optional[float] = None
    
    def calculate_overall_score(self) -> float:
        """Calculate weighted overall score."""
        scores = [
            score for score in [
                self.faithfulness,
                self.answer_relevancy, 
                self.context_precision,
                self.context_recall
            ] if score is not None
        ]
        
        if not scores:
            return 0.0
            
        self.overall_score = sum(scores) / len(scores)
        return self.overall_score


@dataclass 
class BatchEvaluationResult:
    """Results from batch evaluation."""
    individual_results: List[EvaluationResult]
    average_faithfulness: float
    average_answer_relevancy: float
    average_context_precision: float
    average_context_recall: float
    overall_average: float
    evaluation_time: float
    
    @classmethod
    def from_results(cls, results: List[EvaluationResult], evaluation_time: float):
        """Create batch result from individual results."""
        if not results:
            return cls(
                individual_results=[],
                average_faithfulness=0.0,
                average_answer_relevancy=0.0,
                average_context_precision=0.0,
                average_context_recall=0.0,
                overall_average=0.0,
                evaluation_time=evaluation_time
            )
        
        # Calculate averages, handling None values
        def safe_average(values: List[Optional[float]]) -> float:
            valid_values = [v for v in values if v is not None]
            return sum(valid_values) / len(valid_values) if valid_values else 0.0
        
        avg_faithfulness = safe_average([r.faithfulness for r in results])
        avg_relevancy = safe_average([r.answer_relevancy for r in results])
        avg_precision = safe_average([r.context_precision for r in results])
        avg_recall = safe_average([r.context_recall for r in results])
        
        overall_avg = safe_average([avg_faithfulness, avg_relevancy, avg_precision, avg_recall])
        
        return cls(
            individual_results=results,
            average_faithfulness=avg_faithfulness,
            average_answer_relevancy=avg_relevancy,
            average_context_precision=avg_precision,
            average_context_recall=avg_recall,
            overall_average=overall_avg,
            evaluation_time=evaluation_time
        )