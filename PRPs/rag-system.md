name: "RAG System Implementation PRP"
description: |

## Purpose

Implement a comprehensive RAG (Retrieval-Augmented Generation) system that can answer questions based on a given set of documents. The system will feature modular architecture, hybrid retrieval, comprehensive evaluation, and production-ready deployment capabilities.

## Core Principles

1. **Context is King**: Comprehensive research findings integrated for optimal implementation
2. **Validation Loops**: Multi-tier testing with RAG-specific metrics
3. **Information Dense**: Leverage latest 2024-2025 RAG techniques and patterns
4. **Progressive Success**: Start with core functionality, validate, then enhance

---

## Goal

Build a production-ready RAG system optimized for 612MB document library (1,566 documents) that can:
- Process 800MB of documents efficiently with local embeddings (2-4 hour initial processing)
- Answer questions using hybrid retrieval (vector + keyword search) with <1s response time
- Provide source attribution and confidence scoring
- Include comprehensive evaluation and monitoring
- Support multiple document formats (PDF, TXT, MD, DOCX)
- Implement advanced RAG techniques (contextual retrieval, reranking)
- Use local models to minimize costs (zero ongoing API costs vs $200-400/month)
- Process documents from `data/SpecificationDocuments/` directory structure

## Why

- **Business Value**: Enable intelligent document search and question-answering
- **Technical Value**: Demonstrate modern RAG implementation patterns and best practices
- **Integration**: Foundation for enterprise knowledge management systems
- **Problems Solved**: Eliminates manual document searching, provides accurate information retrieval with source attribution

## What

A complete RAG system with the following user-visible behavior:
- Command-line interface for document ingestion and querying
- REST API for integration with other systems
- Web interface for interactive question-answering
- Comprehensive evaluation dashboard showing system performance metrics

### Success Criteria

- [ ] Successfully processes and indexes 612MB document library with 1,566 documents (300,000-450,000 chunks)
- [ ] Answers questions with >85% relevance (measured by RAGAs metrics)
- [ ] Provides source attribution for all answers
- [ ] Achieves <1s response time for typical queries (optimized for 612MB dataset)
- [ ] Uses <6GB RAM during operation with local embeddings (optimized for 612MB)
- [ ] Processes initial document ingestion in <3 hours (1,566 documents)
- [ ] Includes comprehensive test coverage (>90%)
- [ ] Passes all RAG-specific evaluation metrics
- [ ] Production-ready with proper error handling and logging

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- url: https://python.langchain.com/docs/tutorials/rag/
  why: Official LangChain RAG tutorial with latest patterns and best practices

- url: https://docs.ragas.io/
  why: RAGAs evaluation framework for comprehensive RAG system testing

- url: https://www.trulens.org/
  why: TruLens RAG Triad evaluation methodology for production systems

- doc: https://www.anthropic.com/news/contextual-retrieval
  section: Contextual Retrieval implementation
  critical: 67% reduction in incorrect retrievals through context preprocessing

- doc: https://arxiv.org/html/2407.21059v1
  section: Modular RAG architecture patterns
  critical: LEGO-like reconfigurable components for production systems

- url: https://github.com/explodinggradients/ragas
  why: Reference implementation for RAG evaluation metrics

- url: https://docs.llamaindex.ai/
  why: Alternative framework option with data-centric approach

- url: https://www.pinecone.io/learn/retrieval-augmented-generation/
  why: Production RAG deployment best practices and patterns

- url: https://docs.trychroma.com/
  why: ChromaDB documentation for local vector storage optimized for 800MB datasets
  critical: Perfect for 612MB dataset, handles millions of vectors with persistent storage

- url: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  why: Local embedding model for cost-effective processing
  critical: 100x faster than API calls, zero ongoing costs, 85-90% of OpenAI performance
```

### Current Codebase Structure

```bash
/Users/kweng/AI/RAG/
├── PRPs/                          # Project Requirements Prompts
│   ├── README.md                  # PRP framework documentation
│   ├── ai_docs/                   # AI documentation context
│   ├── scripts/
│   │   └── prp_runner.py         # PRP execution framework
│   └── templates/
│       └── prp_base.md           # Base PRP template
└── REQUEST.txt                    # Original request specification
```

### Desired Codebase Structure

```bash
/Users/kweng/AI/RAG/
├── PRPs/                          # Existing PRP framework
├── rag_system/                    # Main RAG implementation
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # Configuration management
│   ├── core/
│   │   ├── __init__.py
│   │   ├── document_processor.py # Document ingestion and chunking
│   │   ├── retriever.py          # Hybrid retrieval implementation
│   │   ├── generator.py          # Response generation
│   │   └── evaluator.py          # RAGAs-based evaluation
│   ├── storage/
│   │   ├── __init__.py
│   │   └── vector_store.py       # Vector database abstraction
│   ├── api/
│   │   ├── __init__.py
│   │   ├── cli.py                # Command-line interface
│   │   └── rest_api.py           # FastAPI REST endpoints
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py     # Structured logging setup
│       └── metrics.py            # Performance monitoring
├── tests/                         # Comprehensive test suite
│   ├── __init__.py
│   ├── unit/                     # Unit tests for each component
│   ├── integration/              # End-to-end integration tests
│   └── evaluation/               # RAGAs evaluation tests
├── data/                         # Document storage
│   ├── SpecificationDocuments/   # Input documents directory (1,566 files, 612MB)
│   ├── processed/                # Chunked and processed documents
│   └── indices/                  # Vector database storage
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
├── README.md                    # Project documentation
└── docker-compose.yml          # Development environment
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: LangChain 0.1.0+ requires explicit package imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

# CRITICAL: RAGAs requires async implementations for evaluation
# All evaluation functions must be called with await

# CRITICAL: Chroma requires explicit client initialization
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")

# CRITICAL: For 612MB dataset (1,566 documents), use local embeddings for cost efficiency
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # 22MB model, CPU optimized
embeddings = model.encode(texts, batch_size=1000)  # Batch processing

# CRITICAL: Optimized chunking for 612MB dataset (1,566 documents)
# Reduced overlap for better performance with large datasets
chunk_size = 1000  # Optimal for sentence-transformers
chunk_overlap = 100  # 10% overlap for 612MB efficiency
batch_size = 1000  # Process 1000 chunks at once

# CRITICAL: Memory management for large dataset
# Process in batches to avoid memory issues
max_concurrent_embeddings = 1000  # Adjust based on available RAM

# CRITICAL: Hybrid search requires proper weighting
# alpha=0.5 provides balanced semantic and keyword search
alpha = 0.5  # 0 = pure keyword, 1 = pure semantic
```

## Implementation Blueprint

### Data Models and Structure

Create type-safe data models ensuring consistency across the system:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    source: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    doc_id: str
    start_char: int
    end_char: int
    embedding: Optional[List[float]] = None

@dataclass
class RetrievalResult:
    chunks: List[Chunk]
    scores: List[float]
    query: str
    retrieval_time: float

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, str]]
    confidence_score: float
    query: str
    response_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class DocumentType(Enum):
    PDF = "pdf"
    TEXT = "txt"
    MARKDOWN = "md"
    DOCX = "docx"
    HTML = "html"
```

### List of Tasks to be Completed

```yaml
Task 1 - Project Setup and Configuration:
MODIFY pyproject.toml:
  - CREATE new project configuration with dependencies optimized for 800MB dataset
  - INCLUDE langchain==0.1.0, langchain-community, chromadb (primary vector store)
  - ADD sentence-transformers>=2.2.0 (local embeddings), ragas, fastapi, uvicorn
  - INCLUDE efficient document processing: PyPDF2, python-docx, beautifulsoup4
  - ADD memory optimization: psutil for monitoring, tqdm for progress bars
  - CONFIGURE development dependencies: pytest, ruff, mypy, black

CREATE rag_system/__init__.py:
  - SETUP package structure and version info
  - EXPORT main classes for external use

CREATE rag_system/config/settings.py:
  - IMPLEMENT configuration management using pydantic BaseSettings
  - SUPPORT environment variables and config files
  - INCLUDE all RAG system parameters optimized for 800MB dataset:
    * chunk_size=1000, chunk_overlap=100
    * embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    * vector_store="chromadb", batch_size=1000
    * max_memory_usage=8192  # MB limit

Task 2 - Document Processing Pipeline:
CREATE rag_system/core/document_processor.py:
  - IMPLEMENT DocumentProcessor class with multi-format support
  - SUPPORT PDF (PyPDF2), DOCX (python-docx), TXT, MD, HTML formats
  - IMPLEMENT semantic chunking with configurable overlap
  - ADD metadata extraction (filename, creation date, document type)
  - INCLUDE text preprocessing and cleaning

CREATE rag_system/utils/text_processing.py:
  - IMPLEMENT text cleaning utilities (normalize unicode, remove extra whitespace)
  - ADD content filtering and PII detection capabilities
  - IMPLEMENT contextual retrieval preprocessing

Task 3 - Vector Storage and Retrieval:
CREATE rag_system/storage/vector_store.py:
  - IMPLEMENT VectorStoreManager with ChromaDB backend (optimized for 800MB)
  - SUPPORT persistent storage with efficient indexing for 300K-450K chunks
  - ADD batch embedding with sentence-transformers (1000 batch size)
  - IMPLEMENT similarity search with metadata filtering and memory management
  - ADD progress tracking for large document processing
  - INCLUDE memory monitoring and garbage collection for sustained processing

CREATE rag_system/core/retriever.py:
  - IMPLEMENT HybridRetriever combining vector and BM25 search
  - ADD reranking with cross-encoder models
  - IMPLEMENT query expansion and enhancement
  - SUPPORT contextual retrieval with chunk preprocessing
  - ADD reciprocal rank fusion for multi-method retrieval

Task 4 - Response Generation:
CREATE rag_system/core/generator.py:
  - IMPLEMENT ResponseGenerator with LangChain integration
  - SUPPORT multiple LLM providers (OpenAI, Anthropic, local models)
  - ADD prompt templates for different query types
  - IMPLEMENT confidence scoring and response validation
  - ADD source attribution and citation generation

CREATE rag_system/core/pipeline.py:
  - IMPLEMENT RAGPipeline orchestrating all components
  - ADD async processing capabilities
  - IMPLEMENT error handling with graceful degradation
  - ADD request/response logging and metrics collection

Task 5 - Evaluation Framework:
CREATE rag_system/core/evaluator.py:
  - IMPLEMENT RAGEvaluator using RAGAs framework
  - SUPPORT faithfulness, answer_relevancy, context_precision metrics
  - ADD custom evaluation metrics for domain-specific requirements
  - IMPLEMENT batch evaluation for large test sets
  - ADD evaluation result storage and reporting

CREATE tests/evaluation/rag_metrics.py:
  - IMPLEMENT comprehensive evaluation test suite
  - CREATE benchmark datasets for testing
  - ADD performance benchmarking utilities
  - IMPLEMENT continuous evaluation pipeline

Task 6 - API and Interface Development:
CREATE rag_system/api/cli.py:
  - IMPLEMENT command-line interface using Click
  - SUPPORT document ingestion, querying, and evaluation commands
  - ADD progress bars and status indicators
  - IMPLEMENT configuration management through CLI

CREATE rag_system/api/rest_api.py:
  - IMPLEMENT FastAPI REST endpoints
  - ADD async request handling with proper error responses
  - IMPLEMENT request validation with Pydantic models
  - ADD OpenAPI documentation and interactive interface
  - SUPPORT file upload for document ingestion

CREATE rag_system/api/web_interface.py:
  - IMPLEMENT simple web interface for testing and demos
  - ADD document upload and query capabilities
  - IMPLEMENT real-time evaluation metrics display

Task 7 - Monitoring and Logging:
CREATE rag_system/utils/logging_config.py:
  - IMPLEMENT structured logging with JSON format
  - SUPPORT different log levels and output destinations
  - ADD request tracing and correlation IDs

CREATE rag_system/utils/metrics.py:
  - IMPLEMENT performance monitoring with Prometheus metrics
  - ADD latency, throughput, and accuracy tracking
  - IMPLEMENT health checks and system status monitoring

Task 8 - Testing and Quality Assurance:
CREATE tests/unit/ directory structure:
  - IMPLEMENT unit tests for each component
  - ACHIEVE >90% code coverage
  - ADD property-based testing for edge cases

CREATE tests/integration/test_end_to_end.py:
  - IMPLEMENT end-to-end testing with sample documents
  - ADD performance testing and load testing
  - IMPLEMENT regression testing for evaluation metrics

Task 9 - Documentation and Deployment:
CREATE README.md:
  - PROVIDE comprehensive setup and usage instructions
  - INCLUDE architecture overview and design decisions
  - ADD performance benchmarks and evaluation results

CREATE docker-compose.yml:
  - IMPLEMENT development environment setup
  - INCLUDE vector database, API, and monitoring services
  - ADD sample data and configuration

CREATE deployment/ directory:
  - IMPLEMENT production deployment configurations
  - ADD Kubernetes manifests and Docker files
  - INCLUDE monitoring and alerting setup
```

### Implementation Pseudocode

```python
# Task 3 - Core HybridRetriever Implementation (Optimized for 800MB)
class HybridRetriever:
    def __init__(self, vector_store, bm25_retriever, reranker, alpha=0.5):
        self.vector_store = vector_store  # ChromaDB with 400K-600K chunks
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.alpha = alpha  # Semantic vs keyword search balance
        self.max_results = 20  # Limit for memory efficiency with large dataset
    
    async def retrieve(self, query: str, top_k: int = 20) -> List[Chunk]:
        # PATTERN: Parallel retrieval for better performance
        vector_task = asyncio.create_task(
            self.vector_store.similarity_search(query, k=top_k)
        )
        bm25_task = asyncio.create_task(
            self.bm25_retriever.search(query, k=top_k)
        )
        
        vector_results, bm25_results = await asyncio.gather(
            vector_task, bm25_task
        )
        
        # CRITICAL: Reciprocal Rank Fusion for combining results
        combined_results = self.reciprocal_rank_fusion(
            vector_results, bm25_results
        )
        
        # PATTERN: Reranking for better relevance
        if self.reranker:
            combined_results = await self.reranker.rerank(
                query, combined_results[:top_k*2]
            )
        
        return combined_results[:top_k]

# Task 4 - Response Generation with Validation
class ResponseGenerator:
    def __init__(self, llm, evaluator):
        self.llm = llm
        self.evaluator = evaluator
        self.confidence_threshold = 0.7
    
    async def generate(self, query: str, chunks: List[Chunk]) -> RAGResponse:
        # PATTERN: Context preparation with source attribution
        context = self.prepare_context(chunks)
        
        # CRITICAL: Prompt engineering for better responses
        prompt = self.build_prompt(query, context)
        
        # PATTERN: Response generation with error handling
        try:
            response = await self.llm.agenerate(prompt)
            
            # CRITICAL: Response validation and confidence scoring
            confidence = await self.evaluator.compute_confidence(
                query, context, response
            )
            
            if confidence < self.confidence_threshold:
                response = self.generate_fallback_response(query, chunks)
            
            return RAGResponse(
                answer=response,
                sources=self.extract_sources(chunks),
                confidence_score=confidence,
                query=query,
                response_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self.handle_generation_error(query, e)

# Task 5 - Comprehensive Evaluation
class RAGEvaluator:
    def __init__(self):
        # CRITICAL: Initialize RAGAs metrics
        self.faithfulness = Faithfulness()
        self.answer_relevancy = AnswerRelevancy()
        self.context_precision = ContextPrecision()
        self.context_recall = ContextRecall()
    
    async def evaluate_response(
        self, 
        query: str, 
        response: str, 
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        # PATTERN: Create evaluation sample
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=contexts,
            reference=ground_truth
        )
        
        # CRITICAL: Async evaluation with all metrics
        results = {}
        results['faithfulness'] = await self.faithfulness.single_turn_ascore(sample)
        results['answer_relevancy'] = await self.answer_relevancy.single_turn_ascore(sample)
        results['context_precision'] = await self.context_precision.single_turn_ascore(sample)
        
        if ground_truth:
            results['context_recall'] = await self.context_recall.single_turn_ascore(sample)
        
        return results
```

### Integration Points

```yaml
VECTOR_DATABASE:
  - storage: "ChromaDB with persistent storage (optimal for 612MB dataset)"
  - embedding_model: "sentence-transformers/all-MiniLM-L6-v2 (local, cost-free)"
  - index_type: "ChromaDB default (optimized for millions of vectors)"
  - batch_processing: "1000 chunks per batch for memory efficiency"
  - dataset_info: "1,566 documents totaling 612MB"
  - expected_chunks: "400,000-600,000 total chunks"
  - processing_time: "2-4 hours initial ingestion"
  - memory_usage: "<8GB RAM during operation"

CONFIG_MANAGEMENT:
  - framework: "Pydantic BaseSettings"
  - sources: "Environment variables, YAML config files"
  - pattern: "Hierarchical configuration with environment overrides"

API_ROUTES:
  - add to: rag_system/api/rest_api.py
  - pattern: "FastAPI with async handlers and proper error responses"
  - endpoints: "/ingest", "/query", "/evaluate", "/health"

MONITORING:
  - framework: "Prometheus + Grafana"
  - metrics: "Request latency, accuracy scores, error rates"
  - pattern: "Structured logging with correlation IDs"

DATABASE_STORAGE:
  - vector_storage: "ChromaDB (development and production, scales to millions of vectors)"
  - metadata_storage: "ChromaDB integrated metadata (no separate DB needed)"
  - document_storage: "Local filesystem with organized directory structure"
  - backup_strategy: "ChromaDB persistent client with data directory backup"
  - scaling_path: "Qdrant for >5GB datasets, current setup handles 800MB-5GB"
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Run these FIRST - fix any errors before proceeding
ruff check rag_system/ tests/ --fix     # Auto-fix formatting and imports
ruff format rag_system/ tests/          # Code formatting
mypy rag_system/                        # Type checking with strict mode

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests

```python
# CREATE comprehensive unit test suite
# tests/unit/test_document_processor.py
def test_document_processing():
    """Test document ingestion and chunking"""
    processor = DocumentProcessor()
    documents = processor.process_directory("data/SpecificationDocuments/")
    
    assert len(documents) > 0
    assert all(doc.content for doc in documents)
    assert all(doc.metadata for doc in documents)

def test_chunking_strategy():
    """Test optimized chunking for 800MB dataset"""
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=100)  # Optimized for large datasets
    chunks = processor.chunk_document("Large document content...")
    
    assert len(chunks) >= 1
    assert all(len(chunk.content) <= 1000 for chunk in chunks)
    # Verify 10% overlap for efficiency with large datasets
    # Test memory usage stays under 8GB during processing

# tests/unit/test_retriever.py
def test_hybrid_retrieval():
    """Test combined vector and keyword search"""
    retriever = HybridRetriever(mock_vector_store, mock_bm25, None)
    results = retriever.retrieve("test query", top_k=5)
    
    assert len(results) <= 5
    assert all(isinstance(result, Chunk) for result in results)

def test_reranking():
    """Test cross-encoder reranking"""
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("query", mock_chunks)
    
    # Verify reranking improves relevance scores
    assert all(reranked[i].score >= reranked[i+1].score 
               for i in range(len(reranked)-1))
```

```bash
# Run and iterate until passing:
uv run pytest tests/unit/ -v --cov=rag_system --cov-report=html
# Target: >90% code coverage. If failing: Read error, fix code, re-run
```

### Level 3: RAG-Specific Evaluation

```python
# tests/evaluation/test_rag_metrics.py
import asyncio
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

async def test_rag_evaluation():
    """Test RAG system with RAGAS metrics"""
    # Load test dataset
    test_cases = load_evaluation_dataset("tests/data/eval_dataset.json")
    
    # Run evaluation
    results = []
    for case in test_cases:
        response = await rag_pipeline.process_query(case['query'])
        
        evaluation_result = await evaluator.evaluate_response(
            query=case['query'],
            response=response.answer,
            contexts=[chunk.content for chunk in response.sources],
            ground_truth=case.get('expected_answer')
        )
        results.append(evaluation_result)
    
    # Validate performance thresholds
    avg_faithfulness = sum(r['faithfulness'] for r in results) / len(results)
    avg_relevancy = sum(r['answer_relevancy'] for r in results) / len(results)
    
    assert avg_faithfulness >= 0.85, f"Low faithfulness: {avg_faithfulness}"
    assert avg_relevancy >= 0.85, f"Low relevancy: {avg_relevancy}"

def test_contextual_retrieval():
    """Test contextual retrieval implementation"""
    # Test that contextual preprocessing improves retrieval
    standard_results = standard_retriever.retrieve("test query")
    contextual_results = contextual_retriever.retrieve("test query")
    
    # Contextual retrieval should show improvement
    assert contextual_results[0].score >= standard_results[0].score
```

```bash
# Run RAG evaluation tests:
uv run pytest tests/evaluation/ -v --asyncio-mode=auto
# Expected: All RAG metrics above threshold (>0.85)
```

### Level 4: Integration Testing

```bash
# Start the RAG system
uv run python -m rag_system.api.cli ingest --directory data/SpecificationDocuments/
uv run python -m rag_system.api.rest_api &

# Test document ingestion
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@tests/data/sample_document.pdf"

# Test query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?", "max_sources": 3}'

# Expected: {"answer": "...", "sources": [...], "confidence": 0.XX}
```

### Level 5: Performance and Load Testing

```bash
# Performance benchmarking (612MB dataset targets)
uv run python -m rag_system.utils.benchmark \
  --queries 100 \
  --concurrent_users 5 \
  --target_latency 1000ms \
  --dataset_size 800MB

# Load testing with locust
pip install locust
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Expected: 95th percentile latency < 1s (optimized for 800MB), 99% success rate
```

### Level 6: Production Deployment Testing

```bash
# Docker deployment test
docker-compose up -d
docker-compose ps  # All services should be healthy

# Health check endpoint
curl http://localhost:8000/health
# Expected: {"status": "healthy", "components": {"vector_db": "ok", "llm": "ok"}}

# End-to-end production test
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Test production deployment", "max_sources": 3}'
```

## Final Validation Checklist

- [ ] All unit tests pass: `uv run pytest tests/unit/ -v`
- [ ] No linting errors: `uv run ruff check rag_system/ tests/`
- [ ] No type errors: `uv run mypy rag_system/`
- [ ] RAG evaluation metrics >85%: `uv run pytest tests/evaluation/ -v`
- [ ] Integration tests successful: Manual curl testing
- [ ] Performance targets met: <1s response time, >90% accuracy (612MB optimized)
- [ ] Error cases handled gracefully: Test with invalid inputs
- [ ] Documentation complete: README with setup and usage instructions
- [ ] Docker deployment successful: `docker-compose up -d`

---

## Anti-Patterns to Avoid

- ❌ Don't use fixed-size chunking without considering document structure
- ❌ Don't skip evaluation metrics - always implement RAGAs or equivalent
- ❌ Don't ignore vector database performance - choose appropriate indexing
- ❌ Don't hardcode API keys or sensitive configuration
- ❌ Don't implement RAG without proper error handling and fallbacks
- ❌ Don't skip reranking for better retrieval quality
- ❌ Don't ignore context window limitations - implement proper chunking
- ❌ Don't deploy without comprehensive logging and monitoring

## Quality Assessment

**Confidence Level for One-Pass Implementation: 9.5/10**

This PRP provides comprehensive context including:
- ✅ Latest 2024-2025 RAG research and best practices
- ✅ Detailed implementation blueprint with specific tasks
- ✅ Multiple validation levels with executable commands
- ✅ Production-ready architecture patterns
- ✅ Comprehensive evaluation framework with RAGAs integration
- ✅ Error handling and monitoring strategies
- ✅ Complete technology stack with version specifications
- ✅ **Optimized for 800MB dataset**: ChromaDB + local embeddings + efficient chunking
- ✅ **Cost optimization**: Zero ongoing API costs vs $200-400/month
- ✅ **Performance targets**: <1s queries, <8GB RAM, 2-4h initial processing
- ✅ **Scalability path**: Clear migration to Qdrant for larger datasets

The high confidence score reflects the thorough research, specific implementation guidance optimized for the exact 800MB use case, and comprehensive validation framework that should enable successful one-pass implementation with optimal technology choices.