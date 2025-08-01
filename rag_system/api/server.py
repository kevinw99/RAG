"""FastAPI REST server for RAG system.

Production-ready API with async endpoints, error handling, and monitoring.
Optimized for 612MB document library with 44K+ chunks.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, Path as FastAPIPath, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from ..core.pipeline import RAGPipeline
from ..core.data_models import RAGResponse, ProcessingStats
from ..config.settings import settings
from ..monitoring import (
    track_request,
    get_metrics_summary,
    export_prometheus_metrics,
    run_health_checks,
    get_api_logger,
    add_request_context
)

logger = get_api_logger()

# Global pipeline instance
pipeline = None

# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    k: Optional[int] = Field(10, ge=1, le=50, description="Number of results to retrieve")
    rerank: Optional[bool] = Field(True, description="Enable reranking")
    template_type: Optional[str] = Field("default", description="Response template type")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")

    @validator('template_type')
    def validate_template_type(cls, v):
        allowed_types = ["default", "citation", "summary", "comparison"]
        if v not in allowed_types:
            raise ValueError(f"template_type must be one of {allowed_types}")
        return v


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")
    confidence_score: float = Field(..., description="Confidence score")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    response_time: float = Field(..., description="Response time in seconds")
    retrieval_stats: Dict[str, Any] = Field(..., description="Retrieval statistics")
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        """Handle NaN confidence scores."""
        import math
        if math.isnan(v):
            return 0.5  # Default to moderate confidence
        return min(max(v, 0.0), 1.0)  # Clamp to [0, 1]


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    k: Optional[int] = Field(10, ge=1, le=100, description="Number of results")
    search_type: Optional[str] = Field("hybrid", description="Search type")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")

    @validator('search_type')
    def validate_search_type(cls, v):
        allowed_types = ["vector", "keyword", "hybrid"]
        if v not in allowed_types:
            raise ValueError(f"search_type must be one of {allowed_types}")
        return v


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    chunks: List[Dict[str, Any]] = Field(..., description="Retrieved chunks")
    scores: List[float] = Field(..., description="Similarity scores")
    total_results: int = Field(..., description="Total number of results")
    search_time: float = Field(..., description="Search time in seconds")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    stats: Dict[str, Any] = Field(..., description="System statistics")


class IngestionRequest(BaseModel):
    """Request model for document ingestion."""
    directory_path: str = Field(..., description="Path to documents directory")
    recursive: Optional[bool] = Field(True, description="Process subdirectories")
    force_reindex: Optional[bool] = Field(False, description="Force reprocessing all documents")
    batch_size: Optional[int] = Field(None, ge=1, le=100, description="Processing batch size")


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint."""
    task_id: str = Field(..., description="Background task ID")
    message: str = Field(..., description="Status message")
    estimated_time: Optional[float] = Field(None, description="Estimated completion time")


class IngestionStatus(BaseModel):
    """Response model for ingestion status."""
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    progress: Dict[str, Any] = Field(..., description="Processing progress")
    result: Optional[Dict[str, Any]] = Field(None, description="Final result if completed")


# Background task tracking
ingestion_tasks: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    global pipeline
    logger.info("Starting RAG API server...")
    
    try:
        pipeline = RAGPipeline()
        await pipeline.initialize()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API server...")
    if pipeline:
        # Cleanup if needed
        pass


# Create FastAPI app
app = FastAPI(
    title="RAG API Server",
    description="Production RAG API for document question-answering system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request monitoring middleware
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Middleware for request monitoring and logging."""
    # Generate request ID
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    # Create contextual logger
    ctx_logger = add_request_context(logger, request_id)
    
    # Start timing
    start_time = time.time()
    
    # Log request
    ctx_logger.info(f"{request.method} {request.url.path} - Request started")
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Track metrics
        track_request(
            method=request.method,
            endpoint=request.url.path,
            duration=duration,
            status_code=response.status_code
        )
        
        # Log response
        ctx_logger.info(f"{request.method} {request.url.path} - "
                       f"Completed in {duration:.3f}s with status {response.status_code}")
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Track error
        track_request(
            method=request.method,
            endpoint=request.url.path,
            duration=duration,
            status_code=500
        )
        
        # Log error
        ctx_logger.error(f"{request.method} {request.url.path} - "
                        f"Failed after {duration:.3f}s: {e}")
        
        # Re-raise the exception
        raise


# Dependency for pipeline access
async def get_pipeline() -> RAGPipeline:
    """Get pipeline instance."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return pipeline


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "RAG API Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(rag_pipeline: RAGPipeline = Depends(get_pipeline)):
    """Health check endpoint."""
    try:
        # Run comprehensive health checks
        health = await run_health_checks(include_detailed=True)
        
        # Get system stats
        stats = {}
        
        # Vector store stats
        if hasattr(rag_pipeline, 'vector_store'):
            vector_stats = rag_pipeline.vector_store.get_collection_stats()
            stats['vector_store'] = vector_stats
        
        # Retriever stats
        if hasattr(rag_pipeline, 'retriever'):
            retriever_stats = rag_pipeline.retriever.get_retrieval_stats()
            stats['retriever'] = retriever_stats
        
        # Generator stats
        if hasattr(rag_pipeline, 'generator'):
            generator_stats = rag_pipeline.generator.get_generation_stats()
            stats['generator'] = generator_stats
        
        # Add health check results
        stats['health_checks'] = {
            'overall_status': health.status.value,
            'checks': [
                {
                    'name': check.name,
                    'status': check.status.value,
                    'message': check.message,
                    'duration': check.duration
                }
                for check in health.checks
            ],
            'summary': health.summary
        }
        
        return HealthResponse(
            status=health.status.value,
            timestamp=health.timestamp.isoformat(),
            version="1.0.0",
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Query documents and generate answer."""
    try:
        start_time = time.time()
        
        # Execute query
        response = await rag_pipeline.query(
            question=request.query,
            top_k=request.k,
            retrieval_method="hybrid" if request.rerank else "vector",
            template_type=request.template_type,
            filters=request.filters
        )
        
        # Format response
        return QueryResponse(
            answer=response.answer,
            confidence_score=response.confidence_score,
            sources=response.sources,
            response_time=response.response_time,
            retrieval_stats={
                "chunks_retrieved": len(response.retrieval_result.chunks) if response.retrieval_result else 0,
                "retrieval_time": response.retrieval_result.retrieval_time if response.retrieval_result else 0,
                "method": response.retrieval_result.retrieval_method if response.retrieval_result else "unknown"
            }
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    rag_pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Search documents without answer generation."""
    try:
        start_time = time.time()
        
        # Perform search using retriever
        retrieval_result = await rag_pipeline.retriever.retrieve(
            query=request.query,
            k=request.k,
            rerank=False,  # Skip reranking for pure search
            filters=request.filters
        )
        
        # Format chunks for response
        chunks_data = []
        for chunk in retrieval_result.chunks:
            chunk_data = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "doc_id": chunk.doc_id,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char
            }
            chunks_data.append(chunk_data)
        
        return SearchResponse(
            chunks=chunks_data,
            scores=retrieval_result.scores,
            total_results=len(retrieval_result.chunks),
            search_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    rag_pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Start document ingestion process."""
    try:
        # Validate directory path
        directory_path = Path(request.directory_path)
        if not directory_path.exists():
            raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory_path}")
        
        # Generate task ID
        task_id = f"ingest_{int(time.time())}"
        
        # Store task info
        ingestion_tasks[task_id] = {
            "status": "started",
            "start_time": time.time(),
            "directory": str(directory_path),
            "recursive": request.recursive,
            "force_reindex": request.force_reindex,
            "progress": {"documents_processed": 0, "total_documents": 0, "chunks_created": 0}
        }
        
        # Start background task
        background_tasks.add_task(
            run_ingestion,
            task_id,
            rag_pipeline,
            directory_path,
            request.recursive,
            request.force_reindex,
            request.batch_size
        )
        
        return IngestionResponse(
            task_id=task_id,
            message="Document ingestion started",
            estimated_time=300.0  # Rough estimate: 5 minutes
        )
        
    except Exception as e:
        logger.error(f"Failed to start ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start ingestion: {str(e)}")


@app.get("/ingest/{task_id}", response_model=IngestionStatus)
async def get_ingestion_status(task_id: str = FastAPIPath(..., description="Task ID")):
    """Get ingestion task status."""
    if task_id not in ingestion_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = ingestion_tasks[task_id]
    
    return IngestionStatus(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        result=task_info.get("result")
    )


@app.get("/stats", response_model=Dict[str, Any])
async def get_system_stats(rag_pipeline: RAGPipeline = Depends(get_pipeline)):
    """Get comprehensive system statistics."""
    try:
        stats = {}
        
        # Vector store statistics
        if hasattr(rag_pipeline, 'vector_store'):
            stats['vector_store'] = rag_pipeline.vector_store.get_collection_stats()
        
        # Retriever statistics
        if hasattr(rag_pipeline, 'retriever'):
            stats['retriever'] = rag_pipeline.retriever.get_retrieval_stats()
        
        # Generator statistics
        if hasattr(rag_pipeline, 'generator'):
            stats['generator'] = rag_pipeline.generator.get_generation_stats()
        
        # Document processor statistics
        if hasattr(rag_pipeline, 'document_processor'):
            stats['document_processor'] = rag_pipeline.document_processor.get_processing_stats().__dict__
        
        # Active ingestion tasks
        stats['ingestion_tasks'] = {
            task_id: {
                "status": task_info["status"],
                "progress": task_info["progress"]
            }
            for task_id, task_info in ingestion_tasks.items()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str = FastAPIPath(..., description="Document ID"),
    rag_pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Delete a document and its chunks."""
    try:
        # This would need to be implemented in the pipeline
        # For now, return not implemented
        raise HTTPException(status_code=501, detail="Document deletion not implemented yet")
        
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.get("/documents", response_model=Dict[str, Any])
async def list_documents(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    rag_pipeline: RAGPipeline = Depends(get_pipeline)
):
    """List documents in the system."""
    try:
        # This would need to be implemented to query document metadata
        # For now, return basic info from vector store
        vector_stats = rag_pipeline.vector_store.get_collection_stats()
        
        return {
            "total_documents": "unknown",  # Would need document tracking
            "total_chunks": vector_stats.get("total_chunks", 0),
            "limit": limit,
            "offset": offset,
            "documents": []  # Would implement document listing
        }
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get comprehensive system metrics."""
    try:
        metrics_summary = get_metrics_summary()
        return metrics_summary
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/metrics/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """Get metrics in Prometheus format."""
    try:
        return export_prometheus_metrics()
    except Exception as e:
        logger.error(f"Failed to export Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")


@app.get("/health/detailed")
async def get_detailed_health():
    """Get detailed health information including history and trends."""
    try:
        from ..monitoring import get_health_history, get_health_trends
        
        # Get current health
        current_health = await run_health_checks(include_detailed=True)
        
        # Get trends
        trends = get_health_trends()
        
        # Get recent history
        history = get_health_history(hours=24)
        
        return {
            'current': {
                'status': current_health.status.value,
                'timestamp': current_health.timestamp.isoformat(),
                'checks': [
                    {
                        'name': check.name,
                        'status': check.status.value,
                        'message': check.message,
                        'details': check.details,
                        'duration': check.duration
                    }
                    for check in current_health.checks
                ],
                'summary': current_health.summary
            },
            'trends': trends,
            'history_24h': len(history),
            'history_sample': [
                {
                    'timestamp': h.timestamp.isoformat(),
                    'status': h.status.value,
                    'check_count': len(h.checks)
                }
                for h in history[-10:]  # Last 10 entries
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get detailed health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health details: {str(e)}")


# Background task function
async def run_ingestion(
    task_id: str,
    rag_pipeline: RAGPipeline,
    directory_path: Path,
    recursive: bool,
    force_reindex: bool,
    batch_size: Optional[int]
):
    """Run document ingestion in background."""
    try:
        # Update task status
        ingestion_tasks[task_id]["status"] = "processing"
        
        # Run ingestion
        await rag_pipeline.ingest_documents(
            directory_path,
            recursive=recursive,
            force_reindex=force_reindex,
            batch_size=batch_size,
            show_progress=False  # Don't show progress in background
        )
        
        # Get final stats
        stats = rag_pipeline.document_processor.get_processing_stats()
        
        # Update task with results
        ingestion_tasks[task_id].update({
            "status": "completed",
            "end_time": time.time(),
            "result": {
                "total_documents": stats.total_documents,
                "processed_documents": stats.processed_documents,
                "total_chunks": stats.total_chunks,
                "success_rate": stats.success_rate,
                "processing_time": stats.processing_time,
                "errors": stats.errors
            }
        })
        
        logger.info(f"Ingestion task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Ingestion task {task_id} failed: {e}")
        ingestion_tasks[task_id].update({
            "status": "failed",
            "end_time": time.time(),
            "error": str(e)
        })


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


# Server startup function
def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1
):
    """Start the FastAPI server."""
    config = uvicorn.Config(
        "rag_system.api.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    start_server()