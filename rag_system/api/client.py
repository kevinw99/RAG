"""Python client SDK for RAG API server.

Provides easy access to RAG API endpoints with async support.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import json

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class APIError(Exception):
    """API error exception."""
    def __init__(self, message: str, status_code: int = None, response: Dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class RAGClient:
    """Async client for RAG API server."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 timeout: float = 60.0,
                 api_key: Optional[str] = None):
        """Initialize RAG client.
        
        Args:
            base_url: Base URL of the RAG API server
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Setup headers
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Create HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers
        )
        
        logger.info(f"RAGClient initialized with base_url: {self.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def _make_request(self, 
                           method: str, 
                           endpoint: str, 
                           data: Dict = None,
                           params: Dict = None) -> Dict:
        """Make HTTP request to API."""
        try:
            url = f"{endpoint}"
            
            if method.upper() == "GET":
                response = await self.client.get(url, params=params)
            elif method.upper() == "POST":
                response = await self.client.post(url, json=data, params=params)
            elif method.upper() == "DELETE":
                response = await self.client.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check for HTTP errors
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", f"HTTP {response.status_code}")
                except:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                
                raise APIError(error_msg, response.status_code, error_data if 'error_data' in locals() else None)
            
            return response.json()
            
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API server health.
        
        Returns:
            Health status and statistics
        """
        return await self._make_request("GET", "/health")
    
    async def query(self,
                   query: str,
                   k: int = 10,
                   rerank: bool = True,
                   template_type: str = "default",
                   filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query documents and get answer.
        
        Args:
            query: User question
            k: Number of chunks to retrieve
            rerank: Enable cross-encoder reranking
            template_type: Response template type
            filters: Metadata filters
            
        Returns:
            Generated answer with sources and confidence
        """
        data = {
            "query": query,
            "k": k,
            "rerank": rerank,
            "template_type": template_type
        }
        
        if filters:
            data["filters"] = filters
        
        return await self._make_request("POST", "/query", data)
    
    async def search(self,
                    query: str,
                    k: int = 10,
                    search_type: str = "hybrid",
                    filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search documents without answer generation.
        
        Args:
            query: Search query
            k: Number of results
            search_type: Type of search (vector, keyword, hybrid)
            filters: Metadata filters
            
        Returns:
            Retrieved chunks with scores
        """
        data = {
            "query": query,
            "k": k,
            "search_type": search_type
        }
        
        if filters:
            data["filters"] = filters
        
        return await self._make_request("POST", "/search", data)
    
    async def ingest_documents(self,
                              directory_path: Union[str, Path],
                              recursive: bool = True,
                              force_reindex: bool = False,
                              batch_size: Optional[int] = None) -> Dict[str, str]:
        """Start document ingestion process.
        
        Args:
            directory_path: Path to documents directory
            recursive: Process subdirectories
            force_reindex: Force reprocessing all documents
            batch_size: Processing batch size
            
        Returns:
            Task information
        """
        data = {
            "directory_path": str(directory_path),
            "recursive": recursive,
            "force_reindex": force_reindex
        }
        
        if batch_size:
            data["batch_size"] = batch_size
        
        return await self._make_request("POST", "/ingest", data)
    
    async def get_ingestion_status(self, task_id: str) -> Dict[str, Any]:
        """Get ingestion task status.
        
        Args:
            task_id: Ingestion task ID
            
        Returns:
            Task status and progress
        """
        return await self._make_request("GET", f"/ingest/{task_id}")
    
    async def wait_for_ingestion(self, 
                                task_id: str, 
                                check_interval: float = 5.0,
                                max_wait: float = 3600.0) -> Dict[str, Any]:
        """Wait for ingestion task to complete.
        
        Args:
            task_id: Task ID to wait for
            check_interval: Status check interval in seconds
            max_wait: Maximum wait time in seconds
            
        Returns:
            Final task result
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check if max wait time exceeded
            if asyncio.get_event_loop().time() - start_time > max_wait:
                raise APIError(f"Ingestion task {task_id} exceeded maximum wait time")
            
            # Get task status
            status = await self.get_ingestion_status(task_id)
            
            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                error_msg = status.get("error", "Ingestion failed")
                raise APIError(f"Ingestion task failed: {error_msg}")
            
            # Wait before next check
            await asyncio.sleep(check_interval)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics.
        
        Returns:
            Comprehensive system statistics
        """
        return await self._make_request("GET", "/stats")
    
    async def list_documents(self, 
                           limit: int = 100, 
                           offset: int = 0) -> Dict[str, Any]:
        """List documents in the system.
        
        Args:
            limit: Maximum number of documents
            offset: Offset for pagination
            
        Returns:
            Document list with pagination info
        """
        params = {"limit": limit, "offset": offset}
        return await self._make_request("GET", "/documents", params=params)
    
    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Deletion result
        """
        return await self._make_request("DELETE", f"/documents/{doc_id}")


# Convenience functions for synchronous usage
class SyncRAGClient:
    """Synchronous wrapper for RAG client."""
    
    def __init__(self, **kwargs):
        """Initialize sync client."""
        self._async_client = RAGClient(**kwargs)
        self._loop = None
    
    def _get_loop(self):
        """Get or create event loop."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            return self._loop
    
    def _run_async(self, coro):
        """Run async coroutine."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)
    
    def health_check(self) -> Dict[str, Any]:
        """Sync health check."""
        return self._run_async(self._async_client.health_check())
    
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Sync query."""
        return self._run_async(self._async_client.query(query, **kwargs))
    
    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Sync search."""
        return self._run_async(self._async_client.search(query, **kwargs))
    
    def ingest_documents(self, directory_path: Union[str, Path], **kwargs) -> Dict[str, str]:
        """Sync document ingestion."""
        return self._run_async(self._async_client.ingest_documents(directory_path, **kwargs))
    
    def get_ingestion_status(self, task_id: str) -> Dict[str, Any]:
        """Sync ingestion status."""
        return self._run_async(self._async_client.get_ingestion_status(task_id))
    
    def wait_for_ingestion(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """Sync wait for ingestion."""
        return self._run_async(self._async_client.wait_for_ingestion(task_id, **kwargs))
    
    def get_stats(self) -> Dict[str, Any]:
        """Sync get stats."""
        return self._run_async(self._async_client.get_stats())
    
    def list_documents(self, **kwargs) -> Dict[str, Any]:
        """Sync list documents."""
        return self._run_async(self._async_client.list_documents(**kwargs))
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Sync delete document."""
        return self._run_async(self._async_client.delete_document(doc_id))
    
    def close(self):
        """Close client."""
        self._run_async(self._async_client.close())
        if self._loop:
            self._loop.close()


# Example usage functions
async def example_usage():
    """Example usage of RAG client."""
    async with RAGClient() as client:
        # Check health
        health = await client.health_check()
        print(f"Server health: {health['status']}")
        
        # Query documents
        result = await client.query(
            query="What are the key retirement planning strategies?",
            k=5,
            rerank=True
        )
        
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print(f"Sources: {len(result['sources'])}")
        
        # Search documents
        search_result = await client.search(
            query="social security optimization",
            k=10,
            search_type="hybrid"
        )
        
        print(f"Found {search_result['total_results']} relevant chunks")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())