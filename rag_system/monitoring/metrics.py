"""Metrics collection and monitoring system for RAG.

Provides comprehensive metrics collection using Prometheus format.
Tracks performance, usage, and system health metrics.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available - metrics will be stored in memory only")

from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class MetricEvent:
    """Individual metric event."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemStats:
    """System statistics snapshot."""
    timestamp: datetime
    memory_usage_mb: float
    cpu_percent: float
    active_connections: int
    total_requests: int
    error_rate: float
    avg_response_time: float


class MetricsCollector:
    """Centralized metrics collection system."""
    
    def __init__(self, enable_prometheus: bool = True):
        """Initialize metrics collector.
        
        Args:
            enable_prometheus: Enable Prometheus metrics export
        """
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.registry = CollectorRegistry() if self.enable_prometheus else None
        
        # In-memory storage for metrics (fallback and additional analytics)
        self.events: deque = deque(maxlen=10000)  # Keep last 10k events
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # System stats history
        self.system_stats_history: deque = deque(maxlen=1000)  # Keep last 1000 entries
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize Prometheus metrics if available
        if self.enable_prometheus:
            self._init_prometheus_metrics()
        
        logger.info(f"MetricsCollector initialized (Prometheus: {self.enable_prometheus})")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        if not self.enable_prometheus:
            return
        
        # Request metrics
        self.prom_request_count = Counter(
            'rag_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.prom_request_duration = Histogram(
            'rag_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Document processing metrics
        self.prom_documents_processed = Counter(
            'rag_documents_processed_total',
            'Total documents processed',
            ['status'],
            registry=self.registry
        )
        
        self.prom_chunks_created = Counter(
            'rag_chunks_created_total',
            'Total chunks created',
            registry=self.registry
        )
        
        # Query metrics
        self.prom_queries_total = Counter(
            'rag_queries_total',
            'Total queries processed',
            ['retrieval_method', 'template_type'],
            registry=self.registry
        )
        
        self.prom_query_duration = Histogram(
            'rag_query_duration_seconds',
            'Query processing duration',
            ['retrieval_method'],
            registry=self.registry
        )
        
        self.prom_retrieval_duration = Histogram(
            'rag_retrieval_duration_seconds',
            'Retrieval duration',
            ['method'],
            registry=self.registry
        )
        
        self.prom_generation_duration = Histogram(
            'rag_generation_duration_seconds',
            'Response generation duration',
            registry=self.registry
        )
        
        # System metrics
        self.prom_memory_usage = Gauge(
            'rag_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.prom_active_connections = Gauge(
            'rag_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.prom_vector_store_size = Gauge(
            'rag_vector_store_chunks',
            'Number of chunks in vector store',
            registry=self.registry
        )
        
        # Quality metrics
        self.prom_confidence_score = Histogram(
            'rag_confidence_score',
            'Response confidence scores',
            registry=self.registry
        )
        
        self.prom_retrieval_score = Histogram(
            'rag_retrieval_score',
            'Retrieval confidence scores',
            registry=self.registry
        )
        
        # System info
        self.prom_info = Info(
            'rag_system_info',
            'RAG system information',
            registry=self.registry
        )
        
        # Set system info
        self.prom_info.info({
            'version': '1.0.0',
            'embedding_model': settings.embedding_model,
            'llm_provider': settings.llm_provider,
            'llm_model': settings.llm_model,
            'collection_name': settings.collection_name
        })
    
    def record_event(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a metric event.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
        """
        labels = labels or {}
        event = MetricEvent(name=name, value=value, labels=labels)
        
        with self._lock:
            self.events.append(event)
            
            # Update in-memory counters
            label_key = "_".join(f"{k}:{v}" for k, v in sorted(labels.items()))
            metric_key = f"{name}_{label_key}" if label_key else name
            self.counters[metric_key] += value
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value.
        
        Args:
            name: Gauge name
            value: Current value
            labels: Optional labels
        """
        labels = labels or {}
        label_key = "_".join(f"{k}:{v}" for k, v in sorted(labels.items()))
        metric_key = f"{name}_{label_key}" if label_key else name
        
        with self._lock:
            self.gauges[metric_key] = value
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record histogram value.
        
        Args:
            name: Histogram name
            value: Value to record
            labels: Optional labels
        """
        labels = labels or {}
        label_key = "_".join(f"{k}:{v}" for k, v in sorted(labels.items()))
        metric_key = f"{name}_{label_key}" if label_key else name
        
        with self._lock:
            self.histograms[metric_key].append(value)
            # Keep only last 1000 values per histogram
            if len(self.histograms[metric_key]) > 1000:
                self.histograms[metric_key] = self.histograms[metric_key][-1000:]
    
    # High-level tracking methods
    def track_request(self, method: str, endpoint: str, duration: float, status_code: int):
        """Track HTTP request metrics."""
        labels = {
            'method': method,
            'endpoint': endpoint,
            'status': str(status_code)
        }
        
        self.record_event('requests_total', labels=labels)
        self.record_histogram('request_duration_seconds', duration, 
                            {'method': method, 'endpoint': endpoint})
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.prom_request_count.labels(**labels).inc()
            self.prom_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def track_query(self, retrieval_method: str, template_type: str, 
                   query_duration: float, retrieval_duration: float,
                   generation_duration: float, confidence_score: float):
        """Track query processing metrics."""
        labels = {
            'retrieval_method': retrieval_method,
            'template_type': template_type
        }
        
        self.record_event('queries_total', labels=labels)
        self.record_histogram('query_duration_seconds', query_duration, labels)
        self.record_histogram('confidence_score', confidence_score)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.prom_queries_total.labels(**labels).inc()
            self.prom_query_duration.labels(retrieval_method=retrieval_method).observe(query_duration)
            self.prom_retrieval_duration.labels(method=retrieval_method).observe(retrieval_duration)
            self.prom_generation_duration.observe(generation_duration)
            self.prom_confidence_score.observe(confidence_score)
    
    def track_document_processing(self, documents_processed: int, chunks_created: int,
                                processing_time: float, success_count: int, error_count: int):
        """Track document processing metrics."""
        self.record_event('documents_processed', documents_processed)
        self.record_event('chunks_created', chunks_created)
        self.record_histogram('processing_duration_seconds', processing_time)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.prom_documents_processed.labels(status='success').inc(success_count)
            self.prom_documents_processed.labels(status='error').inc(error_count)
            self.prom_chunks_created.inc(chunks_created)
    
    def update_system_stats(self, memory_mb: float, cpu_percent: float, 
                          active_connections: int, vector_store_size: int):
        """Update system statistics."""
        self.set_gauge('memory_usage_mb', memory_mb)
        self.set_gauge('cpu_percent', cpu_percent)
        self.set_gauge('active_connections', active_connections)
        self.set_gauge('vector_store_chunks', vector_store_size)
        
        # Store system stats history
        stats = SystemStats(
            timestamp=datetime.now(),
            memory_usage_mb=memory_mb,
            cpu_percent=cpu_percent,
            active_connections=active_connections,
            total_requests=self.counters.get('requests_total', 0),
            error_rate=0.0,  # Would calculate from error metrics
            avg_response_time=0.0  # Would calculate from duration metrics
        )
        
        with self._lock:
            self.system_stats_history.append(stats)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.prom_memory_usage.set(memory_mb * 1024 * 1024)  # Convert to bytes
            self.prom_active_connections.set(active_connections)
            self.prom_vector_store_size.set(vector_store_size)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self._lock:
            now = datetime.now()
            
            # Recent events (last hour)
            hour_ago = now - timedelta(hours=1)
            recent_events = [e for e in self.events if e.timestamp > hour_ago]
            
            # System stats (last 10 entries)
            recent_stats = list(self.system_stats_history)[-10:]
            
            summary = {
                'timestamp': now.isoformat(),
                'total_events': len(self.events),
                'recent_events_1h': len(recent_events),
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {
                    name: {
                        'count': len(values),
                        'min': min(values) if values else 0,
                        'max': max(values) if values else 0,
                        'avg': sum(values) / len(values) if values else 0,
                        'p95': sorted(values)[int(len(values) * 0.95)] if values else 0
                    }
                    for name, values in self.histograms.items()
                },
                'system_stats': {
                    'current': recent_stats[-1].__dict__ if recent_stats else None,
                    'history_count': len(self.system_stats_history)
                }
            }
            
            return summary
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        if not self.enable_prometheus:
            return "# Prometheus metrics not enabled\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health-related metrics."""
        with self._lock:
            recent_stats = list(self.system_stats_history)[-5:]  # Last 5 entries
            
            if not recent_stats:
                return {'status': 'unknown', 'metrics': {}}
            
            latest = recent_stats[-1]
            
            # Determine health status
            status = 'healthy'
            if latest.memory_usage_mb > settings.max_memory_usage_mb * 0.9:
                status = 'warning'
            if latest.memory_usage_mb > settings.max_memory_usage_mb:
                status = 'critical'
            
            return {
                'status': status,
                'metrics': {
                    'memory_usage_mb': latest.memory_usage_mb,
                    'cpu_percent': latest.cpu_percent,
                    'active_connections': latest.active_connections,
                    'total_requests': latest.total_requests,
                    'avg_memory_5m': sum(s.memory_usage_mb for s in recent_stats) / len(recent_stats),
                    'request_rate_1h': len([e for e in self.events 
                                          if e.timestamp > datetime.now() - timedelta(hours=1) 
                                          and e.name == 'requests_total'])
                }
            }


# Global metrics collector instance
metrics_collector = MetricsCollector(enable_prometheus=PROMETHEUS_AVAILABLE)


# Convenience functions
def track_request(method: str, endpoint: str, duration: float, status_code: int):
    """Track HTTP request."""
    metrics_collector.track_request(method, endpoint, duration, status_code)


def track_query(retrieval_method: str, template_type: str, query_duration: float,
               retrieval_duration: float, generation_duration: float, confidence_score: float):
    """Track query processing."""
    metrics_collector.track_query(
        retrieval_method, template_type, query_duration, 
        retrieval_duration, generation_duration, confidence_score
    )


def track_document_processing(documents_processed: int, chunks_created: int,
                            processing_time: float, success_count: int, error_count: int):
    """Track document processing."""
    metrics_collector.track_document_processing(
        documents_processed, chunks_created, processing_time, success_count, error_count
    )


def update_system_stats(memory_mb: float, cpu_percent: float, 
                       active_connections: int, vector_store_size: int):
    """Update system statistics."""
    metrics_collector.update_system_stats(memory_mb, cpu_percent, active_connections, vector_store_size)


def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary."""
    return metrics_collector.get_metrics_summary()


def export_prometheus_metrics() -> str:
    """Export Prometheus metrics."""
    return metrics_collector.export_prometheus_metrics()


def get_health_metrics() -> Dict[str, Any]:
    """Get health metrics."""
    return metrics_collector.get_health_metrics()