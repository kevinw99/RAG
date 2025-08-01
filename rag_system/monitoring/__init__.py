"""Monitoring and observability module for RAG system.

Provides metrics collection, logging, and health monitoring capabilities.
"""

from .metrics import (
    metrics_collector,
    track_request,
    track_query,
    track_document_processing,
    update_system_stats,
    get_metrics_summary,
    export_prometheus_metrics,
    get_health_metrics
)

from .logging_config import (
    configure_logging,
    get_logger,
    get_component_logger,
    add_request_context,
    get_api_logger,
    get_core_logger,
    get_storage_logger,
    get_retrieval_logger,
    get_generation_logger,
    get_monitoring_logger
)

from .health import (
    health_monitor,
    run_health_checks,
    quick_health_check,
    get_health_history,
    get_health_trends,
    HealthStatus,
    HealthCheck,
    SystemHealth
)

__all__ = [
    # Metrics
    'metrics_collector',
    'track_request',
    'track_query', 
    'track_document_processing',
    'update_system_stats',
    'get_metrics_summary',
    'export_prometheus_metrics',
    'get_health_metrics',
    
    # Logging
    'configure_logging',
    'get_logger',
    'get_component_logger',
    'add_request_context',
    'get_api_logger',
    'get_core_logger',
    'get_storage_logger', 
    'get_retrieval_logger',
    'get_generation_logger',
    'get_monitoring_logger',
    
    # Health monitoring
    'health_monitor',
    'run_health_checks',
    'quick_health_check',
    'get_health_history',
    'get_health_trends',
    'HealthStatus',
    'HealthCheck',
    'SystemHealth'
]