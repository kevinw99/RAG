"""Health monitoring and status checking for RAG system.

Provides comprehensive health checks for all system components.
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .metrics import metrics_collector
from .logging_config import get_monitoring_logger
from ..config.settings import settings

logger = get_monitoring_logger()


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    duration: float
    timestamp: datetime


@dataclass 
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    checks: List[HealthCheck]
    timestamp: datetime
    summary: Dict[str, Any]


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.check_history: List[SystemHealth] = []
        self.max_history = 100
    
    async def run_health_checks(self, include_detailed: bool = True) -> SystemHealth:
        """Run all health checks.
        
        Args:
            include_detailed: Include detailed component checks
            
        Returns:
            Overall system health status
        """
        logger.debug("Running health checks")
        start_time = time.time()
        
        checks = []
        
        # Basic system checks
        checks.append(await self._check_system_resources())
        checks.append(await self._check_memory_usage())
        checks.append(await self._check_disk_space())
        
        if include_detailed:
            # Component-specific checks
            checks.append(await self._check_vector_store())
            checks.append(await self._check_embedding_model())
            checks.append(await self._check_llm_availability())
            checks.append(await self._check_metrics_system())
        
        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        
        # Create summary
        summary = self._create_health_summary(checks)
        
        health = SystemHealth(
            status=overall_status,
            checks=checks,
            timestamp=datetime.now(),
            summary=summary
        )
        
        # Store in history
        self.check_history.append(health)
        if len(self.check_history) > self.max_history:
            self.check_history = self.check_history[-self.max_history:]
        
        duration = time.time() - start_time
        logger.info(f"Health checks completed in {duration:.3f}s - Status: {overall_status.value}")
        
        return health
    
    async def _check_system_resources(self) -> HealthCheck:
        """Check system resource usage."""
        start_time = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Load average (Unix only)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                pass  # Windows doesn't have load average
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'load_average': load_avg
            }
            
            # Determine status
            if cpu_percent > 90 or memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            elif cpu_percent > 70 or memory_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Elevated resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Normal resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check system resources: {e}"
            details = {'error': str(e)}
        
        return HealthCheck(
            name="system_resources",
            status=status,
            message=message,
            details=details,
            duration=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    async def _check_memory_usage(self) -> HealthCheck:
        """Check RAG system memory usage."""
        start_time = time.time()
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            details = {
                'memory_mb': memory_mb,
                'max_memory_mb': settings.max_memory_usage_mb,
                'memory_percent': (memory_mb / settings.max_memory_usage_mb) * 100
            }
            
            # Determine status based on configured limits
            if memory_mb > settings.max_memory_usage_mb:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage exceeds limit: {memory_mb:.1f}MB > {settings.max_memory_usage_mb}MB"
            elif memory_mb > settings.max_memory_usage_mb * 0.8:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory_mb:.1f}MB ({(memory_mb/settings.max_memory_usage_mb)*100:.1f}%)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Normal memory usage: {memory_mb:.1f}MB"
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check memory usage: {e}"
            details = {'error': str(e)}
        
        return HealthCheck(
            name="memory_usage",
            status=status,
            message=message,
            details=details,
            duration=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    async def _check_disk_space(self) -> HealthCheck:
        """Check disk space availability."""
        start_time = time.time()
        
        try:
            # Check space for vector store
            vector_store_path = settings.vector_store_path
            disk_usage = psutil.disk_usage(vector_store_path)
            
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_percent = ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100
            
            details = {
                'free_gb': free_gb,
                'total_gb': total_gb,
                'used_percent': used_percent,
                'vector_store_path': str(vector_store_path)
            }
            
            # Determine status
            if free_gb < 1:  # Less than 1GB free
                status = HealthStatus.UNHEALTHY
                message = f"Very low disk space: {free_gb:.1f}GB free"
            elif free_gb < 5:  # Less than 5GB free
                status = HealthStatus.DEGRADED
                message = f"Low disk space: {free_gb:.1f}GB free"
            else:
                status = HealthStatus.HEALTHY
                message = f"Sufficient disk space: {free_gb:.1f}GB free"
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check disk space: {e}"
            details = {'error': str(e)}
        
        return HealthCheck(
            name="disk_space",
            status=status,
            message=message,
            details=details,
            duration=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    async def _check_vector_store(self) -> HealthCheck:
        """Check vector store connectivity and status."""
        start_time = time.time()
        
        try:
            # This would need to be passed in or imported from pipeline
            # For now, we'll create a basic check
            from ..storage.vector_store import create_vector_store
            
            vector_store = create_vector_store()
            stats = vector_store.get_collection_stats()
            
            chunk_count = stats.get('total_chunks', 0)
            
            details = {
                'total_chunks': chunk_count,
                'collection_name': stats.get('collection_name'),
                'embedding_model': stats.get('embedding_model'),
                'persist_directory': stats.get('persist_directory')
            }
            
            if chunk_count == 0:
                status = HealthStatus.DEGRADED
                message = "Vector store is empty - no documents indexed"
            elif chunk_count < 1000:
                status = HealthStatus.DEGRADED
                message = f"Low document count: {chunk_count} chunks"
            else:
                status = HealthStatus.HEALTHY
                message = f"Vector store healthy: {chunk_count:,} chunks"
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Vector store check failed: {e}"
            details = {'error': str(e)}
        
        return HealthCheck(
            name="vector_store",
            status=status,
            message=message,
            details=details,
            duration=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    async def _check_embedding_model(self) -> HealthCheck:
        """Check embedding model availability."""
        start_time = time.time()
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Try to load the model
            model = SentenceTransformer(settings.embedding_model)
            
            # Test with a simple sentence
            test_embedding = model.encode(["Health check test sentence"])
            
            details = {
                'model_name': settings.embedding_model,
                'embedding_dimension': len(test_embedding[0]),
                'model_loaded': True
            }
            
            status = HealthStatus.HEALTHY
            message = f"Embedding model healthy: {settings.embedding_model}"
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Embedding model check failed: {e}"
            details = {'error': str(e), 'model_name': settings.embedding_model}
        
        return HealthCheck(
            name="embedding_model",
            status=status,
            message=message,
            details=details,
            duration=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    async def _check_llm_availability(self) -> HealthCheck:
        """Check LLM availability for response generation."""
        start_time = time.time()
        
        try:
            # Check if API keys are configured
            api_key_configured = False
            
            if settings.llm_provider.lower() == "openai" and settings.openai_api_key:
                api_key_configured = True
            elif settings.llm_provider.lower() == "anthropic" and settings.anthropic_api_key:
                api_key_configured = True
            
            details = {
                'llm_provider': settings.llm_provider,
                'llm_model': settings.llm_model,
                'api_key_configured': api_key_configured
            }
            
            if not api_key_configured:
                status = HealthStatus.DEGRADED
                message = f"LLM API key not configured for {settings.llm_provider}"
            else:
                # Could add actual API test here
                status = HealthStatus.HEALTHY
                message = f"LLM configured: {settings.llm_provider}/{settings.llm_model}"
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"LLM check failed: {e}"
            details = {'error': str(e)}
        
        return HealthCheck(
            name="llm_availability",
            status=status,
            message=message,
            details=details,
            duration=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    async def _check_metrics_system(self) -> HealthCheck:
        """Check metrics collection system."""
        start_time = time.time()
        
        try:
            # Check metrics collector
            summary = metrics_collector.get_metrics_summary()
            
            details = {
                'total_events': summary.get('total_events', 0),
                'recent_events_1h': summary.get('recent_events_1h', 0),
                'prometheus_enabled': metrics_collector.enable_prometheus,
                'counters_count': len(summary.get('counters', {})),
                'gauges_count': len(summary.get('gauges', {}))
            }
            
            if summary.get('total_events', 0) == 0:
                status = HealthStatus.DEGRADED
                message = "No metrics events recorded"
            else:
                status = HealthStatus.HEALTHY
                message = f"Metrics system healthy: {details['total_events']} events"
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Metrics system check failed: {e}"
            details = {'error': str(e)}
        
        return HealthCheck(
            name="metrics_system",
            status=status,
            message=message,
            details=details,
            duration=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system status from individual checks."""
        if not checks:
            return HealthStatus.UNKNOWN
        
        # Count status types
        status_counts = {status: 0 for status in HealthStatus}
        for check in checks:
            status_counts[check.status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            return HealthStatus.DEGRADED  # Treat unknown as degraded
        else:
            return HealthStatus.HEALTHY
    
    def _create_health_summary(self, checks: List[HealthCheck]) -> Dict[str, Any]:
        """Create health summary from checks."""
        status_counts = {status.value: 0 for status in HealthStatus}
        for check in checks:
            status_counts[check.status.value] += 1
        
        return {
            'total_checks': len(checks),
            'status_counts': status_counts,
            'avg_check_duration': sum(c.duration for c in checks) / len(checks) if checks else 0,
            'failed_checks': [c.name for c in checks if c.status in [HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]],
            'degraded_checks': [c.name for c in checks if c.status == HealthStatus.DEGRADED]
        }
    
    def get_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """Get health check history.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of health check results
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [h for h in self.check_history if h.timestamp > cutoff_time]
    
    def get_health_trends(self) -> Dict[str, Any]:
        """Get health trends and analytics."""
        if not self.check_history:
            return {'error': 'No health history available'}
        
        recent_checks = self.check_history[-10:]  # Last 10 checks
        
        # Calculate trends
        healthy_rate = sum(1 for h in recent_checks if h.status == HealthStatus.HEALTHY) / len(recent_checks)
        avg_check_count = sum(len(h.checks) for h in recent_checks) / len(recent_checks)
        
        # Component reliability
        component_stats = {}
        for health in recent_checks:
            for check in health.checks:
                if check.name not in component_stats:
                    component_stats[check.name] = {'total': 0, 'healthy': 0}
                component_stats[check.name]['total'] += 1
                if check.status == HealthStatus.HEALTHY:
                    component_stats[check.name]['healthy'] += 1
        
        for comp, stats in component_stats.items():
            stats['reliability'] = stats['healthy'] / stats['total'] if stats['total'] > 0 else 0
        
        return {
            'healthy_rate': healthy_rate,
            'avg_checks_per_run': avg_check_count,
            'component_reliability': component_stats,
            'history_count': len(self.check_history),
            'latest_status': self.check_history[-1].status.value if self.check_history else 'unknown'
        }


# Global health monitor instance
health_monitor = HealthMonitor()


# Convenience functions
async def run_health_checks(include_detailed: bool = True) -> SystemHealth:
    """Run comprehensive health checks."""
    return await health_monitor.run_health_checks(include_detailed)


async def quick_health_check() -> Tuple[HealthStatus, str]:
    """Quick health check returning just status and message."""
    health = await health_monitor.run_health_checks(include_detailed=False)
    return health.status, f"{len(health.checks)} checks completed"


def get_health_history(hours: int = 24) -> List[SystemHealth]:
    """Get health check history."""
    return health_monitor.get_health_history(hours)


def get_health_trends() -> Dict[str, Any]:
    """Get health trends."""
    return health_monitor.get_health_trends()