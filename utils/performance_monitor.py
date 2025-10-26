"""
Performance monitoring and optimization utilities for AstraTrade Ultra AI
"""
import time
import functools
import logging
import psutil
import numpy as np
from typing import Dict, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
import cProfile
import pstats
import io

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    latency: float

class PerformanceMonitor:
    """Performance monitoring and optimization toolkit"""
    
    def __init__(self):
        self.metrics_history: Dict[str, list] = {}
        self.threshold_alerts = {
            'execution_time': 1.0,  # seconds
            'memory_usage': 85.0,   # percent
            'cpu_usage': 80.0,      # percent
            'latency': 0.5          # seconds
        }
        
    def performance_decorator(self, component_name: str) -> Callable:
        """Decorator to monitor performance of components"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_percent()
                start_cpu = psutil.cpu_percent()
                
                try:
                    result = func(*args, **kwargs)
                    
                    metrics = PerformanceMetrics(
                        execution_time=time.perf_counter() - start_time,
                        memory_usage=psutil.Process().memory_percent() - start_memory,
                        cpu_usage=psutil.cpu_percent() - start_cpu,
                        latency=0.0  # Will be updated for async operations
                    )
                    
                    self._record_metrics(component_name, metrics)
                    self._check_thresholds(component_name, metrics)
                    
                    return result
                except Exception as e:
                    logger.error(f"Performance monitoring failed for {component_name}: {str(e)}")
                    raise
                    
            return wrapper
        return decorator
        
    def async_performance_decorator(self, component_name: str) -> Callable:
        """Decorator for async component performance monitoring"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_percent()
                start_cpu = psutil.cpu_percent()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    metrics = PerformanceMetrics(
                        execution_time=time.perf_counter() - start_time,
                        memory_usage=psutil.Process().memory_percent() - start_memory,
                        cpu_usage=psutil.cpu_percent() - start_cpu,
                        latency=time.perf_counter() - start_time
                    )
                    
                    self._record_metrics(component_name, metrics)
                    self._check_thresholds(component_name, metrics)
                    
                    return result
                except Exception as e:
                    logger.error(f"Async performance monitoring failed for {component_name}: {str(e)}")
                    raise
                    
            return wrapper
        return decorator
        
    def _record_metrics(self, component_name: str, metrics: PerformanceMetrics) -> None:
        """Record performance metrics"""
        if component_name not in self.metrics_history:
            self.metrics_history[component_name] = []
        self.metrics_history[component_name].append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics_history[component_name]) > 1000:
            self.metrics_history[component_name].pop(0)
            
    def _check_thresholds(self, component_name: str, metrics: PerformanceMetrics) -> None:
        """Check if metrics exceed thresholds"""
        if metrics.execution_time > self.threshold_alerts['execution_time']:
            logger.warning(f"{component_name} execution time exceeded threshold: {metrics.execution_time:.2f}s")
            
        if metrics.memory_usage > self.threshold_alerts['memory_usage']:
            logger.warning(f"{component_name} memory usage exceeded threshold: {metrics.memory_usage:.2f}%")
            
        if metrics.cpu_usage > self.threshold_alerts['cpu_usage']:
            logger.warning(f"{component_name} CPU usage exceeded threshold: {metrics.cpu_usage:.2f}%")
            
        if metrics.latency > self.threshold_alerts['latency']:
            logger.warning(f"{component_name} latency exceeded threshold: {metrics.latency:.2f}s")
            
    def get_component_stats(self, component_name: str) -> Dict[str, float]:
        """Get statistical analysis of component performance"""
        if component_name not in self.metrics_history:
            return {}
            
        metrics = self.metrics_history[component_name]
        if not metrics:
            return {}
            
        execution_times = [m.execution_time for m in metrics]
        memory_usages = [m.memory_usage for m in metrics]
        cpu_usages = [m.cpu_usage for m in metrics]
        latencies = [m.latency for m in metrics]
        
        return {
            'execution_time_mean': np.mean(execution_times),
            'execution_time_std': np.std(execution_times),
            'execution_time_95th': np.percentile(execution_times, 95),
            'memory_usage_mean': np.mean(memory_usages),
            'cpu_usage_mean': np.mean(cpu_usages),
            'latency_mean': np.mean(latencies),
            'latency_95th': np.percentile(latencies, 95)
        }
        
    def profile_component(self, func: Callable, *args, **kwargs) -> str:
        """Profile a component's execution"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            func(*args, **kwargs)
        finally:
            profiler.disable()
            
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        stats.print_stats()
        
        return s.getvalue()
        
    def optimize_batch_size(self, func: Callable, data: Any, 
                          min_batch: int = 32, max_batch: int = 512) -> int:
        """Find optimal batch size for processing"""
        batch_sizes = [min_batch * (2**i) for i in range(5) if min_batch * (2**i) <= max_batch]
        times = []
        
        for batch_size in batch_sizes:
            start_time = time.perf_counter()
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                func(batch)
                
            times.append(time.perf_counter() - start_time)
            
        # Return batch size with minimum processing time
        return batch_sizes[np.argmin(times)]
        
    def parallel_process(self, func: Callable, data: list, max_workers: int = None) -> list:
        """Process data in parallel"""
        if max_workers is None:
            max_workers = psutil.cpu_count()
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(func, data))
            
    async def async_batch_process(self, func: Callable, data: list, 
                                batch_size: int = 100) -> list:
        """Process data in async batches"""
        results = []
        tasks = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            task = asyncio.create_task(func(batch))
            tasks.append(task)
            
        batch_results = await asyncio.gather(*tasks)
        for batch_result in batch_results:
            results.extend(batch_result)
            
        return results

# Global performance monitor instance
monitor = PerformanceMonitor()