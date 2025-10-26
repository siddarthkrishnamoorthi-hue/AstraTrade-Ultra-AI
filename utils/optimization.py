"""
Optimization utilities for AstraTrade Ultra AI components
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler
import concurrent.futures
import asyncio
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    parameters: Dict[str, Any]
    performance_score: float
    execution_time: float
    memory_usage: float

class ComponentOptimizer:
    """Optimizer for trading components"""
    
    def __init__(self):
        self.param_ranges = {
            'batch_size': (32, 512),
            'window_size': (100, 1000),
            'confidence_threshold': (0.5, 0.95),
            'update_frequency': (10, 100)
        }
        self.scaler = StandardScaler()
        
    @lru_cache(maxsize=1000)
    def get_optimal_parameters(self, component_type: str, 
                             performance_history: Tuple[float, ...]) -> Dict[str, Any]:
        """Get optimal parameters based on performance history"""
        if not performance_history:
            return self._get_default_params(component_type)
            
        # Convert history to numpy array
        history = np.array(performance_history)
        
        # Calculate optimization metrics
        volatility = np.std(history)
        trend = np.mean(history[-10:]) - np.mean(history[:10])
        
        return self._optimize_params(component_type, volatility, trend)
        
    def _optimize_params(self, component_type: str, 
                        volatility: float, trend: float) -> Dict[str, Any]:
        """Optimize parameters based on market conditions"""
        if component_type == 'adaptive_agent':
            return {
                'batch_size': self._optimize_batch_size(volatility),
                'window_size': self._optimize_window_size(volatility, trend),
                'confidence_threshold': self._optimize_confidence(volatility),
                'update_frequency': self._optimize_update_freq(volatility)
            }
        elif component_type == 'ml_ensemble':
            return {
                'batch_size': self._optimize_batch_size(volatility),
                'ensemble_weights': self._optimize_ensemble_weights(trend),
                'learning_rate': self._optimize_learning_rate(volatility)
            }
        else:
            return self._get_default_params(component_type)
            
    def _optimize_batch_size(self, volatility: float) -> int:
        """Optimize batch size based on market volatility"""
        base_size = 64
        if volatility > 0.2:
            return max(32, int(base_size * (1 - volatility)))
        else:
            return min(512, int(base_size * (1 + volatility)))
            
    def _optimize_window_size(self, volatility: float, trend: float) -> int:
        """Optimize analysis window size"""
        base_size = 500
        if abs(trend) > 0.1:  # Strong trend
            return int(base_size * (1 - abs(trend)))
        else:  # Range-bound
            return int(base_size * (1 + volatility))
            
    def _optimize_confidence(self, volatility: float) -> float:
        """Optimize confidence threshold"""
        base_confidence = 0.7
        return min(0.95, base_confidence + (volatility * 0.2))
        
    def _optimize_update_freq(self, volatility: float) -> int:
        """Optimize model update frequency"""
        base_freq = 50
        return max(10, int(base_freq * (1 - volatility)))
        
    def _optimize_ensemble_weights(self, trend: float) -> List[float]:
        """Optimize ML ensemble weights"""
        if abs(trend) > 0.1:  # Strong trend
            lstm_weight = 0.7
        else:  # Range-bound
            lstm_weight = 0.3
            
        return [lstm_weight, 1 - lstm_weight]
        
    def _optimize_learning_rate(self, volatility: float) -> float:
        """Optimize learning rate based on volatility"""
        base_lr = 0.001
        return base_lr * (1 + volatility)
        
    def _get_default_params(self, component_type: str) -> Dict[str, Any]:
        """Get default parameters for component"""
        defaults = {
            'adaptive_agent': {
                'batch_size': 64,
                'window_size': 500,
                'confidence_threshold': 0.7,
                'update_frequency': 50
            },
            'ml_ensemble': {
                'batch_size': 64,
                'ensemble_weights': [0.5, 0.5],
                'learning_rate': 0.001
            }
        }
        return defaults.get(component_type, {})

class MemoryOptimizer:
    """Memory usage optimization"""
    
    def __init__(self):
        self.chunk_size = 1000
        self.max_cache_size = 10000
        
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                if df[col].nunique() / len(df) < 0.5:  # High cardinality
                    df[col] = df[col].astype('category')
            elif col_type == 'float64':
                df[col] = df[col].astype('float32')
            elif col_type == 'int64':
                df[col] = df[col].astype('int32')
                
        return df
        
    def chunk_processor(self, data: List[Any], process_func: callable) -> List[Any]:
        """Process large datasets in chunks"""
        results = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            chunk_result = process_func(chunk)
            results.extend(chunk_result)
        return results
        
    async def async_chunk_processor(self, data: List[Any], 
                                  process_func: callable) -> List[Any]:
        """Process chunks asynchronously"""
        tasks = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            task = asyncio.create_task(process_func(chunk))
            tasks.append(task)
            
        results = []
        for completed_task in asyncio.as_completed(tasks):
            chunk_result = await completed_task
            results.extend(chunk_result)
            
        return results

class ConcurrencyOptimizer:
    """Optimize concurrent operations"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or (concurrent.futures.cpu_count() * 2)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
    def parallel_map(self, func: callable, data: List[Any]) -> List[Any]:
        """Execute function on data in parallel"""
        return list(self.executor.map(func, data))
        
    async def gather_with_concurrency(self, n: int, *tasks):
        """Run tasks with limited concurrency"""
        semaphore = asyncio.Semaphore(n)
        
        async def sem_task(task):
            async with semaphore:
                return await task
                
        return await asyncio.gather(*(sem_task(task) for task in tasks))
        
    def cleanup(self):
        """Cleanup executor resources"""
        self.executor.shutdown()