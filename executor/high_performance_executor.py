"""
High-performance live execution handler with advanced order routing
"""

import MetaTrader5 as mt5
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
from .execution_strategies import (
    ExecutionStrategy,
    AggressiveStrategy,
    BalancedStrategy,
    ConservativeStrategy
)

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    
class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class ExecutionConfig:
    max_slippage: float = 2.0  # pips
    retry_attempts: int = 3
    retry_delay: float = 0.1  # seconds
    max_spread: float = 5.0  # pips
    execution_timeout: float = 5.0  # seconds
    price_improvement: bool = True
    smart_routing: bool = True
    adaptive_timing: bool = True

class HighPerformanceExecutor:
    def __init__(
        self,
        broker_name: str,
        symbols: List[str],
        config: ExecutionConfig,
        risk_manager: Any,
        demo_mode: bool = True
    ):
        self.broker_name = broker_name
        self.symbols = symbols
        self.config = config
        self.risk_manager = risk_manager
        self.demo_mode = demo_mode
        
        # Initialize execution components
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.order_queue = asyncio.Queue()
        self.price_cache = {}
        self.market_depth = {}
        
        # Initialize execution strategies
        self.strategies = {
            'aggressive': AggressiveStrategy(self.config),
            'balanced': BalancedStrategy(self.config),
            'conservative': ConservativeStrategy(self.config)
        }
        
        # Performance metrics
        self.execution_metrics = {
            symbol: {
                'latency': [],
                'slippage': [],
                'rejection_rate': 0.0,
                'fill_ratio': 1.0
            }
            for symbol in symbols
        }
    
    async def initialize(self) -> bool:
        """Initialize execution system"""
        try:
            # Initialize MT5 connection
            if not mt5.initialize(
                path=None,
                login=0,
                password="",
                server="",
                timeout=30000
            ):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Start background tasks
            asyncio.create_task(self._process_order_queue())
            asyncio.create_task(self._update_market_metrics())
            
            logger.info("High-performance executor initialized")
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            return False
    
    async def execute_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        volume: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        execution_style: str = 'balanced'
    ) -> Dict:
        """Execute order with optimal strategy"""
        try:
            start_time = datetime.now()
            
            # Validate order parameters
            if not self._validate_order_params(
                symbol, order_type, side, volume, price
            ):
                return {
                    'success': False,
                    'error': 'Invalid order parameters'
                }
            
            # Select execution strategy
            strategy = self._select_strategy(
                symbol,
                execution_style,
                volume
            )
            
            # Smart order routing
            if self.config.smart_routing:
                routing_venue = await self._get_optimal_venue(
                    symbol,
                    volume
                )
            else:
                routing_venue = 'default'
            
            # Execute with retry logic
            for attempt in range(self.config.retry_attempts):
                try:
                    # Get execution price
                    if order_type == OrderType.MARKET:
                        exec_price = (
                            await self._get_optimal_price(symbol)
                            if price is None
                            else price
                        )
                    else:
                        exec_price = price
                    
                    # Execute order
                    result = await strategy.execute(
                        symbol=symbol,
                        order_type=order_type.value,
                        side=side.value,
                        volume=volume,
                        price=exec_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if result['success']:
                        # Update metrics
                        execution_time = (
                            datetime.now() - start_time
                        ).total_seconds() * 1000
                        
                        await self._update_execution_metrics(
                            symbol,
                            execution_time,
                            result
                        )
                        
                        return {
                            'success': True,
                            'order_id': result['order_id'],
                            'executed_price': result['executed_price'],
                            'execution_time': execution_time,
                            'venue': routing_venue
                        }
                    
                    # Handle rejection
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(
                            self.config.retry_delay * (attempt + 1)
                        )
                    
                except Exception as e:
                    logger.error(f"Execution attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.config.retry_attempts - 1:
                        raise
            
            return {
                'success': False,
                'error': 'Max retry attempts reached'
            }
            
        except Exception as e:
            logger.error(f"Order execution error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _get_optimal_price(self, symbol: str) -> float:
        """Get optimal execution price with minimal latency"""
        try:
            # Check price cache
            cache_key = f"{symbol}_price"
            cache_time = self.price_cache.get(f"{symbol}_time")
            
            if (
                cache_key in self.price_cache and
                cache_time and
                (datetime.now() - cache_time).total_seconds() < 0.1
            ):
                return self.price_cache[cache_key]
            
            # Get fresh prices with parallel requests
            prices = await asyncio.gather(*[
                self._request_price(symbol) for _ in range(3)
            ])
            
            valid_prices = [p for p in prices if p is not None]
            if not valid_prices:
                raise ValueError("No valid prices received")
            
            # Calculate optimal price
            price = float(np.median(valid_prices))
            
            # Update cache
            self.price_cache[cache_key] = price
            self.price_cache[f"{symbol}_time"] = datetime.now()
            
            return price
            
        except Exception as e:
            logger.error(f"Optimal price error: {str(e)}")
            raise
    
    def _select_strategy(
        self,
        symbol: str,
        execution_style: str,
        volume: float
    ) -> ExecutionStrategy:
        """Select optimal execution strategy"""
        try:
            # Get market conditions
            metrics = self.execution_metrics[symbol]
            
            # Use conservative for large orders
            if volume > self.config.large_order_threshold:
                return self.strategies['conservative']
            
            # Use aggressive for good conditions
            if (
                metrics['latency'][-1] < 10 and  # Low latency
                metrics['rejection_rate'] < 0.05 and  # Low rejections
                metrics['fill_ratio'] > 0.95  # Good fills
            ):
                return self.strategies['aggressive']
            
            # Default to balanced
            return self.strategies[execution_style]
            
        except Exception as e:
            logger.error(f"Strategy selection error: {str(e)}")
            return self.strategies['balanced']
    
    async def _get_optimal_venue(
        self,
        symbol: str,
        volume: float
    ) -> str:
        """Get optimal execution venue"""
        try:
            # Analyze market depth
            depth = await self._get_market_depth(symbol)
            
            # Calculate fill probability
            venues = await self._analyze_venues(symbol, volume)
            
            # Select best venue
            best_venue = max(
                venues.items(),
                key=lambda x: x[1]['fill_probability']
            )
            
            return best_venue[0]
            
        except Exception as e:
            logger.error(f"Venue selection error: {str(e)}")
            return 'default'
    
    async def _update_execution_metrics(
        self,
        symbol: str,
        execution_time: float,
        result: Dict
    ):
        """Update execution quality metrics"""
        try:
            metrics = self.execution_metrics[symbol]
            
            # Update latency
            metrics['latency'].append(execution_time)
            if len(metrics['latency']) > 100:
                metrics['latency'] = metrics['latency'][-100:]
            
            # Update slippage
            if 'executed_price' in result and 'requested_price' in result:
                slippage = abs(
                    result['executed_price'] - result['requested_price']
                )
                metrics['slippage'].append(slippage)
                if len(metrics['slippage']) > 100:
                    metrics['slippage'] = metrics['slippage'][-100:]
            
            # Update rejection rate
            metrics['rejection_rate'] = (
                0.95 * metrics['rejection_rate'] +
                0.05 * (0 if result['success'] else 1)
            )
            
            # Update fill ratio
            metrics['fill_ratio'] = (
                0.95 * metrics['fill_ratio'] +
                0.05 * (1 if result['success'] else 0)
            )
            
        except Exception as e:
            logger.error(f"Metrics update error: {str(e)}")
    
    async def _process_order_queue(self):
        """Process queued orders"""
        while True:
            try:
                order = await self.order_queue.get()
                
                # Execute order
                result = await self.execute_order(**order)
                
                # Handle result
                if not result['success']:
                    logger.error(f"Queue order failed: {result['error']}")
                
                self.order_queue.task_done()
                
            except Exception as e:
                logger.error(f"Order queue processing error: {str(e)}")
            
            await asyncio.sleep(0.01)
    
    async def _update_market_metrics(self):
        """Update market metrics in background"""
        while True:
            try:
                for symbol in self.symbols:
                    # Update market depth
                    depth = await self._get_market_depth(symbol)
                    self.market_depth[symbol] = depth
                    
                    # Update spreads
                    spread = await self._get_current_spread(symbol)
                    self.price_cache[f"{symbol}_spread"] = spread
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Market metrics update error: {str(e)}")
                await asyncio.sleep(5)  # Retry after error