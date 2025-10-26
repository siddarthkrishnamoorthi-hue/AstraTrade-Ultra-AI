"""
High-performance broker adapter with dynamic latency optimization
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import MetaTrader5 as mt5
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class BrokerConfig:
    name: str
    server: str
    execution_mode: str  # 'instant', 'market', 'limit'
    typical_spread: float
    commission_per_lot: float
    min_lot: float
    lot_step: float
    margin_call_level: float
    stop_out_level: float
    hedge_allowed: bool
    price_digits: Dict[str, int]
    swap_rates: Dict[str, Tuple[float, float]]

@dataclass
class ExecutionMetrics:
    latency: float  # ms
    slippage: float  # pips
    rejection_rate: float
    fill_ratio: float
    last_update: datetime

class BrokerAdapter:
    def __init__(
        self,
        broker_name: str,
        account_type: str = "standard",
        execution_mode: str = "market",
        optimization_level: str = "maximum"
    ):
        self.broker_name = broker_name
        self.account_type = account_type
        self.execution_mode = execution_mode
        self.optimization_level = optimization_level
        
        # Initialize performance metrics
        self.metrics = {}
        self.latency_history = []
        self.spread_history = {}
        self.rejection_history = {}
        
        # Optimization settings
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.price_cache = {}
        self.cache_ttl = 0.1  # 100ms
        
        # Load broker-specific configurations
        self.config = self._load_broker_config()
        
        # Initialize connection pools
        self._init_connection_pools()
    
    def _load_broker_config(self) -> BrokerConfig:
        """Load broker-specific configuration"""
        configs = {
            "ICMarkets": BrokerConfig(
                name="ICMarkets",
                server="ICMarkets-Demo",
                execution_mode="market",
                typical_spread=0.1,
                commission_per_lot=7.0,
                min_lot=0.01,
                lot_step=0.01,
                margin_call_level=80.0,
                stop_out_level=50.0,
                hedge_allowed=True,
                price_digits={"EURUSD": 5, "XAUUSD": 2},
                swap_rates={"EURUSD": (-0.35, 0.12)}
            ),
            "FXCM": BrokerConfig(
                name="FXCM",
                server="FXCM-Demo",
                execution_mode="market",
                typical_spread=0.2,
                commission_per_lot=0.0,
                min_lot=0.01,
                lot_step=0.01,
                margin_call_level=75.0,
                stop_out_level=45.0,
                hedge_allowed=False,
                price_digits={"EURUSD": 5, "XAUUSD": 2},
                swap_rates={"EURUSD": (-0.42, 0.15)}
            )
        }
        return configs.get(self.broker_name, configs["ICMarkets"])
    
    def _init_connection_pools(self):
        """Initialize connection pools for parallel execution"""
        self.price_stream = asyncio.Queue()
        self.order_stream = asyncio.Queue()
        self.max_connections = 5
        self.active_connections = []
    
    async def optimize_execution(self):
        """Continuously optimize execution parameters"""
        while True:
            try:
                # Update latency measurements
                latencies = await self._measure_latencies()
                self.latency_history.append(latencies)
                
                # Adjust execution parameters
                if len(self.latency_history) >= 100:
                    self._adjust_execution_params()
                    self.latency_history = self.latency_history[-100:]
                
                # Update spread statistics
                await self._update_spread_stats()
                
                # Clean up old data
                self._cleanup_old_metrics()
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in execution optimization: {str(e)}")
                await asyncio.sleep(5)
    
    async def execute_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        slippage: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """Execute order with smart routing and latency optimization"""
        try:
            # Normalize symbol for broker
            symbol = self._normalize_symbol(symbol)
            
            # Pre-execution checks
            if not await self._validate_order_params(
                symbol, order_type, volume, price
            ):
                return {"success": False, "error": "Invalid parameters"}
            
            # Calculate optimal execution parameters
            exec_params = self._calculate_execution_params(
                symbol, order_type, volume, price
            )
            
            # Select best execution strategy
            strategy = self._select_execution_strategy(
                symbol, exec_params, self.metrics.get(symbol, {})
            )
            
            # Execute with retry logic
            for attempt in range(3):
                try:
                    # Get fresh price if needed
                    if order_type == "market":
                        price = await self._get_optimal_price(symbol)
                    
                    # Apply execution strategy
                    result = await strategy.execute(
                        symbol=symbol,
                        order_type=order_type,
                        volume=volume,
                        price=price,
                        slippage=exec_params["max_slippage"],
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    # Post-execution analysis
                    await self._analyze_execution(result)
                    
                    return result
                    
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        raise e
                    await asyncio.sleep(0.1 * (attempt + 1))
            
        except Exception as e:
            logger.error(f"Order execution error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _measure_latencies(self) -> Dict[str, float]:
        """Measure current execution latencies"""
        latencies = {}
        for symbol in self.config.price_digits.keys():
            start = datetime.now()
            try:
                await self._get_price(symbol)
                latency = (datetime.now() - start).total_seconds() * 1000
                latencies[symbol] = latency
            except:
                continue
        return latencies
    
    def _adjust_execution_params(self):
        """Dynamically adjust execution parameters based on performance"""
        for symbol, latencies in self.latency_history[-100:]:
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            # Adjust slippage tolerance
            self.metrics[symbol] = {
                "max_slippage": max(
                    1.0,  # Minimum 1 pip
                    avg_latency / 100 + 2 * std_latency / 100
                ),
                "retry_delay": min(
                    avg_latency / 2,
                    100  # Max 100ms delay
                )
            }
    
    async def _validate_order_params(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float]
    ) -> bool:
        """Validate order parameters against broker rules"""
        try:
            # Check symbol
            if symbol not in self.config.price_digits:
                return False
            
            # Check volume constraints
            if volume < self.config.min_lot:
                return False
            if volume % self.config.lot_step != 0:
                return False
            
            # Check price precision
            if price:
                digits = self.config.price_digits[symbol]
                if len(str(price).split('.')[-1]) > digits:
                    return False
            
            # Check account margin
            if not await self._check_margin_requirements(symbol, volume):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation error: {str(e)}")
            return False
    
    def _select_execution_strategy(
        self,
        symbol: str,
        params: Dict,
        metrics: Dict
    ) -> 'ExecutionStrategy':
        """Select optimal execution strategy based on conditions"""
        if metrics.get('rejection_rate', 0) > 0.1:
            return ConservativeStrategy(self.config)
        elif metrics.get('latency', 0) < 10:
            return AggressiveStrategy(self.config)
        else:
            return BalancedStrategy(self.config)
    
    async def _get_optimal_price(self, symbol: str) -> float:
        """Get optimal price with minimal latency"""
        cache_key = f"{symbol}_price"
        cache_time = self.price_cache.get(f"{symbol}_time")
        
        # Use cached price if fresh enough
        if (
            cache_key in self.price_cache and
            cache_time and
            (datetime.now() - cache_time).total_seconds() < self.cache_ttl
        ):
            return self.price_cache[cache_key]
        
        # Get fresh price with parallel requests
        prices = await asyncio.gather(*[
            self._get_price(symbol) for _ in range(3)
        ])
        
        # Filter out None values and get median
        valid_prices = [p for p in prices if p is not None]
        if not valid_prices:
            raise Exception("Could not get valid price")
        
        optimal_price = float(np.median(valid_prices))
        
        # Update cache
        self.price_cache[cache_key] = optimal_price
        self.price_cache[f"{symbol}_time"] = datetime.now()
        
        return optimal_price
    
    async def _analyze_execution(self, result: Dict):
        """Analyze execution quality and update metrics"""
        symbol = result.get('symbol')
        if not symbol:
            return
            
        # Update execution metrics
        self.metrics[symbol] = {
            'latency': result.get('execution_time', 1000),
            'slippage': abs(
                result.get('executed_price', 0) - 
                result.get('requested_price', 0)
            ),
            'rejection_rate': (
                self.metrics.get(symbol, {}).get('rejection_rate', 0) * 0.95 +
                (0.05 if not result.get('success') else 0)
            ),
            'last_update': datetime.now()
        }
        
        # Log unusual patterns
        if self.metrics[symbol]['slippage'] > 5:  # More than 5 pips
            logger.warning(
                f"High slippage detected for {symbol}: "
                f"{self.metrics[symbol]['slippage']} pips"
            )
        
        if self.metrics[symbol]['latency'] > 500:  # More than 500ms
            logger.warning(
                f"High latency detected for {symbol}: "
                f"{self.metrics[symbol]['latency']}ms"
            )