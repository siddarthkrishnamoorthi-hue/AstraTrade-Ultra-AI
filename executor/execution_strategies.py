"""
High-performance execution strategies with smart order routing
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
import numpy as np
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExecutionStats:
    latency: float
    slippage: float
    fill_ratio: float
    rejection_rate: float
    retry_count: int
    execution_time: datetime

class ExecutionStrategy(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.stats = {}
        self.execution_cache = {}
        self.last_prices = {}
        
    @abstractmethod
    async def execute(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        slippage: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        pass
    
    async def _validate_execution_conditions(
        self,
        symbol: str,
        price: float,
        volume: float
    ) -> bool:
        """Validate pre-execution conditions"""
        try:
            # Check spread conditions
            current_spread = await self._get_current_spread(symbol)
            if current_spread > self.config.max_spread:
                return False
            
            # Check volume conditions
            if volume < self.config.min_lot or volume > self.config.max_lot:
                return False
            
            # Check price validity
            if not await self._is_price_valid(symbol, price):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Execution validation error: {str(e)}")
            return False
    
    async def _optimize_execution_timing(
        self,
        symbol: str,
        order_type: str
    ) -> bool:
        """Optimize order execution timing"""
        try:
            # Check recent execution statistics
            stats = self.stats.get(symbol, {})
            if not stats:
                return True
            
            # Check market conditions
            spread = await self._get_current_spread(symbol)
            if spread > stats.get('avg_spread', 0) * 1.5:
                return False
            
            # Check execution window
            if order_type == 'market':
                return await self._check_market_conditions(symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"Execution timing error: {str(e)}")
            return True
    
    async def _route_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float]
    ) -> Tuple[bool, Dict]:
        """Smart order routing with latency optimization"""
        try:
            # Select optimal execution venue
            venue = await self._select_execution_venue(symbol)
            
            # Prepare order parameters
            params = self._prepare_order_params(
                symbol, order_type, volume, price
            )
            
            # Execute with retry logic
            result = await self._execute_with_retry(venue, params)
            
            return True, result
            
        except Exception as e:
            logger.error(f"Order routing error: {str(e)}")
            return False, {"error": str(e)}

class AggressiveStrategy(ExecutionStrategy):
    """Aggressive execution strategy for high-speed trading"""
    
    async def execute(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        slippage: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        try:
            start_time = datetime.now()
            
            # Ultra-fast price check
            if not price and order_type == 'market':
                price = await self._get_fast_price(symbol)
            
            # Parallel validation
            validation_tasks = [
                self._validate_execution_conditions(symbol, price, volume),
                self._optimize_execution_timing(symbol, order_type),
                self._check_market_impact(symbol, volume)
            ]
            
            results = await asyncio.gather(*validation_tasks)
            if not all(results):
                return {
                    "success": False,
                    "error": "Validation failed",
                    "execution_time": (datetime.now() - start_time).total_seconds() * 1000
                }
            
            # Smart order routing
            success, result = await self._route_order(
                symbol, order_type, volume, price
            )
            
            if not success:
                return {
                    "success": False,
                    "error": result.get("error", "Routing failed"),
                    "execution_time": (datetime.now() - start_time).total_seconds() * 1000
                }
            
            # Post-execution optimization
            await self._update_execution_stats(symbol, result)
            
            return {
                "success": True,
                "order_id": result.get("order_id"),
                "executed_price": result.get("executed_price"),
                "execution_time": (datetime.now() - start_time).total_seconds() * 1000
            }
            
        except Exception as e:
            logger.error(f"Aggressive execution error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _get_fast_price(self, symbol: str) -> float:
        """Get price with minimal latency"""
        try:
            # Check price cache
            cached = self.last_prices.get(symbol)
            if cached and (datetime.now() - cached["time"]).total_seconds() < 0.05:
                return cached["price"]
            
            # Parallel price requests
            prices = await asyncio.gather(*[
                self._request_price(symbol) for _ in range(3)
            ])
            
            # Filter and process
            valid_prices = [p for p in prices if p is not None]
            if not valid_prices:
                raise ValueError("No valid prices received")
            
            # Use median for stability
            price = float(np.median(valid_prices))
            
            # Update cache
            self.last_prices[symbol] = {
                "price": price,
                "time": datetime.now()
            }
            
            return price
            
        except Exception as e:
            logger.error(f"Fast price error: {str(e)}")
            if cached:
                return cached["price"]
            raise
    
    async def _check_market_impact(
        self,
        symbol: str,
        volume: float
    ) -> bool:
        """Check potential market impact"""
        try:
            # Get market depth
            depth = await self._get_market_depth(symbol)
            
            # Calculate potential impact
            impact = self._calculate_market_impact(depth, volume)
            
            # Check against thresholds
            return impact < self.config.max_impact
            
        except Exception as e:
            logger.error(f"Market impact check error: {str(e)}")
            return True  # Continue if check fails
    
    async def _update_execution_stats(
        self,
        symbol: str,
        result: Dict
    ):
        """Update execution statistics"""
        try:
            stats = self.stats.get(symbol, ExecutionStats(
                latency=0,
                slippage=0,
                fill_ratio=1,
                rejection_rate=0,
                retry_count=0,
                execution_time=datetime.now()
            ))
            
            # Update metrics
            stats.latency = 0.95 * stats.latency + 0.05 * result.get("execution_time", 0)
            stats.slippage = 0.95 * stats.slippage + 0.05 * abs(
                result.get("executed_price", 0) - 
                result.get("requested_price", 0)
            )
            stats.fill_ratio = 0.95 * stats.fill_ratio + 0.05 * (
                1 if result.get("success") else 0
            )
            stats.rejection_rate = 0.95 * stats.rejection_rate + 0.05 * (
                0 if result.get("success") else 1
            )
            stats.retry_count = result.get("retries", 0)
            stats.execution_time = datetime.now()
            
            self.stats[symbol] = stats
            
        except Exception as e:
            logger.error(f"Stats update error: {str(e)}")

class BalancedStrategy(ExecutionStrategy):
    """Balanced execution strategy with dynamic adaptation"""
    
    async def execute(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        slippage: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        try:
            start_time = datetime.now()
            
            # Dynamic slippage adjustment
            if slippage is None:
                slippage = await self._calculate_optimal_slippage(symbol)
            
            # Smart price improvement
            if order_type == 'market':
                price = await self._get_improved_price(symbol)
            
            # Execute with balance between speed and price
            result = await self._execute_balanced(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                price=price,
                slippage=slippage
            )
            
            # Post-execution analysis
            await self._analyze_execution(result)
            
            return {
                "success": result.get("success", False),
                "order_id": result.get("order_id"),
                "executed_price": result.get("executed_price"),
                "execution_time": (datetime.now() - start_time).total_seconds() * 1000
            }
            
        except Exception as e:
            logger.error(f"Balanced execution error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _calculate_optimal_slippage(self, symbol: str) -> float:
        """Calculate optimal slippage based on market conditions"""
        try:
            # Get recent execution statistics
            stats = self.stats.get(symbol, {})
            
            # Calculate dynamic slippage
            base_slippage = self.config.base_slippage
            if stats:
                # Adjust based on recent performance
                latency_factor = min(stats.get('latency', 0) / 100, 1)
                rejection_factor = min(stats.get('rejection_rate', 0) * 2, 1)
                
                return base_slippage * (1 + latency_factor + rejection_factor)
            
            return base_slippage
            
        except Exception as e:
            logger.error(f"Slippage calculation error: {str(e)}")
            return self.config.base_slippage
    
    async def _get_improved_price(self, symbol: str) -> float:
        """Get improved price with smart aggregation"""
        try:
            # Get prices from multiple sources
            prices = await asyncio.gather(*[
                self._request_price(symbol) for _ in range(2)
            ])
            
            # Filter and weight prices
            valid_prices = [p for p in prices if p is not None]
            if not valid_prices:
                raise ValueError("No valid prices received")
            
            # Use weighted average
            weights = [0.7, 0.3]  # Prioritize first source
            price = np.average(valid_prices[:2], weights=weights[:len(valid_prices)])
            
            return float(price)
            
        except Exception as e:
            logger.error(f"Price improvement error: {str(e)}")
            return await self._get_fast_price(symbol)
    
    async def _execute_balanced(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        slippage: float
    ) -> Dict:
        """Execute with balanced approach"""
        try:
            # Pre-execution checks
            if not await self._validate_execution_conditions(
                symbol, price, volume
            ):
                return {"success": False, "error": "Validation failed"}
            
            # Smart routing
            success, result = await self._route_order(
                symbol, order_type, volume, price
            )
            
            if not success:
                return {"success": False, "error": result.get("error")}
            
            return result
            
        except Exception as e:
            logger.error(f"Balanced execution error: {str(e)}")
            return {"success": False, "error": str(e)}

class ConservativeStrategy(ExecutionStrategy):
    """Conservative execution strategy prioritizing fill quality"""
    
    async def execute(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        slippage: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        try:
            start_time = datetime.now()
            
            # Enhanced validation
            if not await self._enhanced_validation(symbol, volume):
                return {
                    "success": False,
                    "error": "Enhanced validation failed",
                    "execution_time": (datetime.now() - start_time).total_seconds() * 1000
                }
            
            # Conservative price improvement
            if order_type == 'market':
                price = await self._get_conservative_price(symbol)
            
            # Execute with focus on quality
            result = await self._execute_conservative(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                price=price,
                slippage=slippage
            )
            
            return {
                "success": result.get("success", False),
                "order_id": result.get("order_id"),
                "executed_price": result.get("executed_price"),
                "execution_time": (datetime.now() - start_time).total_seconds() * 1000
            }
            
        except Exception as e:
            logger.error(f"Conservative execution error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _enhanced_validation(
        self,
        symbol: str,
        volume: float
    ) -> bool:
        """Enhanced validation for conservative execution"""
        try:
            # Standard validation
            if not await self._validate_execution_conditions(
                symbol,
                await self._get_conservative_price(symbol),
                volume
            ):
                return False
            
            # Additional checks
            spread = await self._get_current_spread(symbol)
            if spread > self.config.max_spread * 0.8:  # More conservative
                return False
            
            # Check recent executions
            stats = self.stats.get(symbol, {})
            if stats.get('rejection_rate', 0) > 0.05:  # More conservative
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced validation error: {str(e)}")
            return False
    
    async def _get_conservative_price(self, symbol: str) -> float:
        """Get conservative price estimate"""
        try:
            # Get multiple price samples
            prices = []
            for _ in range(3):
                price = await self._request_price(symbol)
                if price is not None:
                    prices.append(price)
                await asyncio.sleep(0.05)  # Small delay between requests
            
            if not prices:
                raise ValueError("No valid prices received")
            
            # Use most conservative price
            return float(np.median(prices))
            
        except Exception as e:
            logger.error(f"Conservative price error: {str(e)}")
            return await self._get_fast_price(symbol)
    
    async def _execute_conservative(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        slippage: Optional[float]
    ) -> Dict:
        """Execute with conservative approach"""
        try:
            # Use reduced slippage
            if slippage is None:
                slippage = self.config.base_slippage * 0.7
            
            # Execute with retry logic
            for attempt in range(3):
                success, result = await self._route_order(
                    symbol, order_type, volume, price
                )
                
                if success:
                    return result
                
                await asyncio.sleep(0.1 * (attempt + 1))
            
            return {"success": False, "error": "Max retries reached"}
            
        except Exception as e:
            logger.error(f"Conservative execution error: {str(e)}")
            return {"success": False, "error": str(e)}