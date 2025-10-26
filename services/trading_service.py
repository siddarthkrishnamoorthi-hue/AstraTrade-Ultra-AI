"""
Trading service for handling MT5 interactions and order execution
"""

import asyncio
import grpc
from concurrent import futures
import logging
from datetime import datetime, timedelta
import json
import os
import redis
import zmq.asyncio
from kafka import KafkaProducer, KafkaConsumer
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from executor.high_performance_executor import HighPerformanceExecutor
from risk.guardian import RiskGuardian
from utils.symbol_utils import SymbolManager
from agents.news_calendar import NewsCalendar
from agents.news_trader import NewsTrader, NewsTradeSetup
from services.exceptions import (
    TradingServiceError, ExecutorInitError,
    MarketDataError, OrderExecutionError, ConnectionError
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TradingService")

@dataclass
class ServiceConfig:
    """Configuration for TradingService"""
    redis_host: str = "localhost"
    kafka_broker: str = "localhost:9092"
    alpha_vantage_key: Optional[str] = None

class TradingService:
    def __init__(self, config: Optional[ServiceConfig] = None):
        """Initialize trading service with configuration
        
        Args:
            config: Service configuration, if None uses environment variables
        """
        if config is None:
            config = ServiceConfig(
                redis_host=os.getenv("REDIS_HOST", "localhost"),
                kafka_broker=os.getenv("KAFKA_BROKER", "localhost:9092"),
                alpha_vantage_key=os.getenv("ALPHA_VANTAGE_KEY")
            )
            
        try:
            # Initialize message brokers
            self.redis = redis.Redis(host=config.redis_host)
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=config.kafka_broker,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            self.kafka_consumer = KafkaConsumer(
                'trade_signals',
                bootstrap_servers=config.kafka_broker,
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to message brokers: {str(e)}")
        
        # Initialize components
        self.executor = HighPerformanceExecutor()
        self.risk_guardian = RiskGuardian()
        self.symbol_manager = SymbolManager()
        
        # Initialize news trading components
        self.news_calendar = NewsCalendar(alpha_vantage_key=config.alpha_vantage_key)
        self.news_trader = NewsTrader(risk_manager=self.risk_guardian)
        
        # Task storage to prevent garbage collection
        self._tasks: List[asyncio.Task] = []
        
        # Start news monitoring task
        self._news_monitor_task = asyncio.create_task(self._monitor_news_events())
        
    async def _monitor_news_events(self):
        """Monitor upcoming news events and prepare for trading"""
        while True:
            try:
                # Fetch news events for next 24 hours
                now = datetime.now()
                events = await self.news_calendar.fetch_economic_calendar(
                    start_date=now,
                    end_date=now + timedelta(days=1)
                )
                
                for event in events:
                    if self.news_trader.should_trade_event(event):
                        # Calculate time until event
                        time_until_event = (event.timestamp - now).total_seconds()
                        
                        if 0 <= time_until_event <= 300:  # Within 5 minutes
                            await self._prepare_news_trade(event)
                            
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in news monitoring: {str(e)}")
                await asyncio.sleep(60)
                
    async def _prepare_news_trade(self, event):
        """Prepare for news trade execution"""
        affected_pairs = self._get_affected_pairs(event.currency)
        
        for symbol in affected_pairs:
            # Get current market data
            current_price = await self.executor.get_current_price(symbol)
            
            # Generate trade setup
            setup = self.news_trader.generate_trade_setup(
                event=event,
                symbol=symbol,
                current_price=current_price
            )
            
            if setup and setup.confidence >= 0.7:  # High confidence threshold for news trades
                # Store pre-news volatility
                self.news_trader.pre_news_volatility[symbol] = (
                    await self._calculate_volatility(symbol)
                )
                
                # Submit trade to executor
                await self._execute_news_trade(setup)
                
    async def _execute_news_trade(self, setup: NewsTradeSetup):
        """Execute news trade with proper risk management"""
        position_size = self.news_trader.calculate_position_size(
            event=setup.event,
            confidence=setup.confidence,
            symbol=setup.entry_price
        )
        
        if position_size > 0:
            await self.executor.submit_order(
                symbol=setup.entry_price,
                order_type="MARKET",
                direction=setup.direction,
                volume=position_size,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit,
                comment=f"NEWS_{setup.event.event}"
            )
            
    def _get_affected_pairs(self, currency: str) -> List[str]:
        """Get trading pairs affected by news for given currency"""
        affected = []
        for pair in self.symbol_manager.get_symbols():
            if currency in pair:
                affected.append(pair)
        return affected
        
    async def _calculate_volatility(self, symbol: str) -> float:
        """Calculate current volatility for symbol"""
        candles = await self.executor.get_candles(
            symbol=symbol,
            timeframe="M5",
            count=12  # Last hour
        )
        if len(candles) > 0:
            return np.std(candles['high'] - candles['low'])
        
        # ZMQ context for high-speed messaging
        self.zmq_context = zmq.asyncio.Context()
        self.tick_socket = self.zmq_context.socket(zmq.SUB)
        self.tick_socket.connect("tcp://localhost:5555")
        
        # State management
        self.active_orders: Dict[str, Dict] = {}
        self.position_cache: Dict[str, Dict] = {}
        self.last_tick: Dict[str, Dict] = {}
    
    async def start(self):
        """Start the trading service"""
        try:
            # Initialize executor
            if not await self.executor.initialize():
                raise Exception("Failed to initialize executor")
            
            # Start background tasks
            asyncio.create_task(self._process_signals())
            asyncio.create_task(self._process_ticks())
            
            logger.info("Trading service started successfully")
            
            # Keep service running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Service error: {str(e)}")
            await self.shutdown()
    
    async def _process_signals(self):
        """Process incoming trade signals"""
        async for message in self.kafka_consumer:
            try:
                signal = message.value
                
                # Validate signal
                if not self._validate_signal(signal):
                    continue
                
                # Check risk limits
                if not await self._check_risk_limits(signal):
                    continue
                
                # Execute trade
                result = await self._execute_trade(signal)
                
                # Publish result
                self.kafka_producer.send(
                    'trade_results',
                    {
                        'signal_id': signal['id'],
                        'result': result
                    }
                )
                
            except Exception as e:
                logger.error(f"Signal processing error: {str(e)}")
    
    async def _process_ticks(self):
        """Process incoming market ticks"""
        while True:
            try:
                # Get tick data
                topic = await self.tick_socket.recv_string()
                data = await self.tick_socket.recv_pyobj()
                
                # Update tick cache
                self.last_tick[topic] = {
                    'price': data['price'],
                    'volume': data['volume'],
                    'timestamp': data['timestamp']
                }
                
                # Check for order updates
                await self._check_order_updates(topic, data)
                
                # Update position cache
                await self._update_position_cache(topic)
                
            except Exception as e:
                logger.error(f"Tick processing error: {str(e)}")
                await asyncio.sleep(0.1)
    
    async def _execute_trade(self, signal: Dict) -> Dict:
        """Execute a trade based on signal"""
        try:
            # Get execution parameters
            params = await self._get_execution_params(signal)
            
            # Execute order
            result = await self.executor.execute_order(
                symbol=signal['symbol'],
                order_type=signal['type'],
                side=signal['side'],
                volume=params['volume'],
                price=params.get('price'),
                stop_loss=params.get('stop_loss'),
                take_profit=params.get('take_profit')
            )
            
            if result['success']:
                # Cache order details
                self.active_orders[result['order_id']] = {
                    'signal': signal,
                    'params': params,
                    'result': result
                }
                
                # Update Redis cache
                self.redis.hset(
                    f"orders:{signal['symbol']}",
                    result['order_id'],
                    json.dumps({
                        'status': 'active',
                        'details': result
                    })
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _get_execution_params(self, signal: Dict) -> Dict:
        """Calculate optimal execution parameters"""
        try:
            # Get market data
            market_data = self._get_market_data(signal['symbol'])
            
            # Calculate position size
            volume = await self.risk_guardian.calculate_position_size(
                signal['symbol'],
                signal['risk_score'],
                market_data
            )
            
            # Calculate entry parameters
            params = {
                'volume': volume,
                'price': None  # For market orders
            }
            
            # Add stop loss and take profit
            if signal.get('stop_loss'):
                params['stop_loss'] = signal['stop_loss']
            if signal.get('take_profit'):
                params['take_profit'] = signal['take_profit']
            
            return params
            
        except Exception as e:
            logger.error(f"Execution parameter calculation error: {str(e)}")
            raise
    
    async def _check_risk_limits(self, signal: Dict) -> bool:
        """Check if trade meets risk requirements"""
        try:
            # Get account state
            account_info = await self._get_account_info()
            
            # Check margin requirements
            if not await self._check_margin(signal, account_info):
                return False
            
            # Check exposure limits
            if not await self._check_exposure(signal):
                return False
            
            # Check correlation limits
            if not await self._check_correlations(signal):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Risk check error: {str(e)}")
            return False
    
    def _validate_signal(self, signal: Dict) -> bool:
        """Validate trading signal"""
        required_fields = {
            'id', 'symbol', 'type', 'side', 'risk_score'
        }
        
        return all(
            field in signal
            for field in required_fields
        )
    
    async def _check_order_updates(self, symbol: str, tick_data: Dict):
        """Check and update order status"""
        try:
            # Get active orders for symbol
            orders = self.active_orders.copy()
            
            for order_id, order in orders.items():
                if order['signal']['symbol'] != symbol:
                    continue
                
                # Check stop loss/take profit
                if await self._check_exit_conditions(
                    order,
                    tick_data
                ):
                    await self._close_position(order_id)
                
                # Update order status
                await self._update_order_status(order_id)
                
        except Exception as e:
            logger.error(f"Order update error: {str(e)}")
    
    async def shutdown(self):
        """Shutdown the service"""
        try:
            # Close all positions
            await self.executor.close_all_positions()
            
            # Cleanup connections
            self.redis.close()
            self.kafka_producer.close()
            self.kafka_consumer.close()
            self.tick_socket.close()
            self.zmq_context.term()
            
            logger.info("Trading service shut down successfully")
            
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")

if __name__ == "__main__":
    service = TradingService(
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        kafka_broker=os.getenv("KAFKA_BROKER", "localhost:9092")
    )
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(service.start())
    except KeyboardInterrupt:
        loop.run_until_complete(service.shutdown())
    finally:
        loop.close()