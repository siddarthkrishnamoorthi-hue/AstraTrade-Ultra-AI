"""
Main orchestration module for AstraTrade Ultra AI
"""

import asyncio
import yaml
import logging
import json
import redis
from kafka import KafkaProducer, KafkaConsumer
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import signal
import sys
import argparse
from datetime import datetime, timedelta

from advanced.micro_structure import TickRecorder, MarketRegimeDetector
from advanced.pattern_detector import PairPatternDetector
from utils.symbol_utils import SymbolProfile

from db.db_handler import DatabaseHandler
from ml.ensemble import MarketPredictor
from advanced.detectors import TechnicalDetector
from agents.news_calendar import NewsCalendar
from risk.guardian import RiskGuardian
from rl.evolutor import StrategyEvolutor
from executor.live import TradingSession, OrderRequest, OrderType, OrderSide

class AstraTrade:
    def __init__(self, config_path: str = "config.yaml", demo_mode: bool = True):
        """Initialize AstraTrade Ultra AI system"""
        self.config_path = Path(config_path)
        self.demo_mode = demo_mode
        self.running = False
        self.initialized = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("astratrade.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AstraTrade")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.db_handler: Optional[DatabaseHandler] = None
        self.market_predictor: Optional[MarketPredictor] = None
        self.technical_detector: Optional[TechnicalDetector] = None
        self.news_calendar: Optional[NewsCalendar] = None
        self.risk_guardian: Optional[RiskGuardian] = None
        self.strategy_evolutor: Optional[StrategyEvolutor] = None
        self.trading_session: Optional[TradingSession] = None
        
        # Runtime state
        self.active_trades: Dict = {}
        self.last_update: Dict[str, datetime] = {}
        self.update_tasks: List[asyncio.Task] = []

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            sys.exit(1)

    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            # Initialize database and ensure schema
            self.db_handler = DatabaseHandler(self.config['database']['url'])
            if not await self.db_handler.init_db():
                self.logger.error("Database initialization failed")
                return False
            
            # Load and initialize ML components with cached models
            try:
                self.market_predictor = MarketPredictor(
                    **self.config['ml']['ensemble'],
                    model_cache_dir=Path("models/cache"),
                    enable_gpu=self.config['ml'].get('enable_gpu', True)
                )
                await self.market_predictor.warm_up()  # Pre-load models
            except Exception as e:
                self.logger.error(f"ML initialization failed: {e}")
                return False
            
            # Initialize technical analysis with adaptive parameters
            try:
                self.technical_detector = TechnicalDetector(
                    adaptation_rate=self.config['technical'].get('adaptation_rate', 0.05),
                    volatility_scaling=True
                )
            except Exception as e:
                self.logger.error(f"Technical analysis initialization failed: {e}")
                return False
            
            # Initialize news analysis with real-time event monitoring
            try:
                # Setup Redis for caching
                self.redis_client = redis.Redis(
                    host=self.config['redis'].get('host', 'localhost'),
                    port=self.config['redis'].get('port', 6379),
                    db=0
                )
                
                # Setup Kafka for message bus
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.config['kafka'].get('servers', ['localhost:9092']),
                    value_serializer=lambda x: json.dumps(x).encode('utf-8')
                )
                
                # Initialize components per symbol
                self.symbol_handlers = {}
                self.pattern_detectors = {}
                self.symbol_profiles = {}
                
                for symbol_config in self.config['symbols']:
                    symbol = symbol_config['symbol']
                    account_number = symbol_config['account_number']
                    
                    # Create symbol-specific handlers
                    self.symbol_handlers[symbol] = {
                        'account': account_number,
                        'session': await TradingSession.create(
                            symbol=symbol,
                            account_number=account_number,
                            timeframes=symbol_config['timeframes'],
                            risk_manager=self.risk_guardian,
                            demo_mode=self.demo_mode
                        ),
                        'tick_recorder': TickRecorder(
                            self.db_handler,
                            symbol=symbol
                        ),
                        'regime_detector': MarketRegimeDetector(
                            symbol=symbol,
                            n_regimes=3,
                            features=['volatility', 'trend_strength', 'liquidity']
                        )
                    }
                    
                    # Initialize pattern detector for the symbol
                    self.pattern_detectors[symbol] = PairPatternDetector(
                        symbol=symbol,
                        config=self.config
                    )
                    
                    # Load symbol profile
                    self.symbol_profiles[symbol] = SymbolProfile(symbol)
                    
                    # Subscribe to symbol-specific Kafka topics
                    self.kafka_consumer = KafkaConsumer(
                        f'predictions_{symbol}',
                        f'signals_{symbol}',
                        f'execution_{symbol}',
                        bootstrap_servers=self.config['kafka'].get('servers', ['localhost:9092']),
                        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                        group_id=f'astratrade_{symbol}'
                    )
                
                # Initialize news calendar with sentiment analysis
                self.news_calendar = NewsCalendar(
                    alpha_vantage_key=self.config['api']['alpha_vantage_key'],
                    base_currency="USD",
                    supported_pairs=self._normalize_symbols([p['symbol'] for p in self.config['pairs']]),
                    event_impact_threshold=self.config['news'].get('impact_threshold', 0.6),
                    enable_sentiment=True
                )
                
            except Exception as e:
                self.logger.error(f"Service initialization failed: {e}")
                return False
                self.news_calendar = NewsCalendar(
                    alpha_vantage_key=self.config['api']['alpha_vantage_key'],
                    base_currency="USD",
                    supported_pairs=self._normalize_symbols([p['symbol'] for p in self.config['pairs']]),
                    event_impact_threshold=self.config['news'].get('impact_threshold', 0.6),
                    enable_sentiment=True
                )
            
            # Initialize risk management
            self.risk_guardian = RiskGuardian(**self.config['risk'])
            
            # Initialize strategy evolution
            self.strategy_evolutor = StrategyEvolutor(
                model_path="models/rl",
                training_episodes=self.config['rl']['training_episodes']
            )
            
            # Initialize trading session
            self.trading_session = TradingSession(
                symbols=[p['symbol'] for p in self.config['pairs']],
                timeframes=self.config['pairs'][0]['timeframes'],
                risk_manager=self.risk_guardian,
                telegram_token=self.config['api']['telegram_token'],
                telegram_chat_id=self.config['api']['telegram_chat_id'],
                demo_mode=self.demo_mode
            )
            
            await self.trading_session.initialize()
            
            self.initialized = True
            self.logger.info("System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False

    async def start(self):
        """Start the trading system"""
        if not self.initialized and not await self.initialize():
            return
        
        self.running = True
        self.logger.info("Starting AstraTrade Ultra AI")
        
        # Register signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self.shutdown(s))
            )
        
        try:
            # Start market data updates
            for pair in self.config['pairs']:
                for timeframe in pair['timeframes']:
                    task = asyncio.create_task(
                        self._market_data_loop(pair['symbol'], timeframe)
                    )
                    self.update_tasks.append(task)
            
            # Start strategy evolution
            evolution_task = asyncio.create_task(self._strategy_evolution_loop())
            self.update_tasks.append(evolution_task)
            
            # Start database cleanup
            cleanup_task = asyncio.create_task(self._database_cleanup_loop())
            self.update_tasks.append(cleanup_task)
            
            # Wait for all tasks
            await asyncio.gather(*self.update_tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
            await self.shutdown()

    async def _market_data_loop(self, symbol: str, timeframe: str):
        """Process market data updates for a symbol/timeframe"""
        while self.running:
            try:
                # Get market data
                data = await self.trading_session.get_market_data(
                    symbol,
                    timeframe,
                    bars=1000
                )
                
                if data.empty:
                    continue
                
                # Technical analysis
                analysis = self.technical_detector.analyze_market_structure(data)
                
                # ML prediction
                prediction = self.market_predictor.predict(data)
                
                # News analysis
                news_probs = await self.news_calendar.get_trade_probability(
                    symbol,
                    prediction.probability,
                    prediction.direction
                )
                
                # Generate trading signal if conditions met
                if self._should_trade(prediction, news_probs, analysis):
                    await self._execute_trade(
                        symbol,
                        prediction,
                        news_probs,
                        analysis,
                        data
                    )
                
                # Update last processed time
                self.last_update[f"{symbol}_{timeframe}"] = datetime.now()
                
                # Sleep until next update
                await asyncio.sleep(self.config['system']['update_interval'])
                
            except Exception as e:
                self.logger.error(
                    f"Error processing {symbol} {timeframe}: {str(e)}"
                )
                await asyncio.sleep(5)  # Backoff on error

    async def _strategy_evolution_loop(self):
        """Periodically evolve trading strategy"""
        while self.running:
            try:
                # Train strategy on recent data
                trades = await self.db_handler.get_trades(
                    start_date=datetime.now() - timedelta(days=30)
                )
                
                if trades:
                    # Update RL model
                    self.strategy_evolutor.train(continue_training=True)
                    
                    # Evaluate performance
                    metrics = self.strategy_evolutor.evaluate()
                    self.logger.info(f"Strategy evolution metrics: {metrics}")
                
                # Sleep until next update
                await asyncio.sleep(
                    self.config['rl']['update_frequency'] * 24 * 3600
                )
                
            except Exception as e:
                self.logger.error(f"Error in strategy evolution: {str(e)}")
                await asyncio.sleep(3600)  # Retry hourly on error

    async def _database_cleanup_loop(self):
        """Periodically clean up old data"""
        while self.running:
            try:
                await self.db_handler.cleanup_old_data(
                    self.config['database']['cleanup_days']
                )
                await asyncio.sleep(self.config['system']['cleanup_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in database cleanup: {str(e)}")
                await asyncio.sleep(3600)

    def _should_trade(
        self,
        prediction: Dict,
        news_probs: Dict,
        analysis: Dict
    ) -> bool:
        """Determine if conditions warrant opening a trade"""
        # Check probability thresholds
        if prediction.probability < self.config['risk']['min_win_probability']:
            return False
        
        # Check news risk
        if news_probs['news_risk'] > 0.5:  # High news risk
            return False
        
        # Check technical confluence
        confluence = sum(
            item.get('confluence_score', 0)
            for items in analysis.values()
            for item in items
        ) / len(analysis)
        
        if confluence < self.config['technical']['min_confluence']:
            return False
        
        return True

    async def _execute_trade(
        self,
        symbol: str,
        prediction: Dict,
        news_probs: Dict,
        analysis: Dict,
        data: pd.DataFrame
    ):
        """Execute a trade based on analysis"""
        try:
            # Calculate entry and exit points
            current_price = data['close'].iloc[-1]
            atr = self.technical_detector._calculate_atr(data)
            
            # Determine stop loss and take profit
            if prediction.direction == "LONG":
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 4)  # 1:2 RR
            else:
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 4)
            
            # Get real-time account info for position sizing
            account_info = await self._get_account_info()
            
            # Detect broker-specific symbol format
            normalized_symbol = self._normalize_symbol(symbol)
            symbol_info = await self.trading_session.get_symbol_info(normalized_symbol)
            
            # Dynamic position sizing with broker-aware calculations
            position = await self.risk_guardian.calculate_dynamic_position(
                pair=normalized_symbol,
                entry_price=current_price,
                stop_loss=stop_loss,
                account_info=account_info,
                symbol_info=symbol_info,
                win_probability=prediction.probability,
                volatility=data['close'].pct_change().std(),
                news_risk=news_probs['news_risk'],
                market_state={
                    'spread': await self.trading_session.get_spread(normalized_symbol),
                    'tick_value': symbol_info.get('tick_value', 0.01),
                    'contract_size': symbol_info.get('contract_size', 100000),
                    'margin_required': symbol_info.get('margin_required', 1000.0)
                },
                risk_params={
                    'max_risk_per_trade': self.config['risk']['max_risk_per_trade'],
                    'max_position_size': self.config['risk']['max_position_size'],
                    'kelly_fraction': self.config['risk']['kelly_fraction'],
                    'volatility_scale': True
                }
            )
            
            if position.size <= 0:
                return
            
            # Create order request
            order = OrderRequest(
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY if prediction.direction == "LONG" else OrderSide.SELL,
                volume=position.size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Execute order
            result = await self.trading_session.execute_order(order)
            
            if result.success:
                # Log trade to database
                await self.db_handler.add_trade({
                    "pair": symbol,
                    "signal_type": "ENSEMBLE",
                    "direction": prediction.direction,
                    "entry_price": current_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "position_size": position.size,
                    "prob": prediction.probability,
                    "features_json": {
                        "technical": analysis,
                        "ml": prediction.__dict__,
                        "news": news_probs
                    }
                })
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")

    async def _get_account_info(self) -> Dict[str, float]:
        """Get comprehensive account information including balance, equity, margin"""
        try:
            if not self.trading_session:
                return {
                    'balance': 0.0,
                    'equity': 0.0,
                    'margin': 0.0,
                    'free_margin': 0.0,
                    'margin_level': 0.0,
                    'leverage': 1.0
                }
            
            info = await self.trading_session.get_account_info()
            
            # Cache account info for rapid access
            self._account_info_cache = {
                'balance': float(info['balance']),
                'equity': float(info['equity']),
                'margin': float(info['margin']),
                'free_margin': float(info['margin_free']),
                'margin_level': float(info['margin_level']),
                'leverage': float(info['leverage'])
            }
            
            # Update risk metrics based on account state
            await self.risk_guardian.update_risk_metrics(
                balance=self._account_info_cache['balance'],
                equity=self._account_info_cache['equity'],
                margin_level=self._account_info_cache['margin_level']
            )
            
            return self._account_info_cache
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            if hasattr(self, '_account_info_cache'):
                self.logger.info("Using cached account info")
                return self._account_info_cache
            return {
                'balance': 0.0,
                'equity': 0.0,
                'margin': 0.0,
                'free_margin': 0.0,
                'margin_level': 0.0,
                'leverage': 1.0
            }

    async def shutdown(self, signal = None):
        """Gracefully shut down the system"""
        if signal:
            self.logger.info(f"Received shutdown signal: {signal}")
        
        self.running = False
        
        try:
            # Cancel all tasks
            for task in self.update_tasks:
                task.cancel()
            
            # Close all positions if in demo mode
            if self.trading_session:
                await self.trading_session.close_all_positions()
                await self.trading_session.shutdown()
            
            # Clean up resources
            if self.db_handler:
                await self.db_handler.cleanup_old_data()
            
            self.logger.info("System shut down successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
        
        finally:
            sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AstraTrade Ultra AI")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Run in demo mode"
    )
    args = parser.parse_args()
    
    # Create and run trading system
    system = AstraTrade(args.config, args.demo)
    
    try:
        asyncio.run(system.start())
    except KeyboardInterrupt:
        asyncio.run(system.shutdown())

if __name__ == "__main__":
    main()