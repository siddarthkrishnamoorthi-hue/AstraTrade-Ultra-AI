"""
MetaTrader 5 execution system with order management and Telegram integration
"""

import MetaTrader5 as mt5
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
import logging
from pathlib import Path
import json

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class OrderRequest:
    symbol: str
    order_type: OrderType
    side: OrderSide
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    magic: int = 123456
    comment: str = "AstraTrade Ultra AI"

@dataclass
class OrderResult:
    success: bool
    order_id: Optional[int]
    message: str
    details: Dict

class TradingSession:
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        risk_manager: Any,
        telegram_token: str,
        telegram_chat_id: str,
        demo_mode: bool = True
    ):
        self.symbols = symbols
        self.timeframes = timeframes
        self.risk_manager = risk_manager
        self.demo_mode = demo_mode
        
        # MT5 connection
        self.connected = False
        self.orders: Dict[int, Dict] = {}
        self.positions: Dict[int, Dict] = {}
        
        # Telegram setup
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.telegram_bot: Optional[Bot] = None
        self.telegram_app: Optional[Application] = None
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("trading_log.txt"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AstraTrade")

    async def initialize(self) -> bool:
        """
        Initialize MT5 connection and Telegram bot
        """
        # Initialize MT5
        if not mt5.initialize(
            path=None,  # Use default path
            login=0,    # Demo account
            password="",
            server="",
            timeout=30000
        ):
            self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        self.connected = True
        
        # Initialize Telegram bot
        try:
            self.telegram_bot = Bot(self.telegram_token)
            self.telegram_app = (
                Application.builder()
                .token(self.telegram_token)
                .build()
            )
            
            # Register command handlers
            self.telegram_app.add_handler(
                CommandHandler("status", self._handle_status_command)
            )
            self.telegram_app.add_handler(
                CommandHandler("positions", self._handle_positions_command)
            )
            
            # Start bot
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            await self.send_telegram_message("AstraTrade Ultra AI initialized")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Telegram initialization failed: {str(e)}")
            return False

    async def shutdown(self) -> None:
        """
        Clean shutdown of MT5 and Telegram connections
        """
        try:
            # Close all positions if in demo mode
            if self.demo_mode:
                await self.close_all_positions()
            
            # Shutdown Telegram
            if self.telegram_app:
                await self.telegram_app.stop()
                await self.telegram_app.shutdown()
            
            # Shutdown MT5
            if self.connected:
                mt5.shutdown()
                self.connected = False
            
            self.logger.info("Trading session shut down successfully")
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

    async def execute_order(self, request: OrderRequest) -> OrderResult:
        """
        Execute trade order with risk checks and logging
        """
        if not self.connected:
            return OrderResult(
                success=False,
                order_id=None,
                message="MT5 not connected",
                details={}
            )
        
        try:
            # Risk check
            risk_check = await self._check_risk_limits(request)
            if not risk_check['approved']:
                return OrderResult(
                    success=False,
                    order_id=None,
                    message=f"Risk check failed: {risk_check['reason']}",
                    details=risk_check
                )
            
            # Prepare order request
            mt5_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": request.symbol,
                "volume": request.volume,
                "type": self._get_mt5_order_type(request),
                "magic": request.magic,
                "comment": request.comment,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add price levels
            if request.price:
                mt5_request["price"] = request.price
            if request.stop_loss:
                mt5_request["sl"] = request.stop_loss
            if request.take_profit:
                mt5_request["tp"] = request.take_profit
            
            # Execute order
            if not self.demo_mode:
                result = mt5.order_send(mt5_request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    return OrderResult(
                        success=False,
                        order_id=None,
                        message=f"Order failed: {result.comment}",
                        details={"retcode": result.retcode}
                    )
                order_id = result.order
            else:
                # Simulate order in demo mode
                order_id = len(self.orders) + 1
            
            # Log order
            order_info = {
                "id": order_id,
                "symbol": request.symbol,
                "type": request.order_type.value,
                "side": request.side.value,
                "volume": request.volume,
                "price": request.price,
                "sl": request.stop_loss,
                "tp": request.take_profit,
                "timestamp": datetime.now().isoformat()
            }
            
            self.orders[order_id] = order_info
            
            # Send Telegram notification with chart
            await self._send_order_notification(order_info)
            
            return OrderResult(
                success=True,
                order_id=order_id,
                message="Order executed successfully",
                details=order_info
            )
        
        except Exception as e:
            self.logger.error(f"Order execution error: {str(e)}")
            return OrderResult(
                success=False,
                order_id=None,
                message=f"Execution error: {str(e)}",
                details={}
            )

    async def modify_order(
        self,
        order_id: int,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None
    ) -> OrderResult:
        """
        Modify existing order's stop loss and take profit
        """
        if not self.connected:
            return OrderResult(
                success=False,
                order_id=None,
                message="MT5 not connected",
                details={}
            )
        
        try:
            if not self.demo_mode:
                position = mt5.positions_get(ticket=order_id)
                if not position:
                    return OrderResult(
                        success=False,
                        order_id=None,
                        message="Position not found",
                        details={}
                    )
                
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": order_id,
                    "symbol": position[0].symbol
                }
                
                if new_sl is not None:
                    request["sl"] = new_sl
                if new_tp is not None:
                    request["tp"] = new_tp
                
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    return OrderResult(
                        success=False,
                        order_id=order_id,
                        message=f"Modification failed: {result.comment}",
                        details={"retcode": result.retcode}
                    )
            
            # Update order info
            if order_id in self.orders:
                if new_sl is not None:
                    self.orders[order_id]["sl"] = new_sl
                if new_tp is not None:
                    self.orders[order_id]["tp"] = new_tp
            
            return OrderResult(
                success=True,
                order_id=order_id,
                message="Order modified successfully",
                details=self.orders.get(order_id, {})
            )
        
        except Exception as e:
            self.logger.error(f"Order modification error: {str(e)}")
            return OrderResult(
                success=False,
                order_id=order_id,
                message=f"Modification error: {str(e)}",
                details={}
            )

    async def close_position(self, position_id: int) -> OrderResult:
        """
        Close a specific position
        """
        if not self.connected:
            return OrderResult(
                success=False,
                order_id=None,
                message="MT5 not connected",
                details={}
            )
        
        try:
            if not self.demo_mode:
                position = mt5.positions_get(ticket=position_id)
                if not position:
                    return OrderResult(
                        success=False,
                        order_id=None,
                        message="Position not found",
                        details={}
                    )
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": position_id,
                    "symbol": position[0].symbol,
                    "volume": position[0].volume,
                    "type": mt5.ORDER_TYPE_SELL if position[0].type == 0 else mt5.ORDER_TYPE_BUY,
                    "magic": position[0].magic,
                    "comment": "AstraTrade Ultra AI - Close",
                    "type_filling": mt5.ORDER_FILLING_IOC
                }
                
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    return OrderResult(
                        success=False,
                        order_id=position_id,
                        message=f"Close failed: {result.comment}",
                        details={"retcode": result.retcode}
                    )
            
            # Update position tracking
            if position_id in self.positions:
                del self.positions[position_id]
            
            return OrderResult(
                success=True,
                order_id=position_id,
                message="Position closed successfully",
                details={}
            )
        
        except Exception as e:
            self.logger.error(f"Position close error: {str(e)}")
            return OrderResult(
                success=False,
                order_id=position_id,
                message=f"Close error: {str(e)}",
                details={}
            )

    async def close_all_positions(self) -> List[OrderResult]:
        """
        Close all open positions
        """
        results = []
        if not self.demo_mode:
            positions = mt5.positions_get()
            if positions:
                for position in positions:
                    result = await self.close_position(position.ticket)
                    results.append(result)
        else:
            # Close simulated positions
            position_ids = list(self.positions.keys())
            for pos_id in position_ids:
                result = await self.close_position(pos_id)
                results.append(result)
        
        return results

    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 1000
    ) -> pd.DataFrame:
        """
        Get market data from MT5
        """
        if not self.connected:
            return pd.DataFrame()
        
        try:
            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            if not self.demo_mode:
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
                if rates is None:
                    return pd.DataFrame()
                
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                return df
            else:
                # Generate simulated data for demo mode
                return self._generate_demo_data(bars)
        
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return pd.DataFrame()

    async def _check_risk_limits(self, request: OrderRequest) -> Dict:
        """
        Check if order meets risk management criteria
        """
        try:
            # Get current portfolio state
            portfolio_value = self._get_portfolio_value()
            
            # Calculate position risk
            risk_amount = (
                abs(request.price - request.stop_loss)
                * request.volume
                if request.stop_loss
                else 0
            )
            
            risk_percent = risk_amount / portfolio_value if portfolio_value > 0 else 0
            
            # Check against risk limits
            if risk_percent > 0.02:  # 2% max risk per trade
                return {
                    "approved": False,
                    "reason": "Exceeds maximum trade risk",
                    "risk_percent": risk_percent
                }
            
            # Check total portfolio risk
            total_risk = sum(
                self._calculate_position_risk(pos)
                for pos in self.positions.values()
            )
            
            if (total_risk + risk_amount) / portfolio_value > 0.05:  # 5% max portfolio risk
                return {
                    "approved": False,
                    "reason": "Exceeds maximum portfolio risk",
                    "total_risk": total_risk + risk_amount
                }
            
            return {"approved": True, "risk_percent": risk_percent}
        
        except Exception as e:
            self.logger.error(f"Risk check error: {str(e)}")
            return {
                "approved": False,
                "reason": f"Risk check error: {str(e)}"
            }

    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        if not self.demo_mode:
            account_info = mt5.account_info()
            return account_info.balance if account_info else 0.0
        return 100000.0  # Demo mode default

    def _calculate_position_risk(self, position: Dict) -> float:
        """Calculate risk amount for a position"""
        return (
            abs(position.get('price', 0) - position.get('sl', 0))
            * position.get('volume', 0)
            if position.get('sl')
            else 0
        )

    def _get_mt5_order_type(self, request: OrderRequest) -> int:
        """Convert OrderType to MT5 order type"""
        if request.side == OrderSide.BUY:
            return {
                OrderType.MARKET: mt5.ORDER_TYPE_BUY,
                OrderType.LIMIT: mt5.ORDER_TYPE_BUY_LIMIT,
                OrderType.STOP: mt5.ORDER_TYPE_BUY_STOP
            }[request.order_type]
        else:
            return {
                OrderType.MARKET: mt5.ORDER_TYPE_SELL,
                OrderType.LIMIT: mt5.ORDER_TYPE_SELL_LIMIT,
                OrderType.STOP: mt5.ORDER_TYPE_SELL_STOP
            }[request.order_type]

    async def _send_order_notification(self, order_info: Dict) -> None:
        """
        Send order notification with chart to Telegram
        """
        try:
            # Generate chart
            chart_buffer = await self._generate_trade_chart(order_info)
            
            # Prepare message text
            message = (
                f"🔔 New Trade Alert\n"
                f"Symbol: {order_info['symbol']}\n"
                f"Type: {order_info['type']}\n"
                f"Side: {order_info['side']}\n"
                f"Volume: {order_info['volume']}\n"
                f"Price: {order_info['price']}\n"
                f"SL: {order_info['sl']}\n"
                f"TP: {order_info['tp']}\n"
                f"ID: {order_info['id']}"
            )
            
            # Send message with chart
            if self.telegram_bot and chart_buffer:
                await self.telegram_bot.send_photo(
                    chat_id=self.telegram_chat_id,
                    photo=chart_buffer,
                    caption=message
                )
            else:
                await self.send_telegram_message(message)
        
        except Exception as e:
            self.logger.error(f"Failed to send order notification: {str(e)}")

    async def _generate_trade_chart(self, order_info: Dict) -> Optional[io.BytesIO]:
        """
        Generate trade chart with entry, SL, and TP levels
        """
        try:
            # Get recent market data
            df = await self.get_market_data(
                order_info['symbol'],
                "H1",
                bars=100
            )
            
            if df.empty:
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot price
            ax.plot(df.index, df['close'], label='Price')
            
            # Add trade levels
            entry_time = pd.to_datetime(order_info['timestamp'])
            ax.axhline(
                y=order_info['price'],
                color='yellow',
                linestyle='--',
                label='Entry'
            )
            ax.axhline(
                y=order_info['sl'],
                color='red',
                linestyle='--',
                label='Stop Loss'
            )
            ax.axhline(
                y=order_info['tp'],
                color='green',
                linestyle='--',
                label='Take Profit'
            )
            
            # Customize chart
            ax.set_title(f"{order_info['symbol']} - {order_info['side']} Trade")
            ax.legend()
            ax.grid(True)
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plt.close()
            
            return buffer
        
        except Exception as e:
            self.logger.error(f"Failed to generate trade chart: {str(e)}")
            return None

    def _generate_demo_data(self, bars: int) -> pd.DataFrame:
        """Generate simulated market data for demo mode"""
        index = pd.date_range(
            end=datetime.now(),
            periods=bars,
            freq='H'
        )
        
        # Generate random walk prices
        np.random.seed(42)  # For reproducibility
        price = 100 * (1 + np.random.randn(bars).cumsum() * 0.02)
        
        return pd.DataFrame({
            'open': price,
            'high': price * (1 + np.random.rand(bars) * 0.002),
            'low': price * (1 - np.random.rand(bars) * 0.002),
            'close': price * (1 + np.random.randn(bars) * 0.001),
            'tick_volume': np.random.randint(100, 1000, bars),
            'spread': np.random.randint(2, 5, bars),
            'real_volume': np.random.randint(1000, 10000, bars)
        }, index=index)

    async def send_telegram_message(self, message: str) -> None:
        """
        Send message to Telegram
        """
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id,
                    text=message
                )
            except Exception as e:
                self.logger.error(f"Telegram message error: {str(e)}")

    async def _handle_status_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /status command"""
        status = (
            f"📊 AstraTrade Ultra AI Status\n"
            f"Connected: {self.connected}\n"
            f"Mode: {'Demo' if self.demo_mode else 'Live'}\n"
            f"Active Positions: {len(self.positions)}\n"
            f"Total Orders: {len(self.orders)}\n"
            f"Portfolio Value: ${self._get_portfolio_value():,.2f}"
        )
        await update.message.reply_text(status)

    async def _handle_positions_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /positions command"""
        if not self.positions:
            await update.message.reply_text("No active positions")
            return
        
        positions_text = "📈 Active Positions:\n\n"
        for pos_id, pos in self.positions.items():
            positions_text += (
                f"ID: {pos_id}\n"
                f"Symbol: {pos['symbol']}\n"
                f"Side: {pos['side']}\n"
                f"Volume: {pos['volume']}\n"
                f"Entry: {pos['price']}\n"
                f"Current P/L: ${self._calculate_position_pnl(pos):,.2f}\n\n"
            )
        
        await update.message.reply_text(positions_text)