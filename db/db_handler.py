"""
Database handler for AstraTrade Ultra AI with async CRUD operations
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update, delete
from sqlalchemy.future import select as future_select

from .models import Base, Trade, Signal, Model, Config

class DatabaseHandler:
    def __init__(self, db_url: str = "sqlite+aiosqlite:///astratrade.db"):
        """Initialize database connection and sessions"""
        self.engine = create_async_engine(db_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self):
        """Create all database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def cleanup_old_data(self, days: int = 30):
        """Clean up old signals and temporary data"""
        cleanup_date = datetime.utcnow() - timedelta(days=days)
        async with self.async_session() as session:
            await session.execute(
                delete(Signal).where(Signal.timestamp < cleanup_date)
            )
            await session.commit()

    # Trade Operations
    async def add_trade(self, trade_data: Dict[str, Any]) -> Trade:
        """Add a new trade record"""
        async with self.async_session() as session:
            trade = Trade(**trade_data)
            session.add(trade)
            await session.commit()
            return trade

    async def update_trade(self, trade_id: int, update_data: Dict[str, Any]) -> Optional[Trade]:
        """Update an existing trade"""
        async with self.async_session() as session:
            result = await session.execute(
                select(Trade).where(Trade.id == trade_id)
            )
            trade = result.scalar_one_or_none()
            if trade:
                for key, value in update_data.items():
                    setattr(trade, key, value)
                await session.commit()
            return trade

    async def get_trades(
        self,
        pair: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Trade]:
        """Retrieve trades with optional filters"""
        query = select(Trade)
        if pair:
            query = query.where(Trade.pair == pair)
        if start_date:
            query = query.where(Trade.timestamp >= start_date)
        if end_date:
            query = query.where(Trade.timestamp <= end_date)
        
        async with self.async_session() as session:
            result = await session.execute(query)
            return result.scalars().all()

    # Signal Operations
    async def add_signal(self, signal_data: Dict[str, Any]) -> Signal:
        """Add a new market signal"""
        async with self.async_session() as session:
            signal = Signal(**signal_data)
            session.add(signal)
            await session.commit()
            return signal

    async def get_active_signals(self, pair: Optional[str] = None) -> List[Signal]:
        """Get active signals that haven't expired"""
        query = select(Signal).where(Signal.expiry > datetime.utcnow())
        if pair:
            query = query.where(Signal.pair == pair)
        
        async with self.async_session() as session:
            result = await session.execute(query)
            return result.scalars().all()

    # Model Operations
    async def save_model(self, model_data: Dict[str, Any]) -> Model:
        """Save or update a model"""
        async with self.async_session() as session:
            if 'name' in model_data and 'version' in model_data:
                # Check for existing model
                result = await session.execute(
                    select(Model).where(
                        Model.name == model_data['name'],
                        Model.version == model_data['version']
                    )
                )
                model = result.scalar_one_or_none()
                if model:
                    # Update existing model
                    for key, value in model_data.items():
                        setattr(model, key, value)
                else:
                    # Create new model
                    model = Model(**model_data)
                    session.add(model)
                await session.commit()
                return model
            raise ValueError("Model name and version required")

    async def get_active_model(self, name: str) -> Optional[Model]:
        """Get the latest active model by name"""
        async with self.async_session() as session:
            result = await session.execute(
                select(Model)
                .where(Model.name == name, Model.is_active == 1)
                .order_by(Model.created_at.desc())
            )
            return result.scalar_one_or_none()

    # Configuration Operations
    async def set_config(self, key: str, value: Any, description: Optional[str] = None):
        """Set a configuration value"""
        async with self.async_session() as session:
            result = await session.execute(
                select(Config).where(Config.key == key)
            )
            config = result.scalar_one_or_none()
            if config:
                config.value = value
                if description:
                    config.description = description
            else:
                config = Config(key=key, value=value, description=description)
                session.add(config)
            await session.commit()

    async def get_config(self, key: str) -> Optional[Any]:
        """Get a configuration value"""
        async with self.async_session() as session:
            result = await session.execute(
                select(Config).where(Config.key == key)
            )
            config = result.scalar_one_or_none()
            return config.value if config else None

    async def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values"""
        async with self.async_session() as session:
            result = await session.execute(select(Config))
            return {config.key: config.value for config in result.scalars().all()}