"""
Database models for AstraTrade Ultra AI using SQLAlchemy ORM
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, LargeBinary, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Trade(Base):
    """Trading transaction records with performance metrics"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    pair = Column(String(12), nullable=False)  # e.g., 'EUR/USD', 'XAUUSD'
    signal_type = Column(String(50))  # e.g., 'SMC_BREAK', 'ICT_JUDAS'
    direction = Column(String(4))  # 'LONG' or 'SHORT'
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)
    prob = Column(Float)  # ML prediction probability
    pnl = Column(Float)  # Realized profit/loss
    features_json = Column(JSON)  # Trading setup features
    exit_timestamp = Column(DateTime)
    exit_price = Column(Float)
    exit_reason = Column(String(50))  # e.g., 'TP_HIT', 'SL_HIT', 'MANUAL'

class Signal(Base):
    """Market analysis signals and predictions"""
    __tablename__ = 'signals'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    pair = Column(String(12), nullable=False)
    signal_type = Column(String(50))
    direction = Column(String(4))
    prob = Column(Float)  # Combined probability
    tech_prob = Column(Float)  # Technical probability
    fund_prob = Column(Float)  # Fundamental probability
    raw_data_json = Column(JSON)  # Raw signal data
    expiry = Column(DateTime)  # Signal validity period

class Model(Base):
    """ML/RL model metadata and weights"""
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    version = Column(String(20))
    type = Column(String(50))  # e.g., 'LSTM', 'XGB', 'PPO'
    weights_blob = Column(LargeBinary)
    metadata_json = Column(JSON)  # Model parameters, training metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)  # For model versioning

class Config(Base):
    """System configuration and parameters"""
    __tablename__ = 'config'

    key = Column(String(100), primary_key=True)
    value = Column(JSON)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    description = Column(String(500))

class MarketRegime(Base):
    """Market regime classifications and parameters"""
    __tablename__ = 'market_regimes'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    pair = Column(String(12), nullable=False)
    regime_type = Column(String(50))  # e.g., 'TRENDING', 'RANGING', 'VOLATILE'
    volatility = Column(Float)
    trend_strength = Column(Float)
    support_resistance = Column(JSON)  # Key price levels
    optimal_params = Column(JSON)  # Optimized parameters for this regime

class AdaptiveStrategy(Base):
    """Adaptive strategy performance and parameters"""
    __tablename__ = 'adaptive_strategies'

    id = Column(Integer, primary_key=True)
    pair = Column(String(12), nullable=False)
    strategy_type = Column(String(50))
    parameters = Column(JSON)  # Current strategy parameters
    performance_metrics = Column(JSON)  # Recent performance statistics
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    confidence_score = Column(Float)
    trades_count = Column(Integer, default=0)
    success_rate = Column(Float)
    avg_profit = Column(Float)
    
class LearningHistory(Base):
    """Historical learning data and outcomes"""
    __tablename__ = 'learning_history'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    pair = Column(String(12), nullable=False)
    market_state = Column(JSON)  # Market conditions
    action_taken = Column(JSON)  # Strategy decisions
    outcome = Column(JSON)  # Trade results
    reward = Column(Float)  # Learning reward value