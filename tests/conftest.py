"""
PyTest configuration and fixtures for AstraTrade Ultra AI testing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import sqlite3
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.models import Base
from db.db_handler import DatabaseHandler
from ml.ensemble import MarketPredictor
from advanced.detectors import TechnicalDetector
from agents.news_calendar import NewsCalendar
from risk.guardian import RiskGuardian
from rl.evolutor import TradingEnvironment, StrategyEvolutor

@pytest.fixture(scope="session")
def config():
    """Load test configuration"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def test_db():
    """Create test database"""
    db_path = "test.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create test database
    conn = sqlite3.connect(db_path)
    conn.close()
    
    yield f"sqlite:///{db_path}"
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest.fixture(scope="session")
async def db_handler(test_db):
    """Initialize database handler"""
    handler = DatabaseHandler(test_db)
    await handler.init_db()
    return handler

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=100),
        end=datetime.now(),
        freq='H'
    )
    
    np.random.seed(42)
    price = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.02)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': price,
        'high': price * (1 + np.random.rand(len(dates)) * 0.002),
        'low': price * (1 - np.random.rand(len(dates)) * 0.002),
        'close': price * (1 + np.random.randn(len(dates)) * 0.001),
        'volume': np.random.randint(100, 1000, len(dates))
    }).set_index('timestamp')

@pytest.fixture
def technical_detector():
    """Initialize technical detector"""
    return TechnicalDetector()

@pytest.fixture
def market_predictor():
    """Initialize market predictor"""
    return MarketPredictor(
        sequence_length=60,
        batch_size=32,
        hidden_size=128,
        num_epochs=2,  # Reduced for testing
        model_path="test_models"
    )

@pytest.fixture
def risk_guardian():
    """Initialize risk guardian"""
    return RiskGuardian(
        base_risk=0.003,
        max_portfolio_risk=0.05,
        max_correlation=0.7,
        drawdown_limit=0.15
    )

@pytest.fixture
def trading_env():
    """Initialize trading environment"""
    return TradingEnvironment(
        initial_balance=100000.0,
        max_positions=5,
        risk_per_trade=0.02,
        feature_space_size=50
    )

@pytest.fixture
def strategy_evolutor(trading_env):
    """Initialize strategy evolutor"""
    return StrategyEvolutor(
        model_path="test_models/rl",
        initial_balance=100000.0,
        training_episodes=10  # Reduced for testing
    )

@pytest.fixture
def news_calendar():
    """Initialize news calendar"""
    return NewsCalendar(
        alpha_vantage_key="test_key",
        base_currency="USD",
        supported_pairs=["EUR/USD", "GBP/USD", "USD/JPY"]
    )