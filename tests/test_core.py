"""
Core system tests for AstraTrade Ultra AI
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from db.models import Trade, Signal, Model, Config
from advanced.detectors import Zone, ZoneType
from ml.ensemble import PredictionResult
from agents.news_calendar import NewsEvent, EventImpact

@pytest.mark.asyncio
async def test_database_operations(db_handler):
    """Test basic database operations"""
    # Test trade creation
    trade_data = {
        "pair": "EUR/USD",
        "signal_type": "SMC_BREAK",
        "direction": "LONG",
        "entry_price": 1.1000,
        "stop_loss": 1.0990,
        "take_profit": 1.1020,
        "position_size": 1.0,
        "prob": 0.82
    }
    
    trade = await db_handler.add_trade(trade_data)
    assert trade.pair == "EUR/USD"
    assert trade.signal_type == "SMC_BREAK"
    
    # Test trade retrieval
    trades = await db_handler.get_trades(pair="EUR/USD")
    assert len(trades) == 1
    assert trades[0].entry_price == 1.1000
    
    # Test config operations
    await db_handler.set_config(
        "risk_settings",
        {"base_risk": 0.003, "max_risk": 0.05},
        "Risk management settings"
    )
    
    config = await db_handler.get_config("risk_settings")
    assert config["base_risk"] == 0.003
    assert config["max_risk"] == 0.05

def test_technical_analysis(technical_detector, sample_market_data):
    """Test technical analysis functionality"""
    # Test FVG detection
    fvgs = technical_detector.detect_fair_value_gaps(sample_market_data)
    assert isinstance(fvgs, list)
    
    # Test orderblock detection
    orderblocks = technical_detector.detect_orderblocks(sample_market_data)
    assert isinstance(orderblocks, list)
    
    # Test liquidity levels
    levels = technical_detector.identify_liquidity_levels(sample_market_data)
    assert isinstance(levels, list)
    
    # Test market structure analysis
    analysis = technical_detector.analyze_market_structure(sample_market_data)
    assert isinstance(analysis, dict)
    assert all(k in analysis for k in ['fvgs', 'orderblocks', 'liquidity_levels'])

def test_ml_predictions(market_predictor, sample_market_data):
    """Test machine learning predictions"""
    # Prepare features
    features = [
        'open', 'high', 'low', 'close', 'volume'
    ]
    
    # Train the model (minimal training for testing)
    metrics = market_predictor.train(
        sample_market_data,
        target_column='close',
        feature_columns=features
    )
    
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ['accuracy', 'precision', 'recall'])
    
    # Test prediction
    latest_data = sample_market_data.tail(60)
    prediction = market_predictor.predict(latest_data[features])
    
    assert isinstance(prediction, PredictionResult)
    assert 0 <= prediction.probability <= 1
    assert prediction.direction in ['LONG', 'SHORT']
    assert isinstance(prediction.features_importance, dict)

@pytest.mark.asyncio
async def test_news_analysis(news_calendar):
    """Test news analysis and sentiment"""
    # Create test event
    event = NewsEvent(
        timestamp=datetime.now(),
        currency="USD",
        event="FOMC Statement",
        impact=EventImpact.CRITICAL,
        actual=None,
        forecast=None,
        previous=None
    )
    
    # Test news risk calculation
    risk_score, events = await news_calendar.calculate_news_risk("EUR/USD")
    assert isinstance(risk_score, float)
    assert 0 <= risk_score <= 1
    
    # Test probability blending
    prob_metrics = await news_calendar.get_trade_probability(
        "EUR/USD",
        technical_prob=0.8,
        direction="LONG"
    )
    
    assert isinstance(prob_metrics, dict)
    assert all(k in prob_metrics for k in [
        'original_prob', 'adjusted_prob', 'news_risk'
    ])
    assert prob_metrics['adjusted_prob'] <= prob_metrics['original_prob']

def test_risk_management(risk_guardian, sample_market_data):
    """Test risk management system"""
    # Test position sizing
    position = risk_guardian.calculate_position_size(
        pair="EUR/USD",
        entry_price=1.1000,
        stop_loss=1.0990,
        account_balance=100000.0,
        win_probability=0.8
    )
    
    assert position.size > 0
    assert position.risk_amount <= 100000.0 * 0.003  # Max 0.3% risk
    assert 0 <= position.kelly_fraction <= 1
    
    # Test portfolio metrics
    metrics = risk_guardian.update_portfolio_metrics(
        positions={
            "1": {
                "pair": "EUR/USD",
                "size": 1.0,
                "risk_amount": 300.0
            }
        },
        returns={"EUR/USD": 0.001},
        prices={"EUR/USD": sample_market_data}
    )
    
    assert hasattr(metrics, 'sharpe_ratio')
    assert hasattr(metrics, 'max_drawdown')
    assert hasattr(metrics, 'var_99')

def test_rl_strategy(trading_env, strategy_evolutor):
    """Test reinforcement learning strategy"""
    # Test environment reset
    obs = trading_env.reset()
    assert isinstance(obs, np.ndarray)
    
    # Test step execution
    action = np.array([0.5, 0.1, 0.2])  # Example action
    obs, reward, done, info = trading_env.step(action)
    
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    # Test strategy training (minimal)
    strategy_evolutor.train(continue_training=False)
    
    # Test strategy evaluation
    metrics = strategy_evolutor.evaluate(num_episodes=2)
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in [
        'mean_reward', 'mean_sharpe', 'mean_drawdown'
    ])

def test_system_integration(
    db_handler,
    technical_detector,
    market_predictor,
    risk_guardian,
    news_calendar,
    sample_market_data
):
    """Test system integration and workflow"""
    # Simulate complete trading workflow
    async def trading_workflow():
        # 1. Technical Analysis
        analysis = technical_detector.analyze_market_structure(sample_market_data)
        assert isinstance(analysis, dict)
        
        # 2. ML Prediction
        features = sample_market_data[['open', 'high', 'low', 'close', 'volume']]
        prediction = market_predictor.predict(features)
        assert isinstance(prediction, PredictionResult)
        
        # 3. News Analysis
        news_probs = await news_calendar.get_trade_probability(
            "EUR/USD",
            prediction.probability,
            prediction.direction
        )
        assert isinstance(news_probs, dict)
        
        # 4. Risk Calculation
        position = risk_guardian.calculate_position_size(
            pair="EUR/USD",
            entry_price=sample_market_data['close'].iloc[-1],
            stop_loss=sample_market_data['close'].iloc[-1] * 0.999,
            account_balance=100000.0,
            win_probability=prediction.probability
        )
        assert position.size > 0
        
        # 5. Database Recording
        trade_data = {
            "pair": "EUR/USD",
            "signal_type": "ML_ENSEMBLE",
            "direction": prediction.direction,
            "entry_price": sample_market_data['close'].iloc[-1],
            "stop_loss": sample_market_data['close'].iloc[-1] * 0.999,
            "take_profit": sample_market_data['close'].iloc[-1] * 1.002,
            "position_size": position.size,
            "prob": prediction.probability
        }
        trade = await db_handler.add_trade(trade_data)
        assert trade.pair == "EUR/USD"
    
    # Run workflow
    import asyncio
    asyncio.run(trading_workflow())