"""
Unit tests for Adaptive Learning Agent
"""
import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from agents.adaptive_agent import AdaptiveLearningAgent, MarketState
from db.models import MarketRegime, AdaptiveStrategy, LearningHistory

@pytest.fixture
def mock_db_session():
    """Create mock database session"""
    session = Mock(spec=Session)
    session.query.return_value.filter_by.return_value.first.return_value = None
    return session

@pytest.fixture
def agent(mock_db_session, tmp_path):
    """Create test agent instance"""
    return AdaptiveLearningAgent(mock_db_session, str(tmp_path / "models"))

@pytest.fixture
def market_state():
    """Create test market state"""
    return MarketState(
        volatility=0.5,
        trend_strength=0.7,
        support_resistance_levels=[100.0, 101.0, 102.0],
        key_price_levels={'support': 100.0, 'resistance': 102.0},
        current_pattern='BULLISH_FLAG'
    )

class TestAdaptiveLearningAgent:
    """Test suite for AdaptiveLearningAgent"""

    def test_init_strategy(self, agent, mock_db_session):
        """Test strategy initialization"""
        symbol = "EURUSD"
        
        # Test successful initialization
        agent.init_strategy(symbol)
        assert symbol in agent.classifiers
        assert symbol in agent.scalers
        assert isinstance(agent.performance_history[symbol], list)
        
        # Verify database interactions
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    def test_market_regime_classification(self, agent, market_state):
        """Test market regime classification"""
        regime = agent._classify_regime(market_state)
        assert isinstance(regime, str)
        assert regime in ['VOLATILE', 'VOLATILE_TREND', 'TRENDING', 'RANGING', 'PATTERN_FORMING', 'UNDEFINED']

    @pytest.mark.parametrize("volatility,trend_strength,expected_regime", [
        (0.9, 0.2, 'VOLATILE'),
        (0.7, 0.8, 'VOLATILE_TREND'),
        (0.3, 0.8, 'TRENDING'),
        (0.3, 0.3, 'RANGING')
    ])
    def test_regime_classification_scenarios(self, agent, market_state, volatility, trend_strength, expected_regime):
        """Test different market regime scenarios"""
        market_state.volatility = volatility
        market_state.trend_strength = trend_strength
        regime = agent._classify_regime(market_state)
        assert regime == expected_regime

    @pytest.mark.benchmark
    def test_adaptation_performance(self, agent, market_state, benchmark):
        """Test adaptation performance"""
        symbol = "EURUSD"
        agent.init_strategy(symbol)
        
        def adapt_iteration():
            return agent.adapt_to_market_state(symbol, market_state)
        
        # Measure adaptation time
        result = benchmark(adapt_iteration)
        assert result.stats.mean < 0.1  # Should complete within 100ms

    def test_strategy_evolution(self, agent, mock_db_session):
        """Test strategy evolution logic"""
        symbol = "EURUSD"
        agent.init_strategy(symbol)
        
        # Simulate successful trading history
        agent.performance_history[symbol] = [0.1] * 50 + [0.2] * 50
        
        # Test evolution with good performance
        agent._evolve_strategy(symbol, performance_threshold=0.6)
        
        # Verify strategy adaptation
        mock_db_session.query.assert_called()
        mock_db_session.commit.assert_called()

    def test_parameter_optimization(self, agent):
        """Test parameter optimization logic"""
        current_params = {
            'risk_adjustment': 1.0,
            'confidence_threshold': 0.7,
            'timeout_adjustment': 1.0
        }
        
        # Test optimization with good performance
        optimized = agent._optimize_parameters(
            current_params,
            win_rate=0.8,
            avg_profit=0.15,
            regime='TRENDING'
        )
        
        assert optimized['risk_adjustment'] > current_params['risk_adjustment']
        assert optimized['confidence_threshold'] <= current_params['confidence_threshold']

    @pytest.mark.asyncio
    async def test_concurrent_adaptation(self, agent, market_state):
        """Test concurrent adaptation capabilities"""
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        # Initialize strategies
        for symbol in symbols:
            agent.init_strategy(symbol)
        
        # Test concurrent adaptations
        import asyncio
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                asyncio.to_thread(agent.adapt_to_market_state, symbol, market_state)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        assert len(results) == len(symbols)
        for result in results:
            assert isinstance(result, dict)

    def test_error_handling(self, agent, market_state):
        """Test error handling and recovery"""
        symbol = "EURUSD"
        agent.init_strategy(symbol)
        
        # Test database error handling
        with patch.object(agent.db_session, 'commit') as mock_commit:
            mock_commit.side_effect = Exception("Database error")
            
            # Should handle database error gracefully
            with pytest.raises(Exception):
                agent.adapt_to_market_state(symbol, market_state)

    def test_model_persistence(self, agent, market_state, tmp_path):
        """Test model saving and loading"""
        symbol = "EURUSD"
        agent.init_strategy(symbol)
        
        # Train and save model
        agent.adapt_to_market_state(symbol, market_state)
        agent._save_model_state(symbol)
        
        # Verify files exist
        assert (tmp_path / "models" / f"{symbol}_model.joblib").exists()
        assert (tmp_path / "models" / f"{symbol}_scaler.joblib").exists()

        # Test loading saved model
        new_agent = AdaptiveLearningAgent(agent.db_session, str(tmp_path / "models"))
        new_agent.init_strategy(symbol)
        assert symbol in new_agent.classifiers
        assert symbol in new_agent.scalers