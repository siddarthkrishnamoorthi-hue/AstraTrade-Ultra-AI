"""
Adaptive Learning Agent for dynamic strategy optimization
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
import joblib
import os
from pathlib import Path
import logging

from db.models import MarketRegime, AdaptiveStrategy, LearningHistory

# Setup logging
logger = logging.getLogger(__name__)

class AdaptiveAgentError(Exception):
    """Base exception for adaptive agent errors"""
    pass

class ModelNotFoundError(AdaptiveAgentError):
    """Raised when model file is not found or is invalid"""
    pass

class DatabaseError(AdaptiveAgentError):
    """Raised when database operations fail"""
    pass

@dataclass
class MarketState:
    """Represents current market conditions"""
    volatility: float
    trend_strength: float
    support_resistance_levels: List[float]
    key_price_levels: Dict[str, float]
    current_pattern: Optional[str]

    def to_dict(self) -> Dict:
        """Convert state to dictionary for storage"""
        return {
            'volatility': self.volatility,
            'trend_strength': self.trend_strength,
            'support_resistance_levels': self.support_resistance_levels,
            'key_price_levels': self.key_price_levels,
            'current_pattern': self.current_pattern
        }

class AdaptiveLearningAgent:
    """Agent that adapts trading strategies based on market conditions"""
    
    def __init__(self, db_session: Session, model_path: str = "models/adaptive"):
        """Initialize the adaptive learning agent
        
        Args:
            db_session: SQLAlchemy database session
            model_path: Path to store ML models
        """
        self.db_session = db_session
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.classifiers: Dict[str, RandomForestClassifier] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.min_samples_for_adaptation = 100
        
    def init_strategy(self, symbol: str) -> None:
        """Initialize or load strategy for symbol
        
        Args:
            symbol: Trading pair symbol
            
        Raises:
            ModelNotFoundError: If model files cannot be loaded
        """
        try:
            model_file = self.model_path / f"{symbol}_model.joblib"
            scaler_file = self.model_path / f"{symbol}_scaler.joblib"
            
            if model_file.exists() and scaler_file.exists():
                self.classifiers[symbol] = joblib.load(model_file)
                self.scalers[symbol] = StandardScaler()  # Always create new scaler for safety
                
                # Validate loaded model
                if not hasattr(self.classifiers[symbol], 'predict_proba'):
                    raise ModelNotFoundError(f"Invalid model format for {symbol}")
            else:
                self.classifiers[symbol] = RandomForestClassifier(
                    n_estimators=100,
                    min_samples_leaf=5,  # More conservative
                    max_features='sqrt',
                    random_state=42,
                    class_weight='balanced'  # Handle imbalanced data
                )
                self.scalers[symbol] = StandardScaler()
            
            self.performance_history[symbol] = []
            
        except Exception as e:
            raise ModelNotFoundError(f"Failed to initialize strategy for {symbol}: {str(e)}")
        
        # Initialize strategy in database if not exists
        self._init_db_strategy(symbol)
        
    def _init_db_strategy(self, symbol: str) -> None:
        """Initialize strategy record in database
        
        Args:
            symbol: Trading pair symbol
            
        Raises:
            DatabaseError: If database operations fail
        """
        try:
            strategy = self.db_session.query(AdaptiveStrategy).filter_by(
                pair=symbol
            ).first()
            
            if not strategy:
                logger.info(f"Initializing new strategy for {symbol}")
                strategy = AdaptiveStrategy(
                    pair=symbol,
                    strategy_type='ADAPTIVE',
                    parameters={
                        'risk_adjustment': 1.0,
                        'confidence_threshold': 0.7,
                        'min_samples': self.min_samples_for_adaptation
                    },
                    confidence_score=0.5
                )
                self.db_session.add(strategy)
                self.db_session.commit()
                logger.debug(f"Strategy initialized for {symbol}")
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to initialize strategy for {symbol}: {str(e)}")
            raise DatabaseError(f"Database initialization failed for {symbol}: {str(e)}")
            
    def adapt_to_market_state(self, symbol: str, state: MarketState) -> Dict[str, float]:
        """Adapt strategy parameters based on current market state
        
        Args:
            symbol: Trading pair symbol
            state: Current market state
            
        Returns:
            Dict of adjusted parameters
        """
        # Store market regime
        regime = MarketRegime(
            pair=symbol,
            regime_type=self._classify_regime(state),
            volatility=state.volatility,
            trend_strength=state.trend_strength,
            support_resistance=state.key_price_levels
        )
        self.db_session.add(regime)
        
        # Extract features
        features = np.array([
            state.volatility,
            state.trend_strength,
            np.mean(state.support_resistance_levels),
            len(state.support_resistance_levels),
            float(bool(state.current_pattern))
        ]).reshape(1, -1)
        
        if symbol not in self.classifiers:
            self.init_strategy(symbol)
            
        # Scale features
        if not self.scalers[symbol].n_samples_seen_:
            self.scalers[symbol].fit(features)
        scaled_features = self.scalers[symbol].transform(features)
        
        # Get strategy from database
        strategy = self.db_session.query(AdaptiveStrategy).filter_by(
            pair=symbol
        ).first()
        
        if len(self.performance_history[symbol]) >= self.min_samples_for_adaptation:
            prediction = self.classifiers[symbol].predict_proba(scaled_features)[0]
            
            params = {
                'risk_adjustment': float(prediction[0]),
                'timeout_adjustment': float(prediction[1]),
                'confidence_threshold': float(prediction[2])
            }
            
            # Update strategy parameters
            strategy.parameters = params
            strategy.updated_at = datetime.utcnow()
            self.db_session.commit()
            
            return params
            
        return strategy.parameters
        
    def _classify_regime(self, state: MarketState) -> str:
        """Classify market regime based on state and historical patterns
        
        This method analyzes multiple factors to determine market regime:
        1. Volatility levels
        2. Trend strength
        3. Support/resistance structure
        4. Pattern recognition
        
        Returns:
            str: Market regime classification
        """
        # Calculate volatility score (0-1)
        vol_score = min(state.volatility / 0.8, 1.0)
        
        # Calculate trend strength score (0-1)
        trend_score = min(state.trend_strength / 0.7, 1.0)
        
        # Analyze price structure
        has_clear_levels = len(state.support_resistance_levels) >= 3
        has_pattern = bool(state.current_pattern)
        
        # Composite regime analysis
        if vol_score > 0.8 and trend_score < 0.3:
            return 'VOLATILE'  # High volatility, low trend - choppy market
        elif vol_score > 0.6 and trend_score > 0.7:
            return 'VOLATILE_TREND'  # High volatility with strong trend
        elif trend_score > 0.7 and has_clear_levels:
            return 'TRENDING'  # Strong trend with clear levels
        elif vol_score < 0.4 and has_clear_levels:
            return 'RANGING'  # Low volatility with clear levels
        elif has_pattern:
            return 'PATTERN_FORMING'  # Clear pattern development
        return 'UNDEFINED'  # Default case
        
    def update_performance(self, symbol: str, profit: float) -> None:
        """Update performance history and retrain if needed
        
        Args:
            symbol: Trading pair symbol
            profit: Trade profit/loss
        """
        self.performance_history[symbol].append(profit)
        
        # Update strategy performance in database
        strategy = self.db_session.query(AdaptiveStrategy).filter_by(
            pair=symbol
        ).first()
        
        strategy.trades_count += 1
        strategy.avg_profit = np.mean(self.performance_history[symbol][-100:])
        strategy.success_rate = np.mean([p > 0 for p in self.performance_history[symbol][-100:]])
        
        # Store learning history
        history = LearningHistory(
            pair=symbol,
            market_state=self._get_current_state(symbol).to_dict(),
            action_taken=strategy.parameters,
            outcome={'profit': profit},
            reward=self._calculate_reward(profit)
        )
        self.db_session.add(history)
        
        # Retrain model if enough samples
        if len(self.performance_history[symbol]) >= self.min_samples_for_adaptation:
            self._retrain_model(symbol)
            
        self.db_session.commit()
            
    def _calculate_reward(self, profit: float) -> float:
        """Calculate reward value for learning"""
        return np.tanh(profit)  # Normalized reward between -1 and 1
        
    def _get_current_state(self, symbol: str) -> MarketState:
        """Get current market state from database"""
        regime = self.db_session.query(MarketRegime).filter_by(
            pair=symbol
        ).order_by(MarketRegime.timestamp.desc()).first()
        
        if regime:
            return MarketState(
                volatility=regime.volatility,
                trend_strength=regime.trend_strength,
                support_resistance_levels=list(regime.support_resistance.values()),
                key_price_levels=regime.support_resistance,
                current_pattern=None
            )
        return MarketState(0.0, 0.0, [], {}, None)
        
    def _retrain_model(self, symbol: str) -> None:
        """Retrain model with recent performance data
        
        Args:
            symbol: Trading pair symbol
            
        Raises:
            DatabaseError: If database operations fail
            ModelNotFoundError: If model training fails
        """
        try:
            if len(self.performance_history[symbol]) < self.min_samples_for_adaptation:
                logger.debug(f"Insufficient samples for {symbol}, skipping retraining")
                return
                
            # Get recent learning history
            history = self.db_session.query(LearningHistory).filter_by(
                pair=symbol
            ).order_by(
                LearningHistory.timestamp.desc()
            ).limit(self.min_samples_for_adaptation).all()
            
            if not history:
                logger.debug(f"No learning history for {symbol}, skipping retraining")
                return
                
            # Prepare training data
            X = []
            y = []
            
            for entry in history:
                state = entry.market_state
                try:
                    X.append([
                        state['volatility'],
                        state['trend_strength'],
                        np.mean(state['support_resistance_levels']),
                        len(state['support_resistance_levels']),
                        float(bool(state.get('current_pattern')))
                    ])
                    y.append(int(entry.reward > 0))
                except (KeyError, TypeError) as e:
                    logger.warning(f"Invalid state data for {symbol}: {str(e)}")
                    continue
                    
            if not X or not y:
                logger.warning(f"No valid training data for {symbol}")
                return
                
            X = np.array(X)
            y = np.array(y)
            
            if len(np.unique(y)) < 2:
                logger.warning(f"Single class in training data for {symbol}")
                return
                
            # Update scaler and model with validation
            try:
                self.scalers[symbol].fit(X)
                X_scaled = self.scalers[symbol].transform(X)
                self.classifiers[symbol].fit(X_scaled, y)
                
                # Validate model performance
                train_score = self.classifiers[symbol].score(X_scaled, y)
                logger.info(f"Model training score for {symbol}: {train_score:.3f}")
                
                # Evaluate model improvement
                if train_score > 0.6:  # Good performance threshold
                    # Save updated models
                    self._save_model_state(symbol)
                    
                    # Evolve strategy based on performance
                    self._evolve_strategy(symbol, performance_threshold=0.6)
                    
                    logger.info(f"Model successfully trained and evolved for {symbol}")
                else:
                    logger.warning(f"Model performance below threshold for {symbol}: {train_score:.3f}")
                    # Trigger adaptive parameter adjustment for poor performance
                    self._evolve_strategy(symbol, performance_threshold=0.4)
                
            except Exception as e:
                logger.error(f"Model training failed for {symbol}: {str(e)}")
                raise ModelNotFoundError(f"Failed to train model for {symbol}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Retraining failed for {symbol}: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Cleanup resources and save final state"""
        try:
            self.db_session.commit()
            for symbol in self.classifiers:
                self._save_model_state(symbol)
        except Exception as e:
            self.db_session.rollback()
            raise RuntimeError(f"Failed to cleanup resources: {str(e)}")
            
    def _evolve_strategy(self, symbol: str, performance_threshold: float = 0.6) -> None:
        """Evolve trading strategy based on performance and market conditions
        
        Args:
            symbol: Trading pair symbol
            performance_threshold: Minimum performance score to trigger evolution
        """
        try:
            # Get recent performance
            recent_trades = self.performance_history[symbol][-100:]
            if not recent_trades:
                return
                
            win_rate = sum(1 for p in recent_trades if p > 0) / len(recent_trades)
            avg_profit = np.mean(recent_trades)
            
            # Get current strategy
            strategy = self.db_session.query(AdaptiveStrategy).filter_by(
                pair=symbol
            ).first()
            
            if not strategy:
                return
                
            # Analyze performance trends
            performance_improving = len(recent_trades) >= 20 and \
                np.mean(recent_trades[-10:]) > np.mean(recent_trades[:-10])
                
            # Get market regime distribution
            regimes = self.db_session.query(MarketRegime).filter_by(
                pair=symbol
            ).order_by(
                MarketRegime.timestamp.desc()
            ).limit(100).all()
            
            regime_counts = {}
            for r in regimes:
                regime_counts[r.regime_type] = regime_counts.get(r.regime_type, 0) + 1
                
            dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0] \
                if regime_counts else 'UNDEFINED'
                
            # Evolution logic
            if win_rate >= performance_threshold and performance_improving:
                # Strategy performing well - fine-tune parameters
                strategy.parameters = self._optimize_parameters(
                    strategy.parameters,
                    win_rate,
                    avg_profit,
                    dominant_regime
                )
                logger.info(f"Strategy evolved for {symbol} - performance improving")
            elif win_rate < performance_threshold:
                # Strategy underperforming - significant adaptation needed
                strategy.parameters = self._adapt_parameters(
                    strategy.parameters,
                    win_rate,
                    avg_profit,
                    dominant_regime
                )
                logger.info(f"Strategy adapted for {symbol} - addressing underperformance")
                
            strategy.updated_at = datetime.utcnow()
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Strategy evolution failed for {symbol}: {str(e)}")
            self.db_session.rollback()
            
    def _optimize_parameters(self, current_params: Dict, win_rate: float, 
                           avg_profit: float, regime: str) -> Dict:
        """Fine-tune strategy parameters based on performance
        
        Args:
            current_params: Current strategy parameters
            win_rate: Current win rate
            avg_profit: Average profit
            regime: Current market regime
        """
        params = current_params.copy()
        
        # Adjust risk based on performance
        if win_rate > 0.7:
            params['risk_adjustment'] = min(params['risk_adjustment'] * 1.1, 1.5)
        
        # Adjust confidence threshold based on market regime
        if regime == 'VOLATILE':
            params['confidence_threshold'] = max(params['confidence_threshold'], 0.8)
        elif regime == 'TRENDING':
            params['confidence_threshold'] = min(params['confidence_threshold'], 0.6)
            
        # Optimize timeout based on profit characteristics
        if avg_profit > 0:
            params['timeout_adjustment'] = max(params['timeout_adjustment'] * 1.05, 0.5)
        else:
            params['timeout_adjustment'] = min(params['timeout_adjustment'] * 0.95, 2.0)
            
        return params
        
    def _adapt_parameters(self, current_params: Dict, win_rate: float, 
                         avg_profit: float, regime: str) -> Dict:
        """Adapt strategy parameters for underperforming scenarios
        
        Args:
            current_params: Current strategy parameters
            win_rate: Current win rate
            avg_profit: Average profit
            regime: Current market regime
        """
        params = current_params.copy()
        
        # Significant adaptations for poor performance
        if win_rate < 0.4:
            params['risk_adjustment'] = max(params['risk_adjustment'] * 0.7, 0.3)
            params['confidence_threshold'] = min(params['confidence_threshold'] * 1.2, 0.9)
            
        # Regime-specific adaptations
        if regime == 'VOLATILE':
            params['risk_adjustment'] = max(params['risk_adjustment'] * 0.8, 0.2)
        elif regime == 'RANGING':
            params['timeout_adjustment'] = min(params['timeout_adjustment'] * 1.2, 2.0)
            
        return params
        
    def _save_model_state(self, symbol: str) -> None:
        """Save model state to disk and evolve strategy if needed
        
        Args:
            symbol: Trading pair symbol
        """
        try:
            # Save classifier
            model_file = self.model_path / f"{symbol}_model.joblib"
            joblib.dump(self.classifiers[symbol], model_file)
            
            # Save scaler
            scaler_file = self.model_path / f"{symbol}_scaler.joblib"
            joblib.dump(self.scalers[symbol], scaler_file)
            
            # Update database
            strategy = self.db_session.query(AdaptiveStrategy).filter_by(
                pair=symbol
            ).first()
            if strategy:
                strategy.updated_at = datetime.utcnow()
                self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            raise RuntimeError(f"Failed to save model state for {symbol}: {str(e)}")
            
    def __enter__(self):
        """Context manager enter"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()