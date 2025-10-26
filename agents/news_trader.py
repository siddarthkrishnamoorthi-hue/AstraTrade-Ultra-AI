"""
Professional News Trading Module for AstraTrade Ultra AI
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .news_calendar import NewsEvent, EventImpact
from utils.symbol_utils import get_pip_value

class NewsTradingStrategy(Enum):
    DEVIATION = "deviation"  # Trade based on deviation from forecast
    MOMENTUM = "momentum"    # Trade in the direction of initial spike
    REVERSAL = "reversal"   # Trade potential reversals after initial spike

@dataclass
class NewsTradeSetup:
    event: NewsEvent
    strategy: NewsTradingStrategy
    entry_price: float
    stop_loss: float
    take_profit: float
    direction: int  # 1 for long, -1 for short
    confidence: float
    timestamp: datetime

class NewsTrader:
    def __init__(self, risk_manager=None):
        self.risk_manager = risk_manager
        self.pre_news_volatility: Dict[str, float] = {}
        self.historical_impacts: Dict[str, List[float]] = {}
        
    def calculate_news_direction(self, event: NewsEvent) -> Tuple[int, float]:
        """
        Calculate expected market direction and confidence level based on news event
        Returns: (direction, confidence)
        direction: 1 for bullish, -1 for bearish
        confidence: 0.0 to 1.0
        """
        if event.actual is None or event.forecast is None:
            return 0, 0.0
            
        # Calculate deviation from forecast
        deviation = event.actual - event.forecast
        
        # Calculate base direction
        direction = np.sign(deviation)
        
        # Calculate confidence based on multiple factors
        confidence_factors = [
            # Deviation significance
            min(abs(deviation / event.forecast) if event.forecast != 0 else abs(deviation), 1.0),
            
            # Impact weight
            self.get_impact_weight(event.impact),
            
            # Sentiment alignment
            self.get_sentiment_alignment(event.sentiment_score, direction),
            
            # Historical reliability
            self.get_historical_reliability(event.event)
        ]
        
        confidence = np.mean(confidence_factors)
        return int(direction), confidence

    def get_impact_weight(self, impact: EventImpact) -> float:
        """Get weight based on event impact"""
        weights = {
            EventImpact.LOW: 0.3,
            EventImpact.MEDIUM: 0.6,
            EventImpact.HIGH: 0.8,
            EventImpact.CRITICAL: 1.0
        }
        return weights[impact]

    def get_sentiment_alignment(self, sentiment_score: Optional[float], direction: int) -> float:
        """Calculate how well sentiment aligns with technical direction"""
        if sentiment_score is None:
            return 0.5
        sentiment_direction = np.sign(sentiment_score)
        return 1.0 if sentiment_direction == direction else 0.2

    def get_historical_reliability(self, event_name: str) -> float:
        """Calculate historical reliability of predictions for this event"""
        if event_name not in self.historical_impacts:
            return 0.5
        impacts = self.historical_impacts[event_name]
        if not impacts:
            return 0.5
        return min(np.mean([abs(i) for i in impacts]), 1.0)

    def calculate_position_size(self, event: NewsEvent, confidence: float, symbol: str) -> float:
        """Calculate position size based on event impact and confidence"""
        if not self.risk_manager:
            return 0.0
            
        # Adjust risk based on event impact and confidence
        base_risk = self.risk_manager.get_base_risk()
        impact_multiplier = self.get_impact_weight(event.impact)
        
        # Calculate final risk percentage
        risk_pct = base_risk * impact_multiplier * confidence
        
        # Get position size from risk manager
        return self.risk_manager.calculate_position_size(symbol, risk_pct)

    def generate_trade_setup(self, event: NewsEvent, symbol: str, current_price: float) -> Optional[NewsTradeSetup]:
        """Generate trade setup for news event"""
        # Calculate direction and confidence
        direction, confidence = self.calculate_news_direction(event)
        
        if confidence < 0.6:  # Minimum confidence threshold
            return None
            
        # Calculate pip value and volatility-based stops
        pip_value = get_pip_value(symbol)
        volatility = self.pre_news_volatility.get(symbol, pip_value * 10)
        
        # Set stop loss and take profit based on volatility and impact
        impact_multiplier = self.get_impact_weight(event.impact)
        stop_distance = volatility * (1 + impact_multiplier)
        tp_distance = stop_distance * 2  # 1:2 risk-reward minimum
        
        return NewsTradeSetup(
            event=event,
            strategy=NewsTradingStrategy.DEVIATION,
            entry_price=current_price,
            stop_loss=current_price - (direction * stop_distance),
            take_profit=current_price + (direction * tp_distance),
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now()
        )

    def update_historical_impact(self, event_name: str, actual_impact: float):
        """Update historical impact database for event"""
        if event_name not in self.historical_impacts:
            self.historical_impacts[event_name] = []
        self.historical_impacts[event_name].append(actual_impact)
        # Keep only last 10 impacts
        self.historical_impacts[event_name] = self.historical_impacts[event_name][-10:]

    def should_trade_event(self, event: NewsEvent) -> bool:
        """Determine if an event should be traded based on rules"""
        return (
            event.impact in [EventImpact.HIGH, EventImpact.CRITICAL] and
            event.forecast is not None and
            abs(event.sentiment_score or 0) > 0.3
        )