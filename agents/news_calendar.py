"""
News calendar and sentiment analysis module with Bayesian probability blending
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import aiohttp
import pandas as pd
import numpy as np
from transformers import pipeline
from dataclasses import dataclass
from enum import Enum
import json

class EventImpact(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class NewsEvent:
    timestamp: datetime
    currency: str
    event: str
    impact: EventImpact
    actual: Optional[float]
    forecast: Optional[float]
    previous: Optional[float]
    sentiment_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'currency': self.currency,
            'event': self.event,
            'impact': self.impact.name,
            'actual': self.actual,
            'forecast': self.forecast,
            'previous': self.previous,
            'sentiment_score': self.sentiment_score
        }

class NewsSentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analysis pipeline"""
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        
    def analyze_text(self, text: str) -> Tuple[float, str]:
        """
        Analyze sentiment of news text
        Returns: (sentiment_score, sentiment_label)
        """
        result = self.sentiment_analyzer(text)[0]
        score = result['score']
        if result['label'] == 'negative':
            score = -score
        return score, result['label']

class NewsCalendar:
    def __init__(
        self,
        alpha_vantage_key: str,
        base_currency: str = "USD",
        supported_pairs: List[str] = None
    ):
        """
        Initialize news calendar with API credentials and settings
        """
        self.alpha_vantage_key = alpha_vantage_key
        self.base_currency = base_currency
        self.supported_pairs = supported_pairs or [
            "EUR/USD", "GBP/USD", "USD/JPY",
            "AUD/USD", "USD/CAD", "XAUUSD"
        ]
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.event_cache: Dict[str, List[NewsEvent]] = {}
        self.impact_weights = {
            EventImpact.LOW: 0.2,
            EventImpact.MEDIUM: 0.5,
            EventImpact.HIGH: 0.8,
            EventImpact.CRITICAL: 1.0
        }

    async def fetch_economic_calendar(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[NewsEvent]:
        """
        Fetch economic calendar events from Alpha Vantage
        """
        async with aiohttp.ClientSession() as session:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "ECONOMIC_CALENDAR",
                "apikey": self.alpha_vantage_key
            }
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                events = []
                
                for event in data.get("events", []):
                    if start_date <= datetime.fromisoformat(event['date']) <= end_date:
                        impact = self._determine_impact(event)
                        events.append(NewsEvent(
                            timestamp=datetime.fromisoformat(event['date']),
                            currency=event['currency'],
                            event=event['event'],
                            impact=impact,
                            actual=float(event['actual']) if event.get('actual') else None,
                            forecast=float(event['forecast']) if event.get('forecast') else None,
                            previous=float(event['previous']) if event.get('previous') else None
                        ))
                
                return events

    async def fetch_market_news(self, pair: str) -> List[Dict]:
        """
        Fetch market news related to currency pair
        """
        currencies = self._extract_currencies(pair)
        async with aiohttp.ClientSession() as session:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": self.alpha_vantage_key,
                "topics": "forex",
                "time_from": (datetime.now() - timedelta(hours=24)).isoformat()
            }
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                news_items = []
                
                for item in data.get("feed", []):
                    if any(curr in item['title'].upper() for curr in currencies):
                        sentiment_score, _ = self.sentiment_analyzer.analyze_text(
                            item['title'] + " " + item['summary']
                        )
                        news_items.append({
                            'timestamp': item['time_published'],
                            'title': item['title'],
                            'sentiment_score': sentiment_score,
                            'source': item['source'],
                            'url': item['url']
                        })
                
                return news_items

    def _determine_impact(self, event: Dict) -> EventImpact:
        """
        Determine event impact based on event type and historical volatility
        """
        high_impact_keywords = [
            "GDP", "NFP", "CPI", "FOMC", "ECB", "BOE", "rate decision"
        ]
        medium_impact_keywords = [
            "PMI", "retail sales", "employment", "trade balance"
        ]
        
        event_name = event['event'].upper()
        
        if any(keyword.upper() in event_name for keyword in high_impact_keywords):
            return EventImpact.CRITICAL if "FOMC" in event_name else EventImpact.HIGH
        elif any(keyword.upper() in event_name for keyword in medium_impact_keywords):
            return EventImpact.MEDIUM
        return EventImpact.LOW

    def _extract_currencies(self, pair: str) -> List[str]:
        """Extract individual currencies from a pair"""
        if pair == "XAUUSD":
            return ["GOLD", "USD"]
        return pair.replace("/", "").split()

    async def calculate_news_risk(
        self,
        pair: str,
        timeframe_hours: int = 24
    ) -> Tuple[float, List[Dict]]:
        """
        Calculate overall news risk score for a currency pair
        Returns: (risk_score, relevant_events)
        """
        start_date = datetime.now()
        end_date = start_date + timedelta(hours=timeframe_hours)
        
        # Fetch economic calendar and news
        events = await self.fetch_economic_calendar(start_date, end_date)
        news = await self.fetch_market_news(pair)
        
        currencies = self._extract_currencies(pair)
        relevant_events = []
        total_risk = 0.0
        max_risk = 0.0
        
        # Process economic events
        for event in events:
            if event.currency in currencies:
                event_risk = self.impact_weights[event.impact]
                time_factor = self._calculate_time_decay(event.timestamp)
                event_risk *= time_factor
                
                relevant_events.append(event.to_dict())
                total_risk += event_risk
                max_risk += 1.0  # Maximum possible risk
        
        # Process news sentiment
        sentiment_risk = 0.0
        for news_item in news:
            sentiment_risk += abs(news_item['sentiment_score']) * 0.3  # Weight news less than economic events
            relevant_events.append(news_item)
        
        # Normalize risk score to [0, 1]
        if max_risk > 0:
            normalized_risk = (total_risk + sentiment_risk) / (max_risk + len(news))
        else:
            normalized_risk = sentiment_risk / len(news) if news else 0.0
        
        return normalized_risk, relevant_events

    def _calculate_time_decay(self, event_time: datetime) -> float:
        """
        Calculate time-based decay factor for event importance
        Closer events have higher impact
        """
        now = datetime.now()
        hours_diff = abs((event_time - now).total_seconds() / 3600)
        
        if hours_diff <= 1:
            return 1.0
        elif hours_diff <= 4:
            return 0.8
        elif hours_diff <= 12:
            return 0.5
        elif hours_diff <= 24:
            return 0.3
        return 0.1

    async def get_trade_probability(
        self,
        pair: str,
        technical_prob: float,
        direction: str
    ) -> Dict[str, float]:
        """
        Blend technical probability with fundamental factors
        Returns adjusted probability and risk metrics
        """
        news_risk, events = await self.calculate_news_risk(pair)
        
        # Calculate sentiment bias from recent events
        sentiment_scores = [
            event.get('sentiment_score', 0)
            for event in events
            if isinstance(event.get('sentiment_score'), (int, float))
        ]
        
        sentiment_bias = np.mean(sentiment_scores) if sentiment_scores else 0
        
        # Adjust probability based on news risk and sentiment
        if direction == "LONG":
            sentiment_factor = 1 + (sentiment_bias * 0.2)  # Max 20% adjustment
        else:
            sentiment_factor = 1 - (sentiment_bias * 0.2)
        
        # Reduce probability when high news risk
        risk_factor = 1 - (news_risk * 0.5)  # Max 50% reduction
        
        adjusted_prob = technical_prob * sentiment_factor * risk_factor
        
        return {
            'original_prob': technical_prob,
            'adjusted_prob': adjusted_prob,
            'news_risk': news_risk,
            'sentiment_bias': sentiment_bias,
            'num_events': len(events)
        }