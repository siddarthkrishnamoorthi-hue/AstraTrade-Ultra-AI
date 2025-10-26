"""
Advanced technical analysis module implementing SMC, ICT, and orderflow concepts
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ZoneType(Enum):
    SUPPLY = 'supply'
    DEMAND = 'demand'
    EQUILIBRIUM = 'equilibrium'

@dataclass
class Zone:
    type: ZoneType
    high: float
    low: float
    strength: float
    timestamp: pd.Timestamp
    timeframe: str
    touched: int = 0

class TechnicalDetector:
    def __init__(self):
        self.zones: List[Zone] = []
        self.fvg_threshold = 0.001  # Minimum gap size as percentage
        self.cvd_window = 1000  # Ticks for CVD calculation
        
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Fair Value Gaps (FVG) using 3-candle pattern
        Returns list of FVGs with coordinates and strength
        """
        fvgs = []
        for i in range(1, len(df)-1):
            # Bullish FVG
            if df['Low'].iloc[i+1] > df['High'].iloc[i-1]:
                gap_size = df['Low'].iloc[i+1] - df['High'].iloc[i-1]
                if gap_size > df['Close'].iloc[i] * self.fvg_threshold:
                    fvgs.append({
                        'type': 'bullish',
                        'top': df['Low'].iloc[i+1],
                        'bottom': df['High'].iloc[i-1],
                        'strength': gap_size / df['Close'].iloc[i],
                        'timestamp': df.index[i]
                    })
            
            # Bearish FVG
            if df['High'].iloc[i+1] < df['Low'].iloc[i-1]:
                gap_size = df['Low'].iloc[i-1] - df['High'].iloc[i+1]
                if gap_size > df['Close'].iloc[i] * self.fvg_threshold:
                    fvgs.append({
                        'type': 'bearish',
                        'top': df['Low'].iloc[i-1],
                        'bottom': df['High'].iloc[i+1],
                        'strength': gap_size / df['Close'].iloc[i],
                        'timestamp': df.index[i]
                    })
        return fvgs

    def calculate_cvd(self, ticks: pd.DataFrame) -> float:
        """
        Calculate Cumulative Volume Delta from tick data
        """
        # Assuming ticks DataFrame has columns: price, volume, direction (1 for buy, -1 for sell)
        return (ticks['volume'] * ticks['direction']).sum()

    def detect_orderblocks(self, df: pd.DataFrame, swing_threshold: float = 0.003) -> List[Dict]:
        """
        Detect SMC Order Blocks using swing structure analysis
        """
        orderblocks = []
        highs = df['High'].rolling(3, center=True).max()
        lows = df['Low'].rolling(3, center=True).min()
        
        for i in range(2, len(df)-2):
            # Bullish OB
            if (df['Low'].iloc[i-1] - df['Low'].iloc[i-2])/df['Low'].iloc[i-2] > swing_threshold:
                if df['High'].iloc[i+1] > df['High'].iloc[i]:
                    orderblocks.append({
                        'type': 'bullish',
                        'high': df['High'].iloc[i],
                        'low': df['Low'].iloc[i],
                        'strength': (df['High'].iloc[i+1] - df['High'].iloc[i])/df['High'].iloc[i],
                        'timestamp': df.index[i]
                    })
            
            # Bearish OB
            if (df['High'].iloc[i-1] - df['High'].iloc[i-2])/df['High'].iloc[i-2] < -swing_threshold:
                if df['Low'].iloc[i+1] < df['Low'].iloc[i]:
                    orderblocks.append({
                        'type': 'bearish',
                        'high': df['High'].iloc[i],
                        'low': df['Low'].iloc[i],
                        'strength': (df['Low'].iloc[i] - df['Low'].iloc[i+1])/df['Low'].iloc[i],
                        'timestamp': df.index[i]
                    })
        return orderblocks

    def identify_liquidity_levels(self, df: pd.DataFrame, window: int = 20) -> List[Dict]:
        """
        Identify potential liquidity levels using swing highs/lows
        """
        levels = []
        
        # Rolling max/min for swing points
        highs = df['High'].rolling(window, center=True).max()
        lows = df['Low'].rolling(window, center=True).min()
        
        for i in range(window, len(df)-window):
            # Swing high liquidity
            if df['High'].iloc[i] == highs.iloc[i]:
                levels.append({
                    'type': 'resistance',
                    'price': df['High'].iloc[i],
                    'strength': self._calculate_level_strength(df, i, 'high'),
                    'timestamp': df.index[i]
                })
            
            # Swing low liquidity
            if df['Low'].iloc[i] == lows.iloc[i]:
                levels.append({
                    'type': 'support',
                    'price': df['Low'].iloc[i],
                    'strength': self._calculate_level_strength(df, i, 'low'),
                    'timestamp': df.index[i]
                })
        
        return levels

    def _calculate_level_strength(self, df: pd.DataFrame, index: int, level_type: str, window: int = 10) -> float:
        """
        Calculate the strength of a liquidity level based on historical tests
        """
        price = df['High'].iloc[index] if level_type == 'high' else df['Low'].iloc[index]
        subsequent_tests = 0
        volume_intensity = 0
        
        for i in range(index + 1, min(index + window, len(df))):
            if level_type == 'high':
                if abs(df['High'].iloc[i] - price) / price < 0.0005:  # 0.05% threshold
                    subsequent_tests += 1
                    volume_intensity += df['Volume'].iloc[i] if 'Volume' in df else 1
            else:
                if abs(df['Low'].iloc[i] - price) / price < 0.0005:
                    subsequent_tests += 1
                    volume_intensity += df['Volume'].iloc[i] if 'Volume' in df else 1
        
        return (subsequent_tests * volume_intensity) / window

    def detect_breaker_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect breaker blocks - OB that gets broken and flips polarity
        """
        breakers = []
        orderblocks = self.detect_orderblocks(df)
        
        for ob in orderblocks:
            # Check if price action after the OB breaks it
            ob_idx = df.index.get_loc(ob['timestamp'])
            subsequent_price = df.iloc[ob_idx+1:ob_idx+10]
            
            if ob['type'] == 'bullish':
                if any(subsequent_price['Low'] < ob['low']):
                    breakers.append({
                        'type': 'bearish_breaker',
                        'high': ob['high'],
                        'low': ob['low'],
                        'original_timestamp': ob['timestamp'],
                        'break_timestamp': subsequent_price[subsequent_price['Low'] < ob['low']].index[0]
                    })
            else:
                if any(subsequent_price['High'] > ob['high']):
                    breakers.append({
                        'type': 'bullish_breaker',
                        'high': ob['high'],
                        'low': ob['low'],
                        'original_timestamp': ob['timestamp'],
                        'break_timestamp': subsequent_price[subsequent_price['High'] > ob['high']].index[0]
                    })
        
        return breakers

    def analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive market structure analysis combining multiple components
        """
        analysis = {
            'fvgs': self.detect_fair_value_gaps(df),
            'orderblocks': self.detect_orderblocks(df),
            'liquidity_levels': self.identify_liquidity_levels(df),
            'breakers': self.detect_breaker_blocks(df)
        }
        
        # Add confluence scoring
        for component in analysis.values():
            for item in component:
                item['confluence_score'] = self._calculate_confluence(df, item)
        
        return analysis

    def _calculate_confluence(self, df: pd.DataFrame, feature: Dict) -> float:
        """
        Calculate confluence score based on multiple technical factors
        """
        score = 0.0
        price_level = feature.get('price', feature.get('top', feature.get('high')))
        
        # Check proximity to key levels
        liquidity_levels = self.identify_liquidity_levels(df)
        for level in liquidity_levels:
            if abs(level['price'] - price_level) / price_level < 0.001:
                score += 0.2
        
        # Check volume confirmation if available
        if 'Volume' in df:
            recent_vol_avg = df['Volume'].tail(20).mean()
            if feature.get('timestamp'):
                idx = df.index.get_loc(feature['timestamp'])
                if idx < len(df):
                    if df['Volume'].iloc[idx] > recent_vol_avg * 1.5:
                        score += 0.3
        
        # Add time-based decay
        if feature.get('timestamp'):
            age = (df.index[-1] - feature['timestamp']).total_seconds() / 3600  # Hours
            time_decay = max(0, 1 - (age / 24))  # Decay over 24 hours
            score += 0.2 * time_decay
        
        return min(1.0, score)