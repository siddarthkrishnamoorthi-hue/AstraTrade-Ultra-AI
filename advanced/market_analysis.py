"""
Advanced market structure and order flow analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from scipy.stats import linregress
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class MarketStructure:
    trend_state: str  # 'accumulation', 'distribution', 'uptrend', 'downtrend'
    key_levels: Dict[str, float]  # Support, resistance, liquidity levels
    volume_profile: Dict[str, float]  # Price levels with significant volume
    order_flow: Dict[str, float]  # Buy/sell pressure metrics
    market_context: Dict[str, any]  # Overall market context

class MarketAnalyzer:
    def __init__(
        self,
        min_swing_length: int = 5,
        volume_threshold: float = 1.5,
        liquidity_impact: float = 0.8
    ):
        self.min_swing_length = min_swing_length
        self.volume_threshold = volume_threshold
        self.liquidity_impact = liquidity_impact
        
        # Historical data storage
        self.swing_points = {}
        self.volume_nodes = {}
        self.order_flow_history = {}
        self.market_structure_history = {}
    
    def analyze_market_structure(
        self,
        data: pd.DataFrame,
        volume_data: pd.DataFrame,
        timeframe: str
    ) -> MarketStructure:
        """Analyze market structure using institutional concepts"""
        try:
            # 1. Order Flow Analysis
            order_flow = self._analyze_order_flow(data, volume_data)
            
            # 2. Smart Money Levels
            key_levels = self._identify_smart_money_levels(data, order_flow)
            
            # 3. Volume Profile
            volume_profile = self._analyze_volume_profile(data, volume_data)
            
            # 4. Market Structure State
            trend_state = self._determine_market_structure(
                data,
                key_levels,
                order_flow
            )
            
            # 5. Market Context
            market_context = self._analyze_market_context(
                data,
                order_flow,
                trend_state,
                timeframe
            )
            
            return MarketStructure(
                trend_state=trend_state,
                key_levels=key_levels,
                volume_profile=volume_profile,
                order_flow=order_flow,
                market_context=market_context
            )
            
        except Exception as e:
            logger.error(f"Market structure analysis error: {str(e)}")
            return self._get_default_structure()
    
    def _analyze_order_flow(
        self,
        data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Analyze real order flow and liquidity dynamics"""
        try:
            # Calculate delta (buy vs sell volume)
            delta = volume_data['buy_volume'] - volume_data['sell_volume']
            cumulative_delta = delta.cumsum()
            
            # Identify large orders (smart money footprints)
            large_orders = volume_data[
                volume_data['volume'] > volume_data['volume'].mean() * 2
            ]
            
            # Calculate buying/selling pressure
            buy_pressure = (
                volume_data['buy_volume'].rolling(20).sum() /
                volume_data['volume'].rolling(20).sum()
            )
            
            # Detect institutional order patterns
            block_trades = self._detect_block_trades(volume_data)
            
            # Analyze price rejection levels
            rejection_levels = self._find_price_rejection(data, volume_data)
            
            return {
                'delta': float(delta.iloc[-1]),
                'cumulative_delta': float(cumulative_delta.iloc[-1]),
                'buy_pressure': float(buy_pressure.iloc[-1]),
                'large_orders': block_trades,
                'rejection_levels': rejection_levels,
                'institutional_activity': self._score_institutional_activity(
                    data, volume_data, block_trades
                )
            }
            
        except Exception as e:
            logger.error(f"Order flow analysis error: {str(e)}")
            return {}
    
    def _identify_smart_money_levels(
        self,
        data: pd.DataFrame,
        order_flow: Dict
    ) -> Dict[str, float]:
        """Identify institutional price levels"""
        try:
            # Find liquidity pools
            liquidity_levels = self._find_liquidity_pools(data)
            
            # Identify institutional swings
            swings = self._find_institutional_swings(data)
            
            # Find order blocks
            order_blocks = self._identify_order_blocks(data, order_flow)
            
            # Detect stop runs
            stop_runs = self._detect_stop_runs(data, liquidity_levels)
            
            # Find fair value gaps
            fvg = self._find_fair_value_gaps(data)
            
            # Combine and rank levels
            all_levels = {
                **liquidity_levels,
                **order_blocks,
                **stop_runs,
                **fvg
            }
            
            # Score and filter levels
            return self._score_and_filter_levels(all_levels, data, order_flow)
            
        except Exception as e:
            logger.error(f"Smart money levels analysis error: {str(e)}")
            return {}
    
    def _find_liquidity_pools(self, data: pd.DataFrame) -> Dict[str, float]:
        """Find areas of resting liquidity"""
        pools = {}
        try:
            # Look for swing highs/lows with multiple tests
            highs = data[data['high'] == data['high'].rolling(10).max()]
            lows = data[data['low'] == data['low'].rolling(10).min()]
            
            # Filter for areas with multiple tests
            for price in highs['high'].unique():
                tests = len(highs[highs['high'] == price])
                if tests >= 2:
                    pools[f'high_{price}'] = {
                        'price': price,
                        'strength': tests * 0.2,
                        'type': 'resistance'
                    }
            
            for price in lows['low'].unique():
                tests = len(lows[lows['low'] == price])
                if tests >= 2:
                    pools[f'low_{price}'] = {
                        'price': price,
                        'strength': tests * 0.2,
                        'type': 'support'
                    }
            
            return pools
            
        except Exception as e:
            logger.error(f"Liquidity pools analysis error: {str(e)}")
            return {}
    
    def _find_institutional_swings(self, data: pd.DataFrame) -> Dict[str, float]:
        """Identify institutional swing points"""
        swings = {}
        try:
            # Calculate price movement and volume profile
            data['price_move'] = data['high'] - data['low']
            data['avg_move'] = data['price_move'].rolling(20).mean()
            
            # Find strong moves with follow-through
            strong_moves = data[data['price_move'] > data['avg_move'] * 1.5]
            
            for idx in strong_moves.index:
                if idx + 5 >= len(data):
                    continue
                    
                # Check follow-through
                follow_through = data.loc[idx:idx+5]
                if len(follow_through[follow_through['price_move'] > data.loc[idx, 'avg_move']]) >= 3:
                    swings[f'swing_{idx}'] = {
                        'price': data.loc[idx, 'close'],
                        'strength': float(data.loc[idx, 'price_move'] / data.loc[idx, 'avg_move']),
                        'type': 'swing'
                    }
            
            return swings
            
        except Exception as e:
            logger.error(f"Institutional swings analysis error: {str(e)}")
            return {}
    
    def _identify_order_blocks(
        self,
        data: pd.DataFrame,
        order_flow: Dict
    ) -> Dict[str, float]:
        """Identify institutional order blocks"""
        blocks = {}
        try:
            # Look for strong moves followed by consolidation
            data['range'] = data['high'] - data['low']
            data['avg_range'] = data['range'].rolling(20).mean()
            
            for i in range(len(data) - 20):
                window = data.iloc[i:i+20]
                
                # Check for expansion followed by contraction
                if (window['range'].iloc[0] > window['avg_range'].iloc[0] * 1.5 and
                    window['range'].iloc[1:].mean() < window['avg_range'].iloc[0]):
                    
                    blocks[f'block_{i}'] = {
                        'price': window['close'].iloc[0],
                        'strength': float(window['range'].iloc[0] / window['avg_range'].iloc[0]),
                        'type': 'order_block'
                    }
            
            return blocks
            
        except Exception as e:
            logger.error(f"Order blocks analysis error: {str(e)}")
            return {}
    
    def _detect_stop_runs(
        self,
        data: pd.DataFrame,
        liquidity_levels: Dict
    ) -> Dict[str, float]:
        """Detect stop hunting patterns"""
        stop_runs = {}
        try:
            for level_id, level in liquidity_levels.items():
                price = level['price']
                
                # Look for price pushing through level then reversing
                breaks = data[
                    (data['high'] > price) & 
                    (data['close'] < price)
                ]
                
                if len(breaks) > 0:
                    stop_runs[f'stop_run_{level_id}'] = {
                        'price': price,
                        'strength': level['strength'] * 1.2,  # Increase importance
                        'type': 'stop_run'
                    }
            
            return stop_runs
            
        except Exception as e:
            logger.error(f"Stop runs detection error: {str(e)}")
            return {}
    
    def _find_fair_value_gaps(self, data: pd.DataFrame) -> Dict[str, float]:
        """Identify fair value gaps"""
        gaps = {}
        try:
            for i in range(1, len(data) - 1):
                # Look for gaps in fair value
                if (data['low'].iloc[i] > data['high'].iloc[i-1] and
                    data['high'].iloc[i+1] < data['low'].iloc[i]):
                    
                    gap_size = data['low'].iloc[i] - data['high'].iloc[i-1]
                    gaps[f'fvg_{i}'] = {
                        'price': data['low'].iloc[i],
                        'strength': float(gap_size / data['atr'].iloc[i]),
                        'type': 'fvg'
                    }
            
            return gaps
            
        except Exception as e:
            logger.error(f"Fair value gaps analysis error: {str(e)}")
            return {}
    
    def _score_and_filter_levels(
        self,
        levels: Dict,
        data: pd.DataFrame,
        order_flow: Dict
    ) -> Dict[str, float]:
        """Score and filter significant levels"""
        scored_levels = {}
        try:
            current_price = data['close'].iloc[-1]
            
            for level_id, level in levels.items():
                score = 0
                
                # Base strength
                score += level['strength']
                
                # Recent tests
                price = level['price']
                recent_tests = len(
                    data[
                        (data['high'] >= price * 0.9999) &
                        (data['low'] <= price * 1.0001)
                    ].index
                )
                score += recent_tests * 0.1
                
                # Order flow confirmation
                if abs(order_flow['delta']) > 0:
                    score *= (1 + order_flow['institutional_activity'])
                
                # Only keep strong levels
                if score >= 1.5:
                    scored_levels[level_id] = {
                        'price': price,
                        'score': float(score),
                        'type': level['type']
                    }
            
            return scored_levels
            
        except Exception as e:
            logger.error(f"Level scoring error: {str(e)}")
            return {}
    
    def _score_institutional_activity(
        self,
        data: pd.DataFrame,
        volume_data: pd.DataFrame,
        block_trades: Dict
    ) -> float:
        """Score the level of institutional activity"""
        try:
            score = 0.0
            
            # Large orders impact
            score += len(block_trades) * 0.2
            
            # Volume consistency
            vol_std = volume_data['volume'].std()
            vol_mean = volume_data['volume'].mean()
            score += 1 - (vol_std / vol_mean)  # Lower variation = higher score
            
            # Price movement quality
            price_efficiency = abs(
                data['close'].iloc[-1] - data['close'].iloc[0]
            ) / data['high'].max() - data['low'].min()
            score += price_efficiency
            
            return min(1.0, score / 3)  # Normalize to [0, 1]
            
        except Exception as e:
            logger.error(f"Institutional activity scoring error: {str(e)}")
            return 0.0
    
    def _get_default_structure(self) -> MarketStructure:
        """Return default market structure when analysis fails"""
        return MarketStructure(
            trend_state='unknown',
            key_levels={},
            volume_profile={},
            order_flow={},
            market_context={}
        )