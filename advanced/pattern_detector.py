"""
Advanced institutional pattern recognition and smart money movement tracker
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.stats import norm, linregress
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class InstitutionalPattern:
    type: str
    timeframe: str
    direction: str
    strength: float
    entry_points: List[float]
    stop_levels: List[float]
    targets: List[float]
    context: Dict[str, any]
    volume_profile: Dict[str, float]

class InstitutionalPatternDetector:
    def __init__(
        self,
        min_pattern_strength: float = 0.85,
        volume_impact_threshold: float = 2.0,
        smart_money_threshold: float = 0.8
    ):
        self.min_pattern_strength = min_pattern_strength
        self.volume_impact_threshold = volume_impact_threshold
        self.smart_money_threshold = smart_money_threshold
        
        # Pattern tracking
        self.detected_patterns = defaultdict(list)
        self.pattern_performance = defaultdict(dict)
        
        # Smart money tracking
        self.smart_money_flows = defaultdict(list)
        self.institutional_levels = defaultdict(dict)
        
        # Market microstructure
        self.micro_patterns = defaultdict(list)
        self.order_flow_history = defaultdict(list)
        
        # Performance metrics
        self.pattern_stats = self._initialize_pattern_stats()
    
    def _initialize_pattern_stats(self) -> Dict:
        """Initialize pattern statistics tracking"""
        return {
            'sweep_continuation': {'success': 0, 'total': 0},
            'liquidity_void': {'success': 0, 'total': 0},
            'institutional_block': {'success': 0, 'total': 0},
            'smart_money_reversal': {'success': 0, 'total': 0},
            'orderflow_imbalance': {'success': 0, 'total': 0},
            'volume_void_fill': {'success': 0, 'total': 0},
            'inefficient_price': {'success': 0, 'total': 0},
            'multi_timeframe_break': {'success': 0, 'total': 0}
        }
    
    async def detect_patterns(
        self,
        data: Dict[str, pd.DataFrame],
        volume_data: pd.DataFrame,
        market_structure: 'MarketStructure'
    ) -> List[InstitutionalPattern]:
        """Detect high-probability institutional patterns"""
        try:
            patterns = []
            
            # 1. Detect Smart Money Movement Patterns
            smart_money = await self._detect_smart_money_patterns(
                data['H4'],
                volume_data
            )
            patterns.extend(smart_money)
            
            # 2. Find Liquidity Engineering Patterns
            liquidity = self._find_liquidity_patterns(
                data,
                market_structure
            )
            patterns.extend(liquidity)
            
            # 3. Detect Order Flow Imbalances
            imbalances = self._detect_orderflow_imbalances(
                data['H1'],
                volume_data
            )
            patterns.extend(imbalances)
            
            # 4. Find Multi-timeframe Break Points
            breaks = self._find_break_points(data, market_structure)
            patterns.extend(breaks)
            
            # 5. Detect Premium/Discount Zones
            zones = self._detect_premium_zones(
                data,
                market_structure,
                volume_data
            )
            patterns.extend(zones)
            
            # Filter and rank patterns
            valid_patterns = self._validate_patterns(patterns, market_structure)
            ranked_patterns = self._rank_patterns(valid_patterns)
            
            return ranked_patterns
            
        except Exception as e:
            logger.error(f"Pattern detection error: {str(e)}")
            return []
    
    async def _detect_smart_money_patterns(
        self,
        data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> List[InstitutionalPattern]:
        """Detect smart money movement patterns"""
        patterns = []
        try:
            # 1. Detect Institutional Sweeps
            sweeps = self._find_institutional_sweeps(data, volume_data)
            for sweep in sweeps:
                if sweep['strength'] >= self.smart_money_threshold:
                    patterns.append(
                        InstitutionalPattern(
                            type='sweep_continuation',
                            timeframe='H4',
                            direction=sweep['direction'],
                            strength=sweep['strength'],
                            entry_points=sweep['entries'],
                            stop_levels=sweep['stops'],
                            targets=sweep['targets'],
                            context={'sweep_type': sweep['sweep_type']},
                            volume_profile=sweep['volume_profile']
                        )
                    )
            
            # 2. Find Liquidity Voids
            voids = self._detect_liquidity_voids(data, volume_data)
            for void in voids:
                patterns.append(
                    InstitutionalPattern(
                        type='liquidity_void',
                        timeframe='H4',
                        direction=void['direction'],
                        strength=void['strength'],
                        entry_points=void['entries'],
                        stop_levels=void['stops'],
                        targets=void['targets'],
                        context={'void_size': void['size']},
                        volume_profile=void['volume_profile']
                    )
                )
            
            # 3. Detect Block Formations
            blocks = self._detect_institutional_blocks(data, volume_data)
            patterns.extend([
                InstitutionalPattern(
                    type='institutional_block',
                    timeframe='H4',
                    direction=block['direction'],
                    strength=block['strength'],
                    entry_points=block['entries'],
                    stop_levels=block['stops'],
                    targets=block['targets'],
                    context={'block_type': block['block_type']},
                    volume_profile=block['volume_profile']
                ) for block in blocks
            ])
            
            return patterns
            
        except Exception as e:
            logger.error(f"Smart money pattern detection error: {str(e)}")
            return []
    
    def _find_institutional_sweeps(
        self,
        data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> List[Dict]:
        """Find institutional sweep patterns"""
        sweeps = []
        try:
            # Calculate volume deltas
            volume_data['delta'] = volume_data['buy_volume'] - volume_data['sell_volume']
            
            # Find high volume price levels
            for i in range(1, len(data)-1):
                # Check for price sweep with high volume
                if (
                    data['high'].iloc[i] > data['high'].iloc[i-1] and
                    data['close'].iloc[i] < data['low'].iloc[i-1] and
                    volume_data['volume'].iloc[i] > volume_data['volume'].rolling(20).mean().iloc[i] * 2
                ):
                    # Detect sweep type
                    sweep_type = 'buy_sweep' if volume_data['delta'].iloc[i] > 0 else 'sell_sweep'
                    
                    # Calculate sweep metrics
                    sweep_size = abs(data['high'].iloc[i] - data['low'].iloc[i])
                    volume_impact = volume_data['volume'].iloc[i] / volume_data['volume'].rolling(20).mean().iloc[i]
                    
                    sweeps.append({
                        'direction': 'long' if sweep_type == 'buy_sweep' else 'short',
                        'strength': min(volume_impact / 3, 1.0),
                        'entries': [data['close'].iloc[i]],
                        'stops': [data['low'].iloc[i] * 0.999],
                        'targets': [
                            data['close'].iloc[i] * 1.005,
                            data['close'].iloc[i] * 1.01
                        ],
                        'sweep_type': sweep_type,
                        'volume_profile': {
                            'sweep_volume': float(volume_data['volume'].iloc[i]),
                            'avg_volume': float(volume_data['volume'].rolling(20).mean().iloc[i])
                        }
                    })
            
            return sweeps
            
        except Exception as e:
            logger.error(f"Institutional sweep detection error: {str(e)}")
            return []
    
    def _detect_liquidity_voids(
        self,
        data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> List[Dict]:
        """Detect liquidity void areas"""
        voids = []
        try:
            # Calculate volume profile
            price_levels = np.linspace(
                data['low'].min(),
                data['high'].max(),
                100
            )
            volume_profile = defaultdict(float)
            
            # Build volume profile
            for price, volume in zip(data['close'], volume_data['volume']):
                idx = np.digitize(price, price_levels) - 1
                if 0 <= idx < len(price_levels):
                    volume_profile[price_levels[idx]] += volume
            
            # Find volume voids
            avg_volume = np.mean(list(volume_profile.values()))
            for i in range(1, len(price_levels)-1):
                current_vol = volume_profile[price_levels[i]]
                if current_vol < avg_volume * 0.2:  # 80% less volume
                    void_size = price_levels[i+1] - price_levels[i-1]
                    
                    voids.append({
                        'direction': 'long' if data['close'].iloc[-1] < price_levels[i] else 'short',
                        'strength': 1 - (current_vol / avg_volume),
                        'entries': [price_levels[i]],
                        'stops': [price_levels[i-1]],
                        'targets': [price_levels[i+1]],
                        'size': void_size,
                        'volume_profile': {
                            'void_volume': float(current_vol),
                            'avg_volume': float(avg_volume)
                        }
                    })
            
            return voids
            
        except Exception as e:
            logger.error(f"Liquidity void detection error: {str(e)}")
            return []
    
    def _detect_institutional_blocks(
        self,
        data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> List[Dict]:
        """Detect institutional block formations"""
        blocks = []
        try:
            # Calculate block metrics
            data['range'] = data['high'] - data['low']
            data['avg_range'] = data['range'].rolling(20).mean()
            volume_data['vol_delta'] = volume_data['buy_volume'] - volume_data['sell_volume']
            
            for i in range(20, len(data)):
                # Look for high volume candles with small ranges
                if (
                    data['range'].iloc[i] < data['avg_range'].iloc[i] * 0.5 and
                    volume_data['volume'].iloc[i] > volume_data['volume'].rolling(20).mean().iloc[i] * 1.5
                ):
                    # Determine block type
                    block_type = 'accumulation' if volume_data['vol_delta'].iloc[i] > 0 else 'distribution'
                    
                    blocks.append({
                        'direction': 'long' if block_type == 'accumulation' else 'short',
                        'strength': min(
                            volume_data['volume'].iloc[i] / volume_data['volume'].rolling(20).mean().iloc[i],
                            1.0
                        ),
                        'entries': [data['close'].iloc[i]],
                        'stops': [
                            data['low'].iloc[i] * 0.999 if block_type == 'accumulation'
                            else data['high'].iloc[i] * 1.001
                        ],
                        'targets': [
                            data['close'].iloc[i] * 1.01 if block_type == 'accumulation'
                            else data['close'].iloc[i] * 0.99
                        ],
                        'block_type': block_type,
                        'volume_profile': {
                            'block_volume': float(volume_data['volume'].iloc[i]),
                            'avg_volume': float(volume_data['volume'].rolling(20).mean().iloc[i]),
                            'delta': float(volume_data['vol_delta'].iloc[i])
                        }
                    })
            
            return blocks
            
        except Exception as e:
            logger.error(f"Institutional block detection error: {str(e)}")
            return []
    
    def _find_liquidity_patterns(
        self,
        data: Dict[str, pd.DataFrame],
        market_structure: 'MarketStructure'
    ) -> List[InstitutionalPattern]:
        """Find liquidity engineering patterns"""
        patterns = []
        try:
            # 1. Find stop hunt setups
            stop_hunts = self._detect_stop_hunts(
                data['H1'],
                market_structure
            )
            patterns.extend(stop_hunts)
            
            # 2. Find liquidity gaps
            gaps = self._find_liquidity_gaps(data)
            patterns.extend(gaps)
            
            # 3. Detect smart money traps
            traps = self._detect_smart_money_traps(
                data,
                market_structure
            )
            patterns.extend(traps)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Liquidity pattern detection error: {str(e)}")
            return []
    
    def _detect_orderflow_imbalances(
        self,
        data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> List[InstitutionalPattern]:
        """Detect order flow imbalances"""
        patterns = []
        try:
            # Calculate order flow metrics
            volume_data['imbalance'] = (
                volume_data['buy_volume'] - volume_data['sell_volume']
            ) / volume_data['volume']
            
            # Find significant imbalances
            for i in range(1, len(data)-1):
                if abs(volume_data['imbalance'].iloc[i]) > 0.7:  # 70% imbalance
                    direction = 'long' if volume_data['imbalance'].iloc[i] > 0 else 'short'
                    
                    patterns.append(
                        InstitutionalPattern(
                            type='orderflow_imbalance',
                            timeframe='H1',
                            direction=direction,
                            strength=abs(volume_data['imbalance'].iloc[i]),
                            entry_points=[data['close'].iloc[i]],
                            stop_levels=[
                                data['low'].iloc[i] * 0.999 if direction == 'long'
                                else data['high'].iloc[i] * 1.001
                            ],
                            targets=[
                                data['close'].iloc[i] * 1.005 if direction == 'long'
                                else data['close'].iloc[i] * 0.995
                            ],
                            context={
                                'imbalance_size': float(volume_data['imbalance'].iloc[i]),
                                'volume_surge': float(
                                    volume_data['volume'].iloc[i] /
                                    volume_data['volume'].rolling(20).mean().iloc[i]
                                )
                            },
                            volume_profile={
                                'buy_volume': float(volume_data['buy_volume'].iloc[i]),
                                'sell_volume': float(volume_data['sell_volume'].iloc[i])
                            }
                        )
                    )
            
            return patterns
            
        except Exception as e:
            logger.error(f"Order flow imbalance detection error: {str(e)}")
            return []
    
    def _validate_patterns(
        self,
        patterns: List[InstitutionalPattern],
        market_structure: 'MarketStructure'
    ) -> List[InstitutionalPattern]:
        """Validate and filter patterns"""
        valid_patterns = []
        try:
            for pattern in patterns:
                # Check pattern strength
                if pattern.strength < self.min_pattern_strength:
                    continue
                
                # Verify pattern alignment with market structure
                if not self._verify_pattern_alignment(
                    pattern,
                    market_structure
                ):
                    continue
                
                # Check historical performance
                if not self._check_pattern_performance(pattern.type):
                    continue
                
                # Verify volume confirmation
                if not self._verify_volume_confirmation(
                    pattern.volume_profile
                ):
                    continue
                
                valid_patterns.append(pattern)
            
            return valid_patterns
            
        except Exception as e:
            logger.error(f"Pattern validation error: {str(e)}")
            return []
    
    def _rank_patterns(
        self,
        patterns: List[InstitutionalPattern]
    ) -> List[InstitutionalPattern]:
        """Rank patterns by quality and probability"""
        try:
            if not patterns:
                return []
            
            # Calculate pattern scores
            scored_patterns = [
                (
                    pattern,
                    self._calculate_pattern_score(pattern)
                )
                for pattern in patterns
            ]
            
            # Sort by score
            scored_patterns.sort(key=lambda x: x[1], reverse=True)
            
            # Return patterns only
            return [p[0] for p in scored_patterns]
            
        except Exception as e:
            logger.error(f"Pattern ranking error: {str(e)}")
            return patterns  # Return original patterns if ranking fails
    
    def _calculate_pattern_score(
        self,
        pattern: InstitutionalPattern
    ) -> float:
        """Calculate comprehensive pattern quality score"""
        try:
            score = 0.0
            
            # Pattern strength (40%)
            score += pattern.strength * 0.4
            
            # Historical performance (30%)
            if pattern.type in self.pattern_stats:
                stats = self.pattern_stats[pattern.type]
                if stats['total'] > 0:
                    win_rate = stats['success'] / stats['total']
                    score += win_rate * 0.3
            
            # Volume confirmation (20%)
            volume_score = self._calculate_volume_score(
                pattern.volume_profile
            )
            score += volume_score * 0.2
            
            # Pattern type priority (10%)
            type_priority = {
                'sweep_continuation': 1.0,
                'institutional_block': 0.9,
                'liquidity_void': 0.8,
                'orderflow_imbalance': 0.7
            }
            score += type_priority.get(pattern.type, 0.5) * 0.1
            
            return score
            
        except Exception as e:
            logger.error(f"Pattern scoring error: {str(e)}")
            return 0.0
    
    def update_pattern_performance(
        self,
        pattern_type: str,
        success: bool
    ):
        """Update pattern performance statistics"""
        try:
            if pattern_type in self.pattern_stats:
                self.pattern_stats[pattern_type]['total'] += 1
                if success:
                    self.pattern_stats[pattern_type]['success'] += 1
        except Exception as e:
            logger.error(f"Pattern performance update error: {str(e)}")
    
    @staticmethod
    def _calculate_volume_score(volume_profile: Dict[str, float]) -> float:
        """Calculate volume-based confirmation score"""
        try:
            if 'avg_volume' in volume_profile and volume_profile['avg_volume'] > 0:
                relative_volume = volume_profile.get(
                    'block_volume',
                    volume_profile.get('sweep_volume', 0)
                ) / volume_profile['avg_volume']
                return min(relative_volume / 2, 1.0)
            return 0.5
        except Exception as e:
            logger.error(f"Volume score calculation error: {str(e)}")
            return 0.5