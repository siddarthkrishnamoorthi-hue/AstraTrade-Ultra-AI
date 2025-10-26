"""
Advanced institutional order flow and market microstructure analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class MicroStructure:
    absorption_levels: Dict[float, float]  # price -> volume
    institutional_footprint: Dict[str, float]
    delta_divergence: Dict[str, float]
    liquidity_profile: Dict[float, float]
    order_flow_imbalance: float

@dataclass
class InstitutionalFootprint:
    type: str  # 'absorption', 'distribution', 'manipulation', 'accumulation'
    price_level: float
    volume_profile: Dict[str, float]
    time_decay: float
    strength: float

class MicroStructureAnalyzer:
    def __init__(
        self,
        tick_window: int = 1000,
        volume_threshold: float = 2.0,
        footprint_decay: float = 0.94
    ):
        self.tick_window = tick_window
        self.volume_threshold = volume_threshold
        self.footprint_decay = footprint_decay
        
        # State tracking
        self.volume_profile = defaultdict(float)
        self.delta_profile = defaultdict(float)
        self.footprint_memory = []
        
        # Pattern tracking
        self.institutional_patterns = {
            'absorption': self._detect_absorption,
            'spring_test': self._detect_spring_test,
            'liquidity_sweep': self._detect_liquidity_sweep,
            'efficiency_test': self._detect_efficiency_test,
            'reversal_block': self._detect_reversal_block,
            'wyckoff_spring': self._detect_wyckoff_spring,
            'composite_man': self._detect_composite_man,
            'range_expansion': self._detect_range_expansion
        }
    
    async def analyze_microstructure(
        self,
        tick_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        depth_data: Optional[pd.DataFrame] = None
    ) -> MicroStructure:
        """Analyze market microstructure in real-time"""
        try:
            # 1. Analyze Order Flow
            absorption = await self._analyze_absorption(tick_data, volume_data)
            
            # 2. Detect Institutional Footprints
            footprints = self._detect_footprints(tick_data, volume_data)
            
            # 3. Calculate Delta Divergence
            delta_div = self._calculate_delta_divergence(tick_data, volume_data)
            
            # 4. Analyze Market Depth (if available)
            liquidity = self._analyze_liquidity_profile(
                depth_data if depth_data is not None else volume_data
            )
            
            # 5. Calculate Order Flow Imbalance
            imbalance = self._calculate_flow_imbalance(volume_data)
            
            return MicroStructure(
                absorption_levels=absorption,
                institutional_footprint=footprints,
                delta_divergence=delta_div,
                liquidity_profile=liquidity,
                order_flow_imbalance=imbalance
            )
            
        except Exception as e:
            logger.error(f"Microstructure analysis error: {str(e)}")
            return self._get_default_structure()
    
    async def detect_advanced_patterns(
        self,
        microstructure: MicroStructure,
        market_data: pd.DataFrame,
        timeframe: str
    ) -> List[Dict]:
        """Detect advanced institutional patterns"""
        patterns = []
        try:
            for pattern_name, detector in self.institutional_patterns.items():
                pattern = await detector(
                    microstructure,
                    market_data,
                    timeframe
                )
                if pattern:
                    patterns.append(pattern)
            
            # Filter and rank patterns
            valid_patterns = self._validate_patterns(patterns)
            ranked_patterns = self._rank_by_probability(valid_patterns)
            
            return ranked_patterns
            
        except Exception as e:
            logger.error(f"Pattern detection error: {str(e)}")
            return []
    
    async def _detect_absorption(
        self,
        microstructure: MicroStructure,
        market_data: pd.DataFrame,
        timeframe: str
    ) -> Optional[Dict]:
        """Detect institutional absorption pattern"""
        try:
            # Find significant volume absorption
            absorption_levels = sorted(
                [
                    (price, vol)
                    for price, vol in microstructure.absorption_levels.items()
                    if vol > self.volume_threshold
                ],
                key=lambda x: x[1],
                reverse=True
            )
            
            if not absorption_levels:
                return None
            
            # Analyze price action around absorption
            for price, volume in absorption_levels:
                price_action = market_data[
                    (market_data['low'] <= price) &
                    (market_data['high'] >= price)
                ].tail(20)
                
                if len(price_action) < 5:
                    continue
                
                # Check for absorption characteristics
                if (
                    price_action['close'].std() < price_action['close'].mean() * 0.001 and
                    volume > price_action['volume'].mean() * 2
                ):
                    return {
                        'type': 'absorption',
                        'subtype': 'institutional_accumulation',
                        'price_level': float(price),
                        'strength': float(volume / price_action['volume'].mean()),
                        'timeframe': timeframe,
                        'volume_profile': {
                            'absorption_volume': float(volume),
                            'avg_volume': float(price_action['volume'].mean()),
                            'volume_ratio': float(volume / price_action['volume'].mean())
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Absorption detection error: {str(e)}")
            return None
    
    async def _detect_spring_test(
        self,
        microstructure: MicroStructure,
        market_data: pd.DataFrame,
        timeframe: str
    ) -> Optional[Dict]:
        """Detect institutional spring test pattern"""
        try:
            # Find recent low with high volume
            lows = market_data[
                market_data['low'] == market_data['low'].rolling(20).min()
            ].copy()
            
            if len(lows) < 2:
                return None
            
            for idx in lows.index[-2:]:
                # Check volume characteristics
                volume_surge = (
                    market_data.loc[idx, 'volume'] >
                    market_data['volume'].rolling(20).mean().loc[idx] * 1.5
                )
                
                # Check price action
                subsequent_data = market_data.loc[idx:].head(5)
                spring_confirmation = (
                    subsequent_data['close'].iloc[-1] >
                    subsequent_data['open'].iloc[0] * 1.001
                )
                
                if volume_surge and spring_confirmation:
                    return {
                        'type': 'spring_test',
                        'subtype': 'wyckoff_spring',
                        'price_level': float(market_data.loc[idx, 'low']),
                        'strength': float(
                            market_data.loc[idx, 'volume'] /
                            market_data['volume'].rolling(20).mean().loc[idx]
                        ),
                        'timeframe': timeframe,
                        'context': {
                            'volume_surge': volume_surge,
                            'spring_confirmation': spring_confirmation,
                            'subsequent_close': float(subsequent_data['close'].iloc[-1])
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Spring test detection error: {str(e)}")
            return None
    
    async def _detect_composite_man(
        self,
        microstructure: MicroStructure,
        market_data: pd.DataFrame,
        timeframe: str
    ) -> Optional[Dict]:
        """Detect composite man movement pattern"""
        try:
            # Look for institutional footprint
            if not microstructure.institutional_footprint:
                return None
            
            # Find price clusters
            price_clusters = self._find_price_clusters(market_data)
            
            for cluster in price_clusters:
                # Check volume distribution
                volume_distribution = self._analyze_volume_distribution(
                    market_data,
                    cluster['price_range']
                )
                
                # Check for institutional characteristics
                if (
                    volume_distribution['skew'] > 0.7 and
                    volume_distribution['concentration'] > 0.8
                ):
                    return {
                        'type': 'composite_man',
                        'subtype': 'institutional_distribution',
                        'price_level': float(cluster['center_price']),
                        'strength': float(volume_distribution['strength']),
                        'timeframe': timeframe,
                        'volume_profile': volume_distribution
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Composite man detection error: {str(e)}")
            return None
    
    def _analyze_volume_distribution(
        self,
        market_data: pd.DataFrame,
        price_range: Tuple[float, float]
    ) -> Dict:
        """Analyze volume distribution in price range"""
        try:
            # Get data in price range
            mask = (
                (market_data['low'] >= price_range[0]) &
                (market_data['high'] <= price_range[1])
            )
            range_data = market_data[mask].copy()
            
            if len(range_data) < 5:
                return {
                    'skew': 0.0,
                    'concentration': 0.0,
                    'strength': 0.0
                }
            
            # Calculate volume metrics
            total_volume = range_data['volume'].sum()
            volume_std = range_data['volume'].std()
            volume_mean = range_data['volume'].mean()
            
            # Calculate distribution characteristics
            skew = (
                len(range_data[range_data['volume'] > volume_mean]) /
                len(range_data)
            )
            
            concentration = 1 - (volume_std / volume_mean)
            
            return {
                'skew': float(skew),
                'concentration': float(concentration),
                'strength': float(total_volume / market_data['volume'].mean()),
                'volume_profile': {
                    'total_volume': float(total_volume),
                    'avg_volume': float(volume_mean),
                    'std_volume': float(volume_std)
                }
            }
            
        except Exception as e:
            logger.error(f"Volume distribution analysis error: {str(e)}")
            return {
                'skew': 0.0,
                'concentration': 0.0,
                'strength': 0.0
            }
    
    def _find_price_clusters(
        self,
        market_data: pd.DataFrame
    ) -> List[Dict]:
        """Find price clusters with significant volume"""
        try:
            clusters = []
            
            # Calculate price levels
            price_levels = np.linspace(
                market_data['low'].min(),
                market_data['high'].max(),
                100
            )
            
            # Find volume clusters
            for i in range(len(price_levels) - 1):
                volume_in_range = market_data[
                    (market_data['low'] >= price_levels[i]) &
                    (market_data['high'] < price_levels[i + 1])
                ]['volume'].sum()
                
                if volume_in_range > market_data['volume'].mean() * 1.5:
                    clusters.append({
                        'price_range': (price_levels[i], price_levels[i + 1]),
                        'center_price': (price_levels[i] + price_levels[i + 1]) / 2,
                        'volume': float(volume_in_range)
                    })
            
            return clusters
            
        except Exception as e:
            logger.error(f"Price cluster detection error: {str(e)}")
            return []
    
    def _validate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Validate detected patterns"""
        valid_patterns = []
        try:
            for pattern in patterns:
                # Basic validation
                if not pattern or 'type' not in pattern:
                    continue
                
                # Check strength threshold
                if pattern.get('strength', 0) < 0.7:
                    continue
                
                # Volume validation
                if 'volume_profile' in pattern:
                    vol_profile = pattern['volume_profile']
                    if vol_profile.get('volume_ratio', 0) < 1.5:
                        continue
                
                valid_patterns.append(pattern)
            
            return valid_patterns
            
        except Exception as e:
            logger.error(f"Pattern validation error: {str(e)}")
            return []
    
    def _rank_by_probability(self, patterns: List[Dict]) -> List[Dict]:
        """Rank patterns by probability of success"""
        try:
            if not patterns:
                return []
            
            # Calculate probability scores
            scored_patterns = []
            for pattern in patterns:
                score = self._calculate_probability_score(pattern)
                scored_patterns.append((pattern, score))
            
            # Sort by score
            scored_patterns.sort(key=lambda x: x[1], reverse=True)
            
            # Return patterns only
            return [p[0] for p in scored_patterns]
            
        except Exception as e:
            logger.error(f"Pattern ranking error: {str(e)}")
            return patterns
    
    def _calculate_probability_score(self, pattern: Dict) -> float:
        """Calculate pattern probability score"""
        try:
            score = 0.0
            
            # Base score from pattern strength
            score += pattern.get('strength', 0) * 0.4
            
            # Volume profile score
            if 'volume_profile' in pattern:
                vol_profile = pattern['volume_profile']
                volume_score = min(
                    vol_profile.get('volume_ratio', 1.0) / 3,
                    1.0
                )
                score += volume_score * 0.3
            
            # Context score
            if 'context' in pattern:
                context = pattern['context']
                context_score = sum(
                    1 for v in context.values()
                    if isinstance(v, bool) and v
                ) / len(context)
                score += context_score * 0.3
            
            return score
            
        except Exception as e:
            logger.error(f"Probability score calculation error: {str(e)}")
            return 0.0
    
    def _get_default_structure(self) -> MicroStructure:
        """Return default microstructure when analysis fails"""
        return MicroStructure(
            absorption_levels={},
            institutional_footprint={},
            delta_divergence={},
            liquidity_profile={},
            order_flow_imbalance=0.0
        )