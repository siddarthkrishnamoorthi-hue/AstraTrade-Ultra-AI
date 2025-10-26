"""
Advanced strategy execution with institutional pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.stats import linregress
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class SetupCondition:
    type: str  # 'breakout', 'reversal', 'continuation'
    timeframe: str
    confidence: float
    context: Dict[str, any]
    entry_zone: Tuple[float, float]
    stop_zone: Tuple[float, float]
    targets: List[float]

@dataclass
class TradeSetup:
    condition: SetupCondition
    direction: str
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    risk_ratio: float
    quality_score: float

class InstitutionalStrategy:
    def __init__(
        self,
        min_quality_score: float = 0.8,
        min_risk_ratio: float = 2.0,
        max_correlation: float = 0.6
    ):
        self.min_quality_score = min_quality_score
        self.min_risk_ratio = min_risk_ratio
        self.max_correlation = max_correlation
        
        # Pattern storage
        self.detected_setups = []
        self.market_context = {}
        self.execution_history = {}
        
        # Performance tracking
        self.setup_performance = {}
        self.market_conditions = {}
    
    async def analyze_setups(
        self,
        market_structure: 'MarketStructure',
        order_flow: Dict,
        timeframes: Dict[str, pd.DataFrame]
    ) -> List[TradeSetup]:
        """Analyze market for institutional setups"""
        try:
            # 1. Analyze Market Context
            context = self._analyze_market_context(
                timeframes,
                market_structure
            )
            
            # 2. Find Key Market Structures
            structures = self._find_key_structures(
                timeframes,
                market_structure
            )
            
            # 3. Detect Institutional Patterns
            patterns = await self._detect_institutional_patterns(
                structures,
                order_flow,
                context
            )
            
            # 4. Filter and Score Setups
            valid_setups = []
            for pattern in patterns:
                setup = self._validate_and_score_setup(
                    pattern,
                    context,
                    market_structure
                )
                if setup and setup.quality_score >= self.min_quality_score:
                    valid_setups.append(setup)
            
            # 5. Apply Smart Filtering
            final_setups = self._smart_filter_setups(valid_setups, context)
            
            return final_setups
            
        except Exception as e:
            logger.error(f"Setup analysis error: {str(e)}")
            return []
    
    def _analyze_market_context(
        self,
        timeframes: Dict[str, pd.DataFrame],
        market_structure: 'MarketStructure'
    ) -> Dict:
        """Analyze broader market context"""
        try:
            context = {}
            
            # Analyze market regime
            context['regime'] = self._determine_market_regime(timeframes['H4'])
            
            # Analyze market efficiency
            context['efficiency'] = self._calculate_market_efficiency(
                timeframes['H1']
            )
            
            # Check institutional activity
            context['institutional_activity'] = (
                market_structure.market_context.get('institutional_activity', 0)
            )
            
            # Analyze volatility state
            context['volatility_state'] = self._analyze_volatility_state(
                timeframes
            )
            
            # Determine trend strength
            context['trend_strength'] = self._calculate_trend_strength(
                timeframes,
                market_structure
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Market context analysis error: {str(e)}")
            return {}
    
    def _determine_market_regime(self, data: pd.DataFrame) -> str:
        """Determine current market regime"""
        try:
            # Calculate directional movement
            closes = data['close'].values
            direction_strength = np.sum(np.diff(closes) > 0) / (len(closes) - 1)
            
            # Calculate volatility regime
            volatility = data['high'].values - data['low'].values
            vol_ratio = np.mean(volatility[-20:]) / np.mean(volatility[-100:])
            
            if direction_strength > 0.7:
                return 'strong_trend'
            elif direction_strength < 0.3:
                return 'reversal'
            elif vol_ratio > 1.5:
                return 'expansion'
            elif vol_ratio < 0.5:
                return 'contraction'
            else:
                return 'ranging'
                
        except Exception as e:
            logger.error(f"Market regime analysis error: {str(e)}")
            return 'unknown'
    
    def _calculate_market_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate market efficiency ratio"""
        try:
            price_movement = abs(data['close'].iloc[-1] - data['close'].iloc[0])
            path_length = np.sum(abs(np.diff(data['close'])))
            
            return price_movement / path_length if path_length > 0 else 0
            
        except Exception as e:
            logger.error(f"Market efficiency calculation error: {str(e)}")
            return 0.0
    
    async def _detect_institutional_patterns(
        self,
        structures: Dict,
        order_flow: Dict,
        context: Dict
    ) -> List[Dict]:
        """Detect institutional trading patterns"""
        try:
            patterns = []
            
            # Check for accumulation/distribution
            if order_flow['institutional_activity'] > 0.7:
                accumulation = self._detect_accumulation(
                    structures,
                    order_flow
                )
                if accumulation:
                    patterns.append(accumulation)
            
            # Check for liquidity grabs
            liquidity_patterns = self._detect_liquidity_patterns(
                structures,
                context
            )
            patterns.extend(liquidity_patterns)
            
            # Check for order block setups
            order_blocks = self._detect_order_blocks(
                structures,
                order_flow
            )
            patterns.extend(order_blocks)
            
            # Check for breaker blocks
            breaker_patterns = self._detect_breaker_patterns(
                structures,
                context
            )
            patterns.extend(breaker_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection error: {str(e)}")
            return []
    
    def _detect_accumulation(
        self,
        structures: Dict,
        order_flow: Dict
    ) -> Optional[Dict]:
        """Detect institutional accumulation patterns"""
        try:
            # Check for absorption
            if (
                structures.get('volume_delta', 0) > 0 and
                structures.get('price_range', 1) < 0.3
            ):
                return {
                    'type': 'accumulation',
                    'confidence': order_flow['institutional_activity'],
                    'direction': 'long',
                    'entry_zone': (
                        structures['support'],
                        structures['support'] * 1.001
                    ),
                    'stop_zone': (
                        structures['support'] * 0.998,
                        structures['support'] * 0.999
                    ),
                    'targets': [
                        structures['resistance'],
                        structures['resistance'] * 1.02
                    ]
                }
            return None
            
        except Exception as e:
            logger.error(f"Accumulation detection error: {str(e)}")
            return None
    
    def _detect_liquidity_patterns(
        self,
        structures: Dict,
        context: Dict
    ) -> List[Dict]:
        """Detect liquidity engineering patterns"""
        patterns = []
        try:
            # Look for stop runs
            if structures.get('liquidity_levels'):
                for level in structures['liquidity_levels']:
                    if self._validate_liquidity_level(level, context):
                        patterns.append({
                            'type': 'liquidity_grab',
                            'confidence': level['strength'],
                            'direction': 'short' if level['type'] == 'resistance' else 'long',
                            'entry_zone': (
                                level['price'] * 0.999,
                                level['price'] * 1.001
                            ),
                            'stop_zone': (
                                level['price'] * 1.002,
                                level['price'] * 1.003
                            ) if level['type'] == 'resistance' else (
                                level['price'] * 0.997,
                                level['price'] * 0.998
                            ),
                            'targets': [
                                level['price'] * 0.995,
                                level['price'] * 0.99
                            ] if level['type'] == 'resistance' else [
                                level['price'] * 1.005,
                                level['price'] * 1.01
                            ]
                        })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Liquidity pattern detection error: {str(e)}")
            return []
    
    def _detect_order_blocks(
        self,
        structures: Dict,
        order_flow: Dict
    ) -> List[Dict]:
        """Detect institutional order blocks"""
        blocks = []
        try:
            if structures.get('order_blocks'):
                for block in structures['order_blocks']:
                    # Validate order block
                    if self._validate_order_block(block, order_flow):
                        blocks.append({
                            'type': 'order_block',
                            'confidence': block['strength'],
                            'direction': block['direction'],
                            'entry_zone': (
                                block['low'],
                                block['high']
                            ),
                            'stop_zone': (
                                block['low'] * 0.998,
                                block['low'] * 0.999
                            ) if block['direction'] == 'long' else (
                                block['high'] * 1.001,
                                block['high'] * 1.002
                            ),
                            'targets': [
                                block['target1'],
                                block['target2']
                            ]
                        })
            
            return blocks
            
        except Exception as e:
            logger.error(f"Order block detection error: {str(e)}")
            return []
    
    def _validate_and_score_setup(
        self,
        pattern: Dict,
        context: Dict,
        market_structure: 'MarketStructure'
    ) -> Optional[TradeSetup]:
        """Validate and score trade setup"""
        try:
            # Basic validation
            if not self._basic_validation(pattern, context):
                return None
            
            # Calculate setup quality score
            quality_score = self._calculate_setup_score(
                pattern,
                context,
                market_structure
            )
            
            # Calculate risk ratio
            risk_ratio = self._calculate_risk_ratio(pattern)
            
            if quality_score >= self.min_quality_score and risk_ratio >= self.min_risk_ratio:
                return TradeSetup(
                    condition=SetupCondition(
                        type=pattern['type'],
                        timeframe=context.get('timeframe', 'H1'),
                        confidence=pattern['confidence'],
                        context=context,
                        entry_zone=pattern['entry_zone'],
                        stop_zone=pattern['stop_zone'],
                        targets=pattern['targets']
                    ),
                    direction=pattern['direction'],
                    entry_price=np.mean(pattern['entry_zone']),
                    stop_loss=np.mean(pattern['stop_zone']),
                    take_profits=pattern['targets'],
                    risk_ratio=risk_ratio,
                    quality_score=quality_score
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Setup validation error: {str(e)}")
            return None
    
    def _calculate_setup_score(
        self,
        pattern: Dict,
        context: Dict,
        market_structure: 'MarketStructure'
    ) -> float:
        """Calculate comprehensive setup quality score"""
        try:
            scores = []
            
            # Pattern strength (30%)
            scores.append(pattern['confidence'] * 0.3)
            
            # Market context alignment (25%)
            context_score = self._score_context_alignment(
                pattern,
                context,
                market_structure
            )
            scores.append(context_score * 0.25)
            
            # Historical performance (20%)
            if pattern['type'] in self.setup_performance:
                hist_score = self.setup_performance[pattern['type']]['win_rate']
                scores.append(hist_score * 0.2)
            
            # Risk/reward quality (15%)
            risk_reward = self._calculate_risk_ratio(pattern)
            scores.append(min(risk_reward / 3, 1.0) * 0.15)
            
            # Technical confluence (10%)
            tech_score = self._score_technical_confluence(
                pattern,
                market_structure
            )
            scores.append(tech_score * 0.1)
            
            return sum(scores)
            
        except Exception as e:
            logger.error(f"Setup scoring error: {str(e)}")
            return 0.0
    
    def _score_context_alignment(
        self,
        pattern: Dict,
        context: Dict,
        market_structure: 'MarketStructure'
    ) -> float:
        """Score how well setup aligns with market context"""
        try:
            score = 0.0
            
            # Trend alignment
            if pattern['direction'] == market_structure.trend_state:
                score += 0.4
            
            # Market regime compatibility
            regime_compat = {
                'accumulation': ['ranging', 'contraction'],
                'liquidity_grab': ['strong_trend', 'expansion'],
                'order_block': ['strong_trend', 'expansion']
            }
            if context['regime'] in regime_compat.get(pattern['type'], []):
                score += 0.3
            
            # Volatility state compatibility
            if (
                (pattern['type'] == 'liquidity_grab' and context['volatility_state'] == 'high') or
                (pattern['type'] == 'accumulation' and context['volatility_state'] == 'low')
            ):
                score += 0.3
            
            return score
            
        except Exception as e:
            logger.error(f"Context alignment scoring error: {str(e)}")
            return 0.0
    
    def _score_technical_confluence(
        self,
        pattern: Dict,
        market_structure: 'MarketStructure'
    ) -> float:
        """Score technical confluence of setup"""
        try:
            score = 0.0
            entry_price = np.mean(pattern['entry_zone'])
            
            # Check key level confluence
            for level in market_structure.key_levels.values():
                if abs(level['price'] - entry_price) / entry_price < 0.001:
                    score += 0.3
                    break
            
            # Check order flow confluence
            if market_structure.order_flow.get('institutional_activity', 0) > 0.7:
                score += 0.4
            
            # Check volume profile confluence
            if entry_price in market_structure.volume_profile:
                score += 0.3
            
            return score
            
        except Exception as e:
            logger.error(f"Technical confluence scoring error: {str(e)}")
            return 0.0
    
    def _smart_filter_setups(
        self,
        setups: List[TradeSetup],
        context: Dict
    ) -> List[TradeSetup]:
        """Apply smart filtering to remove suboptimal setups"""
        try:
            if not setups:
                return []
            
            # Sort by quality score
            setups = sorted(
                setups,
                key=lambda x: x.quality_score,
                reverse=True
            )
            
            # Filter correlated setups
            filtered = []
            for setup in setups:
                # Check correlation with existing setups
                if not self._is_correlated_with_existing(
                    setup,
                    filtered,
                    context
                ):
                    filtered.append(setup)
            
            # Apply market regime filters
            if context['regime'] == 'ranging':
                filtered = [
                    s for s in filtered
                    if s.condition.type in ['accumulation', 'liquidity_grab']
                ]
            elif context['regime'] == 'strong_trend':
                filtered = [
                    s for s in filtered
                    if s.condition.type in ['order_block', 'breaker']
                ]
            
            return filtered
            
        except Exception as e:
            logger.error(f"Setup filtering error: {str(e)}")
            return []
    
    def _is_correlated_with_existing(
        self,
        setup: TradeSetup,
        existing: List[TradeSetup],
        context: Dict
    ) -> bool:
        """Check if setup is correlated with existing ones"""
        try:
            for exist in existing:
                # Check price proximity
                if abs(setup.entry_price - exist.entry_price) / setup.entry_price < 0.002:
                    return True
                
                # Check direction correlation
                if setup.direction == exist.direction:
                    return True
                
                # Check timeframe correlation
                if setup.condition.timeframe == exist.condition.timeframe:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Correlation check error: {str(e)}")
            return True  # Conservative approach
    
    @staticmethod
    def _calculate_risk_ratio(pattern: Dict) -> float:
        """Calculate risk/reward ratio"""
        try:
            entry = np.mean(pattern['entry_zone'])
            stop = np.mean(pattern['stop_zone'])
            target = pattern['targets'][0]  # First target
            
            risk = abs(entry - stop)
            reward = abs(target - entry)
            
            return reward / risk if risk > 0 else 0
            
        except Exception as e:
            logger.error(f"Risk ratio calculation error: {str(e)}")
            return 0.0