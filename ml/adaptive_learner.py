"""
Advanced real-time learning and adaptation system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import GradientBoostingRegressor
import torch
import torch.nn as nn
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    type: str  # 'trending', 'ranging', 'breakout', 'reversal'
    strength: float
    volatility: float
    liquidity: float
    timeframe: str

@dataclass
class AdaptationMetrics:
    win_rate: float
    profit_factor: float
    avg_return: float
    drawdown: float
    recovery_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    trades_per_day: float

class AdaptiveLearner:
    def __init__(
        self,
        learning_rate: float = 0.01,
        adaptation_threshold: float = 0.75,
        regime_memory: int = 1000,
        min_samples_learn: int = 100
    ):
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.regime_memory = regime_memory
        self.min_samples_learn = min_samples_learn
        
        # Initialize learners
        self.pattern_learner = self._init_pattern_learner()
        self.regime_classifier = self._init_regime_classifier()
        self.execution_optimizer = self._init_execution_optimizer()
        
        # State tracking
        self.current_regime = None
        self.performance_history = []
        self.adaptation_history = []
        
        # Pattern memory
        self.pattern_memory = defaultdict(list)
        self.execution_memory = defaultdict(list)
        
        # Performance metrics
        self.metrics = self._initialize_metrics()
    
    async def adapt_to_market(
        self,
        new_data: pd.DataFrame,
        patterns: List['InstitutionalPattern'],
        executions: List[Dict],
        market_structure: 'MarketStructure'
    ):
        """Real-time market adaptation"""
        try:
            # 1. Update Market Regime
            new_regime = await self._detect_regime(
                new_data,
                market_structure
            )
            
            # 2. Update Pattern Performance
            self._update_pattern_memory(patterns, executions)
            
            # 3. Adapt Pattern Recognition
            if len(self.pattern_memory) >= self.min_samples_learn:
                await self._adapt_pattern_recognition()
            
            # 4. Optimize Execution
            if len(self.execution_memory) >= self.min_samples_learn:
                await self._optimize_execution_parameters()
            
            # 5. Update Performance Metrics
            self._update_performance_metrics(executions)
            
            # 6. Trigger Adaptation if Needed
            if self._should_adapt():
                await self._perform_adaptation()
            
        except Exception as e:
            logger.error(f"Market adaptation error: {str(e)}")
    
    async def _detect_regime(
        self,
        data: pd.DataFrame,
        market_structure: 'MarketStructure'
    ) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Calculate key metrics
            volatility = self._calculate_volatility(data)
            trend_strength = self._calculate_trend_strength(data)
            liquidity = self._calculate_liquidity(data)
            
            # Classify regime
            regime_type = self._classify_regime(
                trend_strength,
                volatility,
                liquidity,
                market_structure
            )
            
            # Create regime object
            new_regime = MarketRegime(
                type=regime_type,
                strength=trend_strength,
                volatility=volatility,
                liquidity=liquidity,
                timeframe=market_structure.timeframe
            )
            
            # Update current regime
            self.current_regime = new_regime
            
            return new_regime
            
        except Exception as e:
            logger.error(f"Regime detection error: {str(e)}")
            return self.current_regime
    
    def _update_pattern_memory(
        self,
        patterns: List['InstitutionalPattern'],
        executions: List[Dict]
    ):
        """Update pattern performance memory"""
        try:
            # Match patterns with executions
            for pattern in patterns:
                matched_executions = [
                    ex for ex in executions
                    if self._match_pattern_execution(pattern, ex)
                ]
                
                if matched_executions:
                    # Calculate pattern performance
                    performance = self._calculate_pattern_performance(
                        pattern,
                        matched_executions
                    )
                    
                    # Store in memory
                    self.pattern_memory[pattern.type].append({
                        'pattern': pattern,
                        'performance': performance,
                        'regime': self.current_regime,
                        'timestamp': datetime.now()
                    })
                    
            # Trim old patterns
            self._trim_pattern_memory()
            
        except Exception as e:
            logger.error(f"Pattern memory update error: {str(e)}")
    
    async def _adapt_pattern_recognition(self):
        """Adapt pattern recognition parameters"""
        try:
            # Calculate pattern type performance
            type_performance = self._calculate_type_performance()
            
            # Adapt recognition parameters
            for pattern_type, performance in type_performance.items():
                if performance['trades'] >= self.min_samples_learn:
                    # Calculate adaptation factors
                    adaptation = self._calculate_adaptation_factors(
                        performance,
                        self.current_regime
                    )
                    
                    # Update pattern recognition parameters
                    await self._update_recognition_params(
                        pattern_type,
                        adaptation
                    )
            
        except Exception as e:
            logger.error(f"Pattern recognition adaptation error: {str(e)}")
    
    async def _optimize_execution_parameters(self):
        """Optimize execution parameters based on performance"""
        try:
            # Analyze execution performance
            execution_stats = self._analyze_execution_performance()
            
            # Optimize parameters
            optimized_params = self._calculate_optimal_parameters(
                execution_stats,
                self.current_regime
            )
            
            # Update execution engine
            await self._update_execution_params(optimized_params)
            
        except Exception as e:
            logger.error(f"Execution optimization error: {str(e)}")
    
    def _calculate_pattern_performance(
        self,
        pattern: 'InstitutionalPattern',
        executions: List[Dict]
    ) -> Dict:
        """Calculate comprehensive pattern performance metrics"""
        try:
            performance = {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_return': 0.0,
                'max_adverse_excursion': 0.0,
                'time_to_profit': 0.0,
                'risk_adjusted_return': 0.0
            }
            
            if not executions:
                return performance
            
            # Calculate metrics
            wins = [ex for ex in executions if ex['pnl'] > 0]
            losses = [ex for ex in executions if ex['pnl'] <= 0]
            
            # Win rate
            performance['win_rate'] = len(wins) / len(executions)
            
            # Profit factor
            total_profit = sum(ex['pnl'] for ex in wins)
            total_loss = abs(sum(ex['pnl'] for ex in losses))
            performance['profit_factor'] = (
                total_profit / total_loss if total_loss > 0 else float('inf')
            )
            
            # Average return
            performance['avg_return'] = np.mean([ex['pnl'] for ex in executions])
            
            # Maximum adverse excursion
            performance['max_adverse_excursion'] = max(
                abs(ex['max_drawdown']) for ex in executions
            )
            
            # Time to profit
            performance['time_to_profit'] = np.mean([
                ex['time_to_profit'].total_seconds()
                for ex in executions if ex['pnl'] > 0
            ])
            
            # Risk-adjusted return
            returns = [ex['pnl'] for ex in executions]
            if len(returns) > 1:
                performance['risk_adjusted_return'] = (
                    np.mean(returns) / np.std(returns)
                    if np.std(returns) > 0 else 0
                )
            
            return performance
            
        except Exception as e:
            logger.error(f"Pattern performance calculation error: {str(e)}")
            return {'win_rate': 0.0, 'profit_factor': 0.0}
    
    def _calculate_adaptation_factors(
        self,
        performance: Dict,
        regime: MarketRegime
    ) -> Dict:
        """Calculate adaptation factors for pattern recognition"""
        try:
            factors = {}
            
            # Base adaptation on performance metrics
            base_factor = min(
                performance['win_rate'] * performance['profit_factor'],
                1.0
            )
            
            # Adjust for market regime
            regime_multiplier = self._get_regime_multiplier(regime)
            
            # Calculate specific factors
            factors['strength_threshold'] = max(
                0.7,
                0.85 - (base_factor * regime_multiplier * 0.15)
            )
            
            factors['volume_impact'] = max(
                1.5,
                2.0 + (base_factor * regime_multiplier)
            )
            
            factors['confirmation_required'] = (
                base_factor < 0.8 or
                regime.volatility > 0.8
            )
            
            return factors
            
        except Exception as e:
            logger.error(f"Adaptation factors calculation error: {str(e)}")
            return {}
    
    @staticmethod
    def _get_regime_multiplier(regime: MarketRegime) -> float:
        """Get multiplier based on market regime"""
        multipliers = {
            'trending': 1.2,
            'ranging': 0.8,
            'breakout': 1.1,
            'reversal': 0.9
        }
        return multipliers.get(regime.type, 1.0)
    
    def _analyze_execution_performance(self) -> Dict:
        """Analyze execution performance statistics"""
        try:
            stats = {
                'slippage': [],
                'fill_time': [],
                'rejection_rate': [],
                'partial_fills': [],
                'price_improvement': []
            }
            
            # Calculate execution metrics
            for execution in self.execution_memory.values():
                stats['slippage'].append(
                    execution['executed_price'] - execution['intended_price']
                )
                stats['fill_time'].append(execution['fill_time'])
                stats['rejection_rate'].append(
                    1 if execution['rejected'] else 0
                )
                stats['partial_fills'].append(
                    execution['partial_fill_count']
                )
                stats['price_improvement'].append(
                    execution['price_improvement']
                )
            
            # Calculate averages
            return {
                key: float(np.mean(values))
                for key, values in stats.items()
                if values
            }
            
        except Exception as e:
            logger.error(f"Execution analysis error: {str(e)}")
            return {}
    
    def _calculate_optimal_parameters(
        self,
        stats: Dict,
        regime: MarketRegime
    ) -> Dict:
        """Calculate optimal execution parameters"""
        try:
            params = {}
            
            # Base parameters on execution stats
            avg_slippage = abs(stats.get('slippage', 0))
            avg_fill_time = stats.get('fill_time', 100)
            rejection_rate = stats.get('rejection_rate', 0)
            
            # Adjust for market regime
            volatility_factor = 1 + regime.volatility
            liquidity_factor = 1 / (1 + regime.liquidity)
            
            # Calculate optimal parameters
            params['max_slippage'] = max(
                1.0,  # Minimum 1 pip
                avg_slippage * volatility_factor * 1.5
            )
            
            params['order_timeout'] = max(
                1000,  # Minimum 1 second
                avg_fill_time * liquidity_factor * 2
            )
            
            params['retry_count'] = min(
                3,
                int(2 + rejection_rate * 5)
            )
            
            return params
            
        except Exception as e:
            logger.error(f"Optimal parameters calculation error: {str(e)}")
            return {}