"""
Dynamic risk management and position sizing system using Kelly Criterion
and correlation-based portfolio adjustments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from scipy.stats import norm
from datetime import datetime, timedelta

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskMetrics:
    var_99: float  # 99% Value at Risk
    expected_shortfall: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    correlation_impact: float
    volatility: float
    kelly_fraction: float
    risk_level: RiskLevel

@dataclass
class PositionSize:
    pair: str
    size: float
    risk_amount: float
    kelly_fraction: float
    adjusted_fraction: float
    confidence: float

class RiskGuardian:
    """
    Advanced risk management system with dynamic position sizing
    and portfolio-wide risk controls
    """
    def __init__(
        self,
        base_risk: float = 0.003,  # 0.3% base risk per trade
        max_portfolio_risk: float = 0.05,  # 5% maximum portfolio risk
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        max_correlation: float = 0.7,  # Maximum allowed correlation between positions
        drawdown_limit: float = 0.15,  # 15% maximum drawdown limit
        volatility_lookback: int = 20,  # Days for volatility calculation
        confidence_level: float = 0.99  # VaR confidence level
    ):
        self.base_risk = base_risk
        self.max_portfolio_risk = max_portfolio_risk
        self.risk_free_rate = risk_free_rate
        self.max_correlation = max_correlation
        self.drawdown_limit = drawdown_limit
        self.volatility_lookback = volatility_lookback
        self.confidence_level = confidence_level
        
        # Initialize tracking variables
        self.portfolio_history: List[Dict] = []
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.current_positions: Dict[str, Dict] = {}
        self.daily_returns: Dict[str, List[float]] = {}

    def calculate_position_size(
        self,
        pair: str,
        entry_price: float,
        stop_loss: float,
        account_balance: float,
        win_probability: float,
        volatility: Optional[float] = None,
        news_risk: float = 0.0
    ) -> PositionSize:
        """
        Calculate optimal position size using Kelly Criterion
        with dynamic adjustments
        """
        # Calculate base risk amount
        risk_amount = account_balance * self.base_risk
        
        # Calculate potential loss and gain
        potential_loss = abs(entry_price - stop_loss)
        potential_gain = potential_loss * 2  # Assuming 1:2 RR minimum
        
        # Calculate Kelly fraction
        win_ratio = win_probability
        loss_ratio = 1 - win_ratio
        kelly_fraction = (win_ratio * potential_gain - loss_ratio * potential_loss) / potential_gain
        
        # Apply risk adjustments
        adjusted_fraction = self._adjust_kelly_fraction(
            kelly_fraction,
            pair,
            volatility,
            news_risk
        )
        
        # Calculate final position size
        risk_adjusted_size = (risk_amount * adjusted_fraction) / potential_loss
        
        # Apply portfolio-wide risk limits
        portfolio_adjusted_size = self._apply_portfolio_limits(
            pair,
            risk_adjusted_size,
            potential_loss,
            account_balance
        )
        
        return PositionSize(
            pair=pair,
            size=portfolio_adjusted_size,
            risk_amount=risk_amount,
            kelly_fraction=kelly_fraction,
            adjusted_fraction=adjusted_fraction,
            confidence=win_probability
        )

    def _adjust_kelly_fraction(
        self,
        kelly_fraction: float,
        pair: str,
        volatility: Optional[float],
        news_risk: float
    ) -> float:
        """
        Apply dynamic adjustments to Kelly fraction based on market conditions
        """
        # Start with half-Kelly for safety
        adjusted_fraction = kelly_fraction * 0.5
        
        # Volatility adjustment
        if volatility:
            vol_impact = self._calculate_volatility_impact(volatility)
            adjusted_fraction *= (1 - vol_impact)
        
        # News risk adjustment
        adjusted_fraction *= (1 - news_risk)
        
        # Correlation adjustment
        if self.correlation_matrix is not None and pair in self.correlation_matrix:
            corr_impact = self._calculate_correlation_impact(pair)
            adjusted_fraction *= (1 - corr_impact)
        
        # Ensure fraction stays within reasonable bounds
        return max(0.0, min(adjusted_fraction, 1.0))

    def _calculate_volatility_impact(self, volatility: float) -> float:
        """
        Calculate impact factor based on current volatility
        """
        # Compare current volatility to historical average
        if not self.daily_returns:
            return 0.0
        
        hist_vol = np.std([
            r for returns in self.daily_returns.values()
            for r in returns
        ])
        
        if hist_vol == 0:
            return 0.0
        
        vol_ratio = volatility / hist_vol
        return min(0.5, max(0.0, (vol_ratio - 1) * 0.5))

    def _calculate_correlation_impact(self, pair: str) -> float:
        """
        Calculate impact factor based on correlation with existing positions
        """
        if not self.correlation_matrix is not None:
            return 0.0
        
        # Get correlations with existing positions
        correlations = [
            self.correlation_matrix.loc[pair, p]
            for p in self.current_positions
            if p in self.correlation_matrix.columns
        ]
        
        if not correlations:
            return 0.0
        
        # Use maximum correlation as impact factor
        max_corr = max(abs(np.array(correlations)))
        return min(0.5, max(0.0, (max_corr - self.max_correlation)))

    def _apply_portfolio_limits(
        self,
        pair: str,
        position_size: float,
        potential_loss: float,
        account_balance: float
    ) -> float:
        """
        Apply portfolio-wide risk limits and adjust position size
        """
        # Calculate current portfolio risk
        current_risk = sum(
            pos['risk_amount'] / account_balance
            for pos in self.current_positions.values()
        )
        
        # Calculate new position risk
        new_position_risk = (position_size * potential_loss) / account_balance
        
        # Check if new position would exceed portfolio risk limit
        if current_risk + new_position_risk > self.max_portfolio_risk:
            # Scale down position size to meet risk limit
            available_risk = self.max_portfolio_risk - current_risk
            if available_risk <= 0:
                return 0.0
            return (available_risk * account_balance) / potential_loss
        
        return position_size

    def update_portfolio_metrics(
        self,
        positions: Dict[str, Dict],
        returns: Dict[str, float],
        prices: Dict[str, pd.DataFrame]
    ) -> RiskMetrics:
        """
        Update portfolio risk metrics and correlation matrix
        """
        self.current_positions = positions
        
        # Update returns history
        for pair, ret in returns.items():
            if pair not in self.daily_returns:
                self.daily_returns[pair] = []
            self.daily_returns[pair].append(ret)
            # Keep limited history
            self.daily_returns[pair] = self.daily_returns[pair][-self.volatility_lookback:]
        
        # Update correlation matrix
        self._update_correlation_matrix(prices)
        
        # Calculate portfolio metrics
        metrics = self._calculate_risk_metrics(returns)
        
        # Update portfolio history
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics.__dict__,
            'positions': {
                k: {'size': v['size'], 'risk': v['risk_amount']}
                for k, v in positions.items()
            }
        })
        
        # Trim history to last 90 days
        cutoff = datetime.now() - timedelta(days=90)
        self.portfolio_history = [
            h for h in self.portfolio_history
            if h['timestamp'] >= cutoff
        ]
        
        return metrics

    def _update_correlation_matrix(self, prices: Dict[str, pd.DataFrame]) -> None:
        """
        Update correlation matrix using recent price data
        """
        # Calculate returns for correlation
        returns_data = {}
        for pair, data in prices.items():
            if 'Close' in data.columns:
                returns_data[pair] = data['Close'].pct_change().dropna()
        
        if returns_data:
            self.correlation_matrix = pd.DataFrame(returns_data).corr()

    def _calculate_risk_metrics(self, returns: Dict[str, float]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the portfolio
        """
        if not returns:
            return RiskMetrics(
                var_99=0.0,
                expected_shortfall=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                correlation_impact=0.0,
                volatility=0.0,
                kelly_fraction=0.0,
                risk_level=RiskLevel.LOW
            )
        
        # Convert returns to numpy array
        returns_arr = np.array(list(returns.values()))
        
        # Calculate metrics
        volatility = np.std(returns_arr) * np.sqrt(252)  # Annualized
        mean_return = np.mean(returns_arr) * 252  # Annualized
        
        # VaR and Expected Shortfall
        var_99 = norm.ppf(1 - self.confidence_level) * volatility
        expected_shortfall = -np.mean(
            returns_arr[returns_arr < var_99]
        ) if any(returns_arr < var_99) else var_99
        
        # Sharpe and Sortino ratios
        excess_return = mean_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        downside_returns = returns_arr[returns_arr < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = excess_return / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        # Correlation impact
        correlation_impact = 0.0
        if self.correlation_matrix is not None and len(self.correlation_matrix) > 1:
            correlation_impact = np.mean(np.abs(self.correlation_matrix.values))
        
        # Determine risk level
        risk_level = self._determine_risk_level(
            volatility,
            max_drawdown,
            correlation_impact,
            var_99
        )
        
        # Calculate portfolio Kelly fraction
        portfolio_kelly = self._calculate_portfolio_kelly(
            mean_return,
            volatility,
            correlation_impact
        )
        
        return RiskMetrics(
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            correlation_impact=correlation_impact,
            volatility=volatility,
            kelly_fraction=portfolio_kelly,
            risk_level=risk_level
        )

    def _determine_risk_level(
        self,
        volatility: float,
        max_drawdown: float,
        correlation_impact: float,
        var_99: float
    ) -> RiskLevel:
        """
        Determine overall risk level based on multiple factors
        """
        risk_scores = [
            volatility / 0.2,  # Score of 1 at 20% volatility
            max_drawdown / self.drawdown_limit,
            correlation_impact / self.max_correlation,
            abs(var_99) / 0.03  # Score of 1 at 3% daily VaR
        ]
        
        avg_score = np.mean(risk_scores)
        
        if avg_score < 0.5:
            return RiskLevel.LOW
        elif avg_score < 0.75:
            return RiskLevel.MEDIUM
        elif avg_score < 1.0:
            return RiskLevel.HIGH
        return RiskLevel.EXTREME

    def _calculate_portfolio_kelly(
        self,
        mean_return: float,
        volatility: float,
        correlation_impact: float
    ) -> float:
        """
        Calculate Kelly fraction for the entire portfolio
        """
        if volatility == 0:
            return 0.0
        
        # Basic Kelly calculation
        kelly = (mean_return - self.risk_free_rate) / (volatility ** 2)
        
        # Adjust for correlation
        kelly *= (1 - correlation_impact)
        
        # Conservative adjustment
        kelly *= 0.5  # Half-Kelly for safety
        
        return max(0.0, min(1.0, kelly))

    def should_reduce_exposure(self, metrics: RiskMetrics) -> bool:
        """
        Determine if portfolio exposure should be reduced
        """
        return (
            metrics.risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME] or
            metrics.max_drawdown > self.drawdown_limit or
            metrics.var_99 < -0.03  # 3% daily VaR threshold
        )

    def get_portfolio_summary(self) -> Dict:
        """
        Get current portfolio risk summary
        """
        if not self.portfolio_history:
            return {}
        
        latest = self.portfolio_history[-1]
        return {
            'timestamp': latest['timestamp'].isoformat(),
            'metrics': latest['metrics'],
            'positions': latest['positions'],
            'risk_status': {
                'high_risk_pairs': [
                    pair for pair, pos in self.current_positions.items()
                    if pos['risk_amount'] > self.base_risk
                ],
                'correlation_warnings': [
                    (p1, p2) for p1 in self.current_positions
                    for p2 in self.current_positions
                    if p1 < p2 and self._get_correlation(p1, p2) > self.max_correlation
                ],
                'risk_concentration': max(
                    (pos['risk_amount'] for pos in self.current_positions.values()),
                    default=0.0
                )
            }
        }

    def _get_correlation(self, pair1: str, pair2: str) -> float:
        """Get correlation between two pairs"""
        if self.correlation_matrix is None:
            return 0.0
        if pair1 in self.correlation_matrix.columns and pair2 in self.correlation_matrix.columns:
            return abs(self.correlation_matrix.loc[pair1, pair2])
        return 0.0