"""
Core continuous learning and model evolution system
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader
import mlflow
from stable_baselines3 import PPO
import logging

logger = logging.getLogger(__name__)

class ContinuousLearner:
    def __init__(
        self,
        model_dir: str = "models",
        min_samples: int = 1000,
        update_interval: int = 24,  # hours
        min_performance: Optional[Dict[str, float]] = None
    ):
        # Default performance metrics
        self.min_performance = min_performance or {
            'accuracy': 0.78,      # Increased accuracy requirement
            'sharpe': 2.5,         # Higher Sharpe ratio for better risk-adjusted returns
            'drawdown': 0.12,      # Reduced maximum drawdown tolerance
            'profit_factor': 2.0,  # Minimum profit factor requirement
            'win_rate': 0.75,      # Minimum win rate
            'recovery_factor': 3.0  # Minimum recovery factor
        }
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_samples = min_samples
        self.update_interval = update_interval
        self.min_performance = min_performance
        
        # Initialize trackers
        self.last_update = datetime.now()
        self.training_in_progress = False
        self.performance_history = []
        
        # Set up MLflow
        mlflow.set_tracking_uri("sqlite:///models/mlflow.db")
        mlflow.set_experiment("AstraTrade-Evolution")
    
    async def should_update(self, recent_trades: pd.DataFrame) -> bool:
        """Check if model update is needed"""
        if self.training_in_progress:
            return False
            
        if len(recent_trades) < self.min_samples:
            return False
            
        hours_since_update = (datetime.now() - self.last_update).total_seconds() / 3600
        if hours_since_update < self.update_interval:
            return False
            
        return True
    
    async def evaluate_performance(
        self,
        trades: pd.DataFrame,
        predictions: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate key performance metrics"""
        try:
            # Calculate prediction metrics
            accuracy = accuracy_score(
                trades['direction'],
                predictions['predicted_direction']
            )
            
            precision = precision_score(
                trades['direction'],
                predictions['predicted_direction'],
                average='weighted'
            )
            
            # Calculate trading metrics
            returns = trades['pnl'].pct_change()
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
            
            drawdown = (
                trades['equity'].cummax() - trades['equity']
            ) / trades['equity'].cummax()
            max_drawdown = drawdown.max()
            
            win_rate = len(trades[trades['pnl'] > 0]) / len(trades)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0
            }
    
    async def train_ensemble(
        self,
        trades: pd.DataFrame,
        market_data: pd.DataFrame,
        current_model: 'MarketPredictor'
    ) -> Tuple[bool, Optional[Dict]]:
        """Train new version of prediction ensemble"""
        try:
            with mlflow.start_run(run_name=f"ensemble_train_{datetime.now()}"):
                # Prepare training data
                X_train, y_train = self._prepare_training_data(
                    trades, market_data
                )
                
                # Create new model instance
                new_model = current_model.clone()
                
                # Train on new data
                train_metrics = await new_model.train(
                    X_train,
                    y_train,
                    continue_training=True
                )
                
                # Log metrics
                mlflow.log_metrics(train_metrics)
                
                # Evaluate on validation set
                val_metrics = await new_model.evaluate(
                    X_train[-1000:],
                    y_train[-1000:]
                )
                
                mlflow.log_metrics({
                    f"val_{k}": v for k, v in val_metrics.items()
                })
                
                # Save if performance is good enough
                if self._check_performance(val_metrics):
                    model_path = self.model_dir / f"ensemble_{datetime.now():%Y%m%d_%H%M}.pt"
                    torch.save(new_model.state_dict(), model_path)
                    
                    mlflow.log_artifact(str(model_path))
                    return True, val_metrics
                    
                return False, val_metrics
                
        except Exception as e:
            logger.error(f"Error training ensemble: {str(e)}")
            return False, None
    
    async def train_rl_agent(
        self,
        trades: pd.DataFrame,
        current_agent: PPO
    ) -> Tuple[bool, Optional[Dict]]:
        """Train new version of RL strategy"""
        try:
            with mlflow.start_run(run_name=f"rl_train_{datetime.now()}"):
                # Create training environment
                env = self._create_training_env(trades)
                
                # Clone current agent
                new_agent = PPO(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    tensorboard_log="logs/rl_tensorboard/"
                )
                new_agent.set_parameters(current_agent.get_parameters())
                
                # Train for specified steps
                new_agent.learn(
                    total_timesteps=50000,
                    callback=self._rl_callback
                )
                
                # Evaluate agent
                eval_metrics = self._evaluate_rl_agent(new_agent, env)
                
                mlflow.log_metrics(eval_metrics)
                
                # Save if performance is good
                if self._check_performance(eval_metrics):
                    model_path = self.model_dir / f"rl_{datetime.now():%Y%m%d_%H%M}.zip"
                    new_agent.save(str(model_path))
                    
                    mlflow.log_artifact(str(model_path))
                    return True, eval_metrics
                    
                return False, eval_metrics
                
        except Exception as e:
            logger.error(f"Error training RL agent: {str(e)}")
            return False, None
    
    def _prepare_training_data(
        self,
        trades: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        features = []
        labels = []
        
        for idx, trade in trades.iterrows():
            # Get market data leading up to trade
            trade_data = market_data[
                market_data.index <= trade['entry_time']
            ].tail(100)
            
            if len(trade_data) < 100:
                continue
                
            # Extract features
            feature_vector = np.concatenate([
                trade_data['close'].values,
                trade_data['volume'].values,
                trade_data['high'].values,
                trade_data['low'].values,
                [
                    trade['risk_score'],
                    trade['market_volatility'],
                    trade['news_sentiment']
                ]
            ])
            
            features.append(feature_vector)
            labels.append(1 if trade['pnl'] > 0 else 0)
        
        return np.array(features), np.array(labels)
    
    def _check_performance(self, metrics: Dict[str, float]) -> bool:
        """
        Check if metrics meet minimum performance criteria with advanced validation
        """
        # Basic performance checks
        basic_checks = (
            metrics.get(k, 0) >= v 
            for k, v in self.min_performance.items()
        )
        
        if not all(basic_checks):
            return False
            
        # Advanced stability checks
        stability_score = self._calculate_stability_score(metrics)
        if stability_score < 0.8:  # Require 80% stability
            return False
            
        # Check for overfitting
        if self._detect_overfitting(metrics):
            return False
            
        # Verify model quality
        model_quality = self._assess_model_quality(metrics)
        if model_quality < 0.85:  # Require 85% model quality
            return False
            
        return True
        
    def _calculate_stability_score(self, metrics: Dict[str, float]) -> float:
        """Calculate model stability score"""
        try:
            # Analyze metric consistency
            consistency_scores = [
                min(1.0, metrics.get('train_accuracy', 0) / metrics.get('val_accuracy', 1)),
                min(1.0, metrics.get('train_sharpe', 0) / metrics.get('val_sharpe', 1)),
                1 - abs(metrics.get('train_drawdown', 0) - metrics.get('val_drawdown', 0))
            ]
            return sum(consistency_scores) / len(consistency_scores)
        except ZeroDivisionError:
            return 0.0
            
    def _detect_overfitting(self, metrics: Dict[str, float]) -> bool:
        """Check for signs of overfitting"""
        # Check for excessive performance gap
        train_val_gap = abs(
            metrics.get('train_accuracy', 0) - 
            metrics.get('val_accuracy', 0)
        )
        
        if train_val_gap > 0.1:  # More than 10% gap indicates overfitting
            return True
            
        # Check for unrealistic performance
        if metrics.get('train_accuracy', 0) > 0.95:  # Too good to be true
            return True
            
        return False
        
    def _assess_model_quality(self, metrics: Dict[str, float]) -> float:
        """Assess overall model quality score"""
        quality_metrics = {
            'accuracy': (metrics.get('val_accuracy', 0), 0.4),  # 40% weight
            'sharpe': (min(metrics.get('val_sharpe', 0) / 3, 1), 0.2),  # 20% weight
            'drawdown': (1 - metrics.get('val_drawdown', 1), 0.2),  # 20% weight
            'profit_factor': (min(metrics.get('profit_factor', 0) / 3, 1), 0.2)  # 20% weight
        }
        
        return sum(score * weight for score, weight in quality_metrics.values())
    
    def _create_training_env(self, trades: pd.DataFrame):
        """Create gym environment for RL training"""
        from stable_baselines3.common.env_util import make_vec_env
        
        # Custom trading environment implementation
        # (This would be defined elsewhere)
        from utils.trading_env import TradingEnvironment
        
        env = make_vec_env(
            lambda: TradingEnvironment(trades),
            n_envs=4
        )
        return env
    
    def _evaluate_rl_agent(self, agent: PPO, env) -> Dict[str, float]:
        """Evaluate RL agent performance"""
        obs = env.reset()
        done = False
        returns = []
        
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            returns.append(reward)
            
        return {
            'mean_return': np.mean(returns),
            'sharpe': np.mean(returns) / np.std(returns),
            'max_drawdown': self._calculate_drawdown(returns)
        }
    
    def _calculate_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        return np.max(drawdown)
    
    def _rl_callback(self, locals_: Dict, globals_: Dict) -> bool:
        """Callback for RL training to log metrics"""
        self._episode_count += 1
        
        if self._episode_count % 100 == 0:
            logs = locals_['infos'][0].get('episode')
            if logs:
                mlflow.log_metrics({
                    'episode_reward': logs['r'],
                    'episode_length': logs['l']
                }, step=self._episode_count)
        
        return True