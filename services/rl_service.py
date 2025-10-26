"""
Reinforcement Learning training service
"""

import asyncio
import os
import logging
import json
import redis
from kafka import KafkaProducer, KafkaConsumer
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import mlflow

from rl.evolutor import StrategyEvolutor, TradingEnvironment
from ml.ensemble import MarketPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RLService")

class TrainingCallback(BaseCallback):
    """Callback for tracking training progress"""
    def __init__(self, check_freq: int = 1000):
        super().__init__(verbose=1)
        self.check_freq = check_freq
        self.best_reward = -np.inf
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Log metrics
            mlflow.log_metric(
                "reward",
                self.training_env.get_episode_rewards()[-1]
            )
            mlflow.log_metric(
                "episodes",
                len(self.training_env.get_episode_rewards())
            )
        return True

class RLService:
    def __init__(
        self,
        redis_host: str = "localhost",
        kafka_broker: str = "localhost:9092"
    ):
        self.redis = redis.Redis(host=redis_host)
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_broker,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.kafka_consumer = KafkaConsumer(
            'market_data',
            bootstrap_servers=kafka_broker,
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )
        
        # Initialize components
        self.strategy_evolutor = StrategyEvolutor()
        self.market_predictor = MarketPredictor()
        
        # Training state
        self.is_training = False
        self.current_episode = 0
        self.best_reward = -np.inf
        
        # MLflow setup
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("rl_training")
    
    async def start(self):
        """Start the RL service"""
        try:
            # Initialize models
            await self._initialize_models()
            
            # Start background tasks
            asyncio.create_task(self._continuous_training())
            asyncio.create_task(self._process_market_data())
            
            logger.info("RL service started successfully")
            
            # Keep service running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Service error: {str(e)}")
            await self.shutdown()
    
    async def _initialize_models(self):
        """Initialize RL models"""
        try:
            # Load cached models
            self.strategy_evolutor.load_models()
            
            # Create training environment
            self.env = TradingEnvironment(
                prediction_model=self.market_predictor,
                reward_scaling=True,
                observation_normalization=True
            )
            
            # Initialize PPO model
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                tensorboard_log="./logs/"
            )
            
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            raise
    
    async def _continuous_training(self):
        """Continuously train RL model"""
        while True:
            try:
                # Get training data
                train_data = await self._get_training_data()
                
                if len(train_data) < 1000:
                    await asyncio.sleep(3600)
                    continue
                
                # Start MLflow run
                with mlflow.start_run():
                    self.is_training = True
                    
                    # Train model
                    self.model.learn(
                        total_timesteps=100000,
                        callback=TrainingCallback(1000)
                    )
                    
                    # Evaluate performance
                    eval_reward = await self._evaluate_model()
                    
                    # Log metrics
                    mlflow.log_metric("eval_reward", eval_reward)
                    
                    # Save if better
                    if eval_reward > self.best_reward:
                        self.best_reward = eval_reward
                        self.model.save("models/best_model")
                        mlflow.log_artifact("models/best_model")
                    
                    self.is_training = False
                
                # Sleep between training rounds
                await asyncio.sleep(3600 * 4)  # Train every 4 hours
                
            except Exception as e:
                logger.error(f"Training error: {str(e)}")
                self.is_training = False
                await asyncio.sleep(3600)
    
    async def _process_market_data(self):
        """Process incoming market data"""
        async for message in self.kafka_consumer:
            try:
                data = message.value
                
                # Skip if training
                if self.is_training:
                    continue
                
                # Get RL prediction
                action = await self._get_rl_action(data)
                
                # Combine with other predictions
                prediction = await self._combine_predictions(
                    data,
                    action
                )
                
                # Publish result
                self.kafka_producer.send(
                    'rl_predictions',
                    prediction
                )
                
            except Exception as e:
                logger.error(f"Data processing error: {str(e)}")
    
    async def _get_rl_action(self, data: Dict) -> Dict:
        """Get action from RL model"""
        try:
            # Prepare observation
            obs = self.env._get_observation(data)
            
            # Get action
            action, _states = self.model.predict(
                obs,
                deterministic=True
            )
            
            return {
                'action': int(action),
                'action_prob': float(
                    self.model.policy.get_distribution(obs).probability(action)
                )
            }
            
        except Exception as e:
            logger.error(f"RL prediction error: {str(e)}")
            return {'action': 0, 'action_prob': 0.0}
    
    async def _combine_predictions(
        self,
        data: Dict,
        rl_action: Dict
    ) -> Dict:
        """Combine RL predictions with other signals"""
        try:
            # Get ML prediction
            ml_pred = self.market_predictor.predict(
                data['features']
            )
            
            # Calculate combined signal
            if rl_action['action_prob'] > 0.8:  # High confidence RL
                final_pred = {
                    'direction': 1 if rl_action['action'] == 1 else -1,
                    'probability': rl_action['action_prob'],
                    'source': 'rl'
                }
            elif ml_pred.probability > 0.8:  # High confidence ML
                final_pred = {
                    'direction': 1 if ml_pred.direction == "LONG" else -1,
                    'probability': ml_pred.probability,
                    'source': 'ml'
                }
            else:  # Combine signals
                combined_prob = (
                    rl_action['action_prob'] * 0.6 +
                    ml_pred.probability * 0.4
                )
                final_pred = {
                    'direction': 1 if rl_action['action'] == 1 else -1,
                    'probability': combined_prob,
                    'source': 'ensemble'
                }
            
            return {
                'symbol': data['symbol'],
                'timestamp': data['timestamp'],
                'prediction': final_pred,
                'rl_action': rl_action,
                'ml_prediction': ml_pred.__dict__
            }
            
        except Exception as e:
            logger.error(f"Prediction combination error: {str(e)}")
            return {}
    
    async def _evaluate_model(self) -> float:
        """Evaluate current model performance"""
        try:
            total_reward = 0
            episodes = 10
            
            for _ in range(episodes):
                obs = self.env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action, _states = self.model.predict(
                        obs,
                        deterministic=True
                    )
                    obs, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                
                total_reward += episode_reward
            
            return total_reward / episodes
            
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return -np.inf
    
    async def shutdown(self):
        """Shutdown the service"""
        try:
            # Save models
            if self.model:
                self.model.save("models/final_model")
            
            # Cleanup connections
            self.redis.close()
            self.kafka_producer.close()
            self.kafka_consumer.close()
            
            logger.info("RL service shut down successfully")
            
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")

if __name__ == "__main__":
    service = RLService(
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        kafka_broker=os.getenv("KAFKA_BROKER", "localhost:9092")
    )
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(service.start())
    except KeyboardInterrupt:
        loop.run_until_complete(service.shutdown())
    finally:
        loop.close()