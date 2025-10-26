"""
Machine Learning service for market predictions
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import shap
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

from ml.ensemble import MarketPredictor
from ml.adaptive_learner import AdaptiveLearner
from ml.uncertainty import BayesianDropout

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MLService")

class MLService:
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
        
        # Initialize ML components
        self.market_predictor = MarketPredictor()
        self.adaptive_learner = AdaptiveLearner()
        self.bayesian_dropout = BayesianDropout()
        
        # SHAP explainer
        self.explainer = None
        
        # Feature preprocessing
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.prediction_history: List[Dict] = []
        self.accuracy_metrics: Dict[str, float] = {}
    
    async def start(self):
        """Start the ML service"""
        try:
            # Initialize models
            await self._initialize_models()
            
            # Start background tasks
            asyncio.create_task(self._process_market_data())
            asyncio.create_task(self._update_models())
            
            logger.info("ML service started successfully")
            
            # Keep service running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Service error: {str(e)}")
            await self.shutdown()
    
    async def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Load cached models
            self.market_predictor.load_models()
            
            # Initialize SHAP explainer
            self.explainer = shap.DeepExplainer(
                self.market_predictor.lstm_model,
                self.market_predictor.feature_data[:100]
            )
            
            # Warm up models
            await self._warm_up_models()
            
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            raise
    
    async def _process_market_data(self):
        """Process incoming market data"""
        async for message in self.kafka_consumer:
            try:
                data = message.value
                
                # Preprocess features
                features = self._preprocess_features(data['features'])
                
                # Get prediction with uncertainty
                prediction = await self._get_prediction_with_uncertainty(
                    features
                )
                
                # Get explanation
                explanation = await self._explain_prediction(
                    features,
                    prediction
                )
                
                # Combine results
                result = {
                    'symbol': data['symbol'],
                    'timestamp': data['timestamp'],
                    'prediction': prediction,
                    'explanation': explanation
                }
                
                # Publish result
                self.kafka_producer.send(
                    'ml_predictions',
                    result
                )
                
                # Update Redis cache
                self.redis.setex(
                    f"prediction:{data['symbol']}",
                    300,  # 5 minute expiry
                    json.dumps(result)
                )
                
                # Track performance
                self._update_performance_metrics(result)
                
            except Exception as e:
                logger.error(f"Data processing error: {str(e)}")
    
    async def _get_prediction_with_uncertainty(
        self,
        features: np.ndarray
    ) -> Dict:
        """Get prediction with uncertainty estimation"""
        try:
            # Get base prediction
            base_pred = self.market_predictor.predict(features)
            
            # Monte Carlo Dropout predictions
            mc_preds = []
            for _ in range(20):
                pred = self.bayesian_dropout(features)
                mc_preds.append(pred)
            
            # Calculate uncertainty
            uncertainty = np.std(mc_preds)
            
            # Adaptive threshold
            threshold = self.adaptive_learner.get_threshold(
                base_pred.probability,
                uncertainty
            )
            
            return {
                'direction': base_pred.direction,
                'probability': float(base_pred.probability),
                'uncertainty': float(uncertainty),
                'threshold': float(threshold),
                'confidence': float(base_pred.probability / (1 + uncertainty))
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    async def _explain_prediction(
        self,
        features: np.ndarray,
        prediction: Dict
    ) -> Dict:
        """Generate prediction explanation"""
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(
                features.reshape(1, -1)
            )
            
            # Get LIME explanation
            lime_exp = self.lime_explainer.explain_instance(
                features,
                self.market_predictor.predict_proba
            )
            
            # Extract feature importance
            feature_importance = {
                name: float(importance)
                for name, importance in zip(
                    self.market_predictor.feature_names,
                    shap_values[0]
                )
            }
            
            return {
                'feature_importance': feature_importance,
                'shap_values': shap_values[0].tolist(),
                'lime_explanation': lime_exp.as_list()
            }
            
        except Exception as e:
            logger.error(f"Explanation error: {str(e)}")
            return {}
    
    async def _update_models(self):
        """Periodically update ML models"""
        while True:
            try:
                # Get recent predictions
                recent_preds = self.prediction_history[-1000:]
                
                if len(recent_preds) > 100:
                    # Calculate performance metrics
                    accuracy = self._calculate_accuracy(recent_preds)
                    
                    # Update adaptive learner
                    self.adaptive_learner.update(
                        accuracy,
                        self.accuracy_metrics
                    )
                    
                    # Retrain if needed
                    if accuracy < 0.6:  # Below threshold
                        await self._retrain_models()
                
                await asyncio.sleep(3600)  # Update hourly
                
            except Exception as e:
                logger.error(f"Model update error: {str(e)}")
                await asyncio.sleep(3600)
    
    def _calculate_accuracy(
        self,
        predictions: List[Dict]
    ) -> float:
        """Calculate prediction accuracy"""
        try:
            correct = 0
            total = 0
            
            for pred in predictions:
                if 'actual' in pred:
                    total += 1
                    if pred['actual'] == pred['prediction']['direction']:
                        correct += 1
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Accuracy calculation error: {str(e)}")
            return 0.0
    
    async def _retrain_models(self):
        """Retrain ML models"""
        try:
            logger.info("Starting model retraining")
            
            # Get training data
            train_data = await self._get_training_data()
            
            # Retrain models
            self.market_predictor.train(
                train_data['features'],
                train_data['labels'],
                continue_training=True
            )
            
            # Update explainers
            self.explainer = shap.DeepExplainer(
                self.market_predictor.lstm_model,
                train_data['features'][:100]
            )
            
            # Save models
            self.market_predictor.save_models()
            
            logger.info("Model retraining completed")
            
        except Exception as e:
            logger.error(f"Retraining error: {str(e)}")
    
    async def shutdown(self):
        """Shutdown the service"""
        try:
            # Save models
            self.market_predictor.save_models()
            
            # Cleanup connections
            self.redis.close()
            self.kafka_producer.close()
            self.kafka_consumer.close()
            
            logger.info("ML service shut down successfully")
            
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")

if __name__ == "__main__":
    service = MLService(
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