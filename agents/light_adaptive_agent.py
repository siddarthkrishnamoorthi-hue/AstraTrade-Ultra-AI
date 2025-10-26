"""
Lightweight Adaptive Trading Agent with minimal resource usage
"""
from typing import Dict, Optional
import numpy as np
from datetime import datetime
import logging
from collections import deque
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class LightAdaptiveAgent:
    """Lightweight trading agent with minimal memory footprint"""
    
    def __init__(self, model_path: str = "models"):
        """Initialize with minimal components"""
        self.model_path = model_path
        self.classifier = RandomForestClassifier(
            n_estimators=50,  # Reduced from 100
            min_samples_leaf=10,  # Increased for lighter model
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        
        # Use deque for fixed-size history (prevents memory growth)
        self.performance_history = deque(maxlen=100)
        self.state_history = deque(maxlen=50)
        
        # Lightweight parameter storage
        self.params = {
            'risk_adjustment': 1.0,
            'confidence_threshold': 0.7
        }
        
    def adapt_to_market(self, state: Dict) -> Dict[str, float]:
        """Lightweight market adaptation"""
        # Extract essential features only
        features = np.array([
            state.get('volatility', 0),
            state.get('trend', 0),
            state.get('volume', 0)
        ]).reshape(1, -1)
        
        # Scale features
        if not self.scaler.n_samples_seen_:
            self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        
        # Make prediction
        if hasattr(self.classifier, 'classes_'):
            prediction = self.classifier.predict_proba(scaled_features)[0]
            self.params['risk_adjustment'] = float(prediction[0])
            
        self.state_history.append(state)
        return self.params
        
    def update_performance(self, profit: float) -> None:
        """Update performance metrics"""
        self.performance_history.append(profit)
        
        # Retrain only if enough samples and performance is poor
        if (len(self.performance_history) >= 100 and 
            np.mean(list(self.performance_history)) < 0):
            self._light_retrain()
            
    def _light_retrain(self) -> None:
        """Lightweight retraining process"""
        if len(self.state_history) < 50:
            return
            
        X = []
        y = []
        
        # Use only recent history for training
        for state, profit in zip(
            list(self.state_history),
            list(self.performance_history)[-50:]
        ):
            try:
                X.append([
                    state.get('volatility', 0),
                    state.get('trend', 0),
                    state.get('volume', 0)
                ])
                y.append(1 if profit > 0 else 0)
            except (KeyError, TypeError):
                continue
                
        if not X or not y:
            return
            
        X = np.array(X)
        y = np.array(y)
        
        if len(np.unique(y)) < 2:
            return
            
        # Retrain with minimal data
        self.classifier.fit(X, y)
        
    def save_state(self) -> None:
        """Save minimal state"""
        try:
            joblib.dump(
                {
                    'classifier': self.classifier,
                    'scaler': self.scaler,
                    'params': self.params
                },
                f"{self.model_path}/light_agent_state.joblib"
            )
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")
            
    def load_state(self) -> None:
        """Load minimal state"""
        try:
            state = joblib.load(f"{self.model_path}/light_agent_state.joblib")
            self.classifier = state['classifier']
            self.scaler = state['scaler']
            self.params = state['params']
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.state_history.clear()
        self.performance_history.clear()