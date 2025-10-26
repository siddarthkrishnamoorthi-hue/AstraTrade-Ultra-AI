"""
Unit tests for ML Ensemble components
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
import tensorflow as tf
from sklearn.metrics import accuracy_score
import xgboost as xgb

from ml.ensemble import MLEnsemble
from ml.continuous_learning import ContinuousLearner

@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    X = np.random.random((1000, 10))
    y = (X.sum(axis=1) > 5).astype(int)
    return X, y

@pytest.fixture
def ensemble():
    """Create ML ensemble instance"""
    return MLEnsemble(
        input_dim=10,
        lstm_units=64,
        xgb_max_depth=5
    )

class TestMLEnsemble:
    """Test suite for ML Ensemble"""

    def test_model_initialization(self, ensemble):
        """Test proper model initialization"""
        assert isinstance(ensemble.lstm_model, tf.keras.Model)
        assert isinstance(ensemble.xgb_model, xgb.XGBClassifier)

    def test_training(self, ensemble, sample_data):
        """Test model training"""
        X, y = sample_data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        # Train models
        history = ensemble.train(
            X_train, y_train,
            X_test, y_test,
            epochs=5
        )
        
        assert isinstance(history, dict)
        assert 'lstm_loss' in history
        assert 'xgb_loss' in history
        
        # Verify performance
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.6  # Should achieve better than random

    @pytest.mark.benchmark
    def test_prediction_speed(self, ensemble, sample_data, benchmark):
        """Test prediction performance"""
        X, _ = sample_data
        
        def predict_batch():
            return ensemble.predict(X[:32])  # Test batch prediction
        
        result = benchmark(predict_batch)
        assert result.stats.mean < 0.05  # Should complete within 50ms

    def test_ensemble_weighting(self, ensemble, sample_data):
        """Test ensemble prediction weighting"""
        X, y = sample_data
        
        # Test different weight combinations
        weights = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]
        predictions = []
        
        for lstm_weight, xgb_weight in weights:
            ensemble.set_weights(lstm_weight, xgb_weight)
            pred = ensemble.predict(X)
            predictions.append(pred)
        
        # Verify different weights produce different results
        assert not np.array_equal(predictions[0], predictions[1])
        assert not np.array_equal(predictions[1], predictions[2])

    def test_model_persistence(self, ensemble, sample_data, tmp_path):
        """Test model saving and loading"""
        X, y = sample_data
        
        # Train and save models
        ensemble.train(X, y, X, y, epochs=1)
        save_path = tmp_path / "models"
        save_path.mkdir()
        ensemble.save_models(str(save_path))
        
        # Load models in new ensemble
        new_ensemble = MLEnsemble(input_dim=10)
        new_ensemble.load_models(str(save_path))
        
        # Verify predictions match
        assert np.array_equal(
            ensemble.predict(X),
            new_ensemble.predict(X)
        )

class TestContinuousLearning:
    """Test suite for Continuous Learning"""

    @pytest.fixture
    def learner(self):
        """Create continuous learner instance"""
        return ContinuousLearner()

    def test_incremental_learning(self, learner, sample_data):
        """Test incremental learning capabilities"""
        X, y = sample_data
        
        # Initial training
        learner.train_increment(X[:500], y[:500])
        initial_pred = learner.predict(X[500:600])
        
        # Incremental update
        learner.train_increment(X[500:600], y[500:600])
        updated_pred = learner.predict(X[500:600])
        
        # Verify model adapts
        assert not np.array_equal(initial_pred, updated_pred)

    def test_concept_drift_detection(self, learner, sample_data):
        """Test concept drift detection"""
        X, y = sample_data
        
        # Simulate concept drift
        drift_X = X.copy()
        drift_X[:, 0] += 2.0  # Shift first feature
        
        drift_detected = learner.detect_concept_drift(drift_X)
        assert drift_detected

    @pytest.mark.parametrize("window_size", [50, 100, 200])
    def test_window_size_impact(self, learner, sample_data, window_size):
        """Test impact of different window sizes"""
        X, y = sample_data
        
        learner.set_window_size(window_size)
        learner.train_increment(X[:window_size], y[:window_size])
        
        # Verify window mechanics
        assert len(learner.get_window_data()[0]) == window_size

    def test_error_handling(self, learner):
        """Test error handling for invalid inputs"""
        with pytest.raises(ValueError):
            learner.train_increment(
                np.random.random((100, 5)),  # Wrong input dimension
                np.random.randint(0, 2, 100)
            )
            
    @pytest.mark.benchmark
    def test_adaptation_speed(self, learner, sample_data, benchmark):
        """Test adaptation performance"""
        X, y = sample_data
        
        def adaptation_iteration():
            learner.train_increment(X[:100], y[:100])
            return learner.predict(X[100:110])
        
        result = benchmark(adaptation_iteration)
        assert result.stats.mean < 0.1  # Should complete within 100ms