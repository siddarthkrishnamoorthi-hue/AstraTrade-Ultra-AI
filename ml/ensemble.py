"""
LSTM + XGBoost ensemble for market prediction with high accuracy target
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Tuple, List, Dict, Optional
import joblib
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PredictionResult:
    probability: float
    direction: str
    confidence: float
    features_importance: Dict[str, float]
    ensemble_agreement: float

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.X) - self.seq_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.X[idx:idx + self.seq_length],
            self.y[idx + self.seq_length]
        )

class LSTMPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Use only the last output
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

class MarketPredictor:
    def __init__(
        self,
        sequence_length: int = 60,
        batch_size: int = 32,
        hidden_size: int = 128,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        model_path: Optional[str] = None
    ):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_path = model_path or "models"
        
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        
        # Ensure model directory exists
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training both LSTM and XGBoost models
        """
        # Scale features
        X = self.scaler.fit_transform(df[feature_columns])
        y = (df[target_column] > 0).astype(int)  # Binary classification
        
        # Create sequences for LSTM
        X_seq = np.array([
            X[i:i + self.sequence_length]
            for i in range(len(X) - self.sequence_length)
        ])
        y_seq = y[self.sequence_length:]
        
        # Split maintaining temporal order
        train_size = int(len(X_seq) * 0.8)
        
        X_train_lstm = X_seq[:train_size]
        X_test_lstm = X_seq[train_size:]
        y_train = y_seq[:train_size]
        y_test = y_seq[train_size:]
        
        # Prepare data for XGBoost (using last sequence for each prediction)
        X_train_xgb = X_train_lstm[:, -1, :]
        X_test_xgb = X_test_lstm[:, -1, :]
        
        return (X_train_lstm, X_train_xgb), (X_test_lstm, X_test_xgb), y_train, y_test

    def train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """
        Train LSTM model with early stopping
        """
        input_size = X_train.shape[2]
        self.lstm_model = LSTMPredictor(input_size, self.hidden_size)
        
        train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.lstm_model.parameters(),
            lr=self.learning_rate
        )
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            self.lstm_model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            self.lstm_model.eval()
            with torch.no_grad():
                val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
                val_loss = 0
                
                for batch_X, batch_y in val_loader:
                    outputs = self.lstm_model(batch_X)
                    val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(
                    self.lstm_model.state_dict(),
                    f"{self.model_path}/lstm_model.pth"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """
        Train XGBoost model with optimal parameters
        """
        self.xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
        )
        
        # Train with early stopping
        self.xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Save model
        joblib.dump(self.xgb_model, f"{self.model_path}/xgb_model.joblib")

    def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str]
    ) -> Dict[str, float]:
        """
        Train both models and return performance metrics
        """
        # Prepare data
        (X_train_lstm, X_train_xgb), (X_test_lstm, X_test_xgb), y_train, y_test = \
            self.prepare_data(df, target_column, feature_columns)
        
        # Train LSTM
        self.train_lstm(X_train_lstm, y_train, X_test_lstm, y_test)
        
        # Train XGBoost
        self.train_xgboost(X_train_xgb, y_train, X_test_xgb, y_test)
        
        # Evaluate ensemble
        metrics = self.evaluate(X_test_lstm, X_test_xgb, y_test)
        return metrics

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """
        Generate ensemble prediction with confidence scores
        """
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # LSTM prediction
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_seq = torch.FloatTensor(scaled_features[-self.sequence_length:])
            lstm_seq = lstm_seq.unsqueeze(0)  # Add batch dimension
            lstm_pred = self.lstm_model(lstm_seq).item()
        
        # XGBoost prediction
        xgb_features = scaled_features[-1].reshape(1, -1)
        xgb_pred = self.xgb_model.predict_proba(xgb_features)[0][1]
        
        # Ensemble prediction
        ensemble_prob = (lstm_pred + xgb_pred) / 2
        ensemble_agreement = 1 - abs(lstm_pred - xgb_pred)
        
        # Get feature importance from XGBoost
        feature_importance = dict(zip(
            features.columns,
            self.xgb_model.feature_importances_
        ))
        
        return PredictionResult(
            probability=ensemble_prob,
            direction="LONG" if ensemble_prob > 0.5 else "SHORT",
            confidence=ensemble_agreement,
            features_importance=feature_importance,
            ensemble_agreement=ensemble_agreement
        )

    def evaluate(
        self,
        X_test_lstm: np.ndarray,
        X_test_xgb: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the ensemble model performance
        """
        # LSTM predictions
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_preds = self.lstm_model(torch.FloatTensor(X_test_lstm))
            lstm_preds = (lstm_preds.numpy() > 0.5).astype(int)
        
        # XGBoost predictions
        xgb_preds = self.xgb_model.predict(X_test_xgb)
        
        # Ensemble predictions (majority voting)
        ensemble_preds = ((lstm_preds + xgb_preds.reshape(-1, 1)) > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, ensemble_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            ensemble_preds,
            average='binary'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def save_models(self) -> None:
        """
        Save both models and scaler
        """
        if self.lstm_model:
            torch.save(
                self.lstm_model.state_dict(),
                f"{self.model_path}/lstm_model.pth"
            )
        if self.xgb_model:
            joblib.dump(
                self.xgb_model,
                f"{self.model_path}/xgb_model.joblib"
            )
        joblib.dump(self.scaler, f"{self.model_path}/scaler.joblib")

    def load_models(self) -> None:
        """
        Load both models and scaler
        """
        lstm_path = f"{self.model_path}/lstm_model.pth"
        xgb_path = f"{self.model_path}/xgb_model.joblib"
        scaler_path = f"{self.model_path}/scaler.joblib"
        
        if Path(lstm_path).exists():
            self.lstm_model.load_state_dict(torch.load(lstm_path))
        if Path(xgb_path).exists():
            self.xgb_model = joblib.load(xgb_path)
        if Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)