import numpy as np
import pandas as pd
import os
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from config import MODELS_DIR, ANOMALY_THRESHOLD

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Detects anomalies in reconciliation data using various ML algorithms
    """
    
    def __init__(self, method: str = 'isolation_forest'):
        """
        Initialize the anomaly detector
        
        Args:
            method (str): Detection method to use ('isolation_forest', 'lof', 'dbscan', 'zscore', 'ensemble')
        """
        self.method = method
        self.model = None
        self.feature_names = None
        self.pca = None  # For dimensionality reduction in visualization
        
    def fit(self, X: np.ndarray, feature_names: List[str]):
        """
        Fit the anomaly detection model
        
        Args:
            X (np.ndarray): Feature matrix
            feature_names (List[str]): Names of features
        """
        self.feature_names = feature_names
        
        if self.method == 'isolation_forest':
            logger.info("Training Isolation Forest model")
            self.model = IsolationForest(contamination=ANOMALY_THRESHOLD, random_state=42)
            self.model.fit(X)
            
        elif self.method == 'lof':
            logger.info("Training Local Outlier Factor model")
            self.model = LocalOutlierFactor(n_neighbors=20, contamination=ANOMALY_THRESHOLD, novelty=True)
            self.model.fit(X)
            
        elif self.method == 'dbscan':
            logger.info("Training DBSCAN model")
            self.model = DBSCAN(eps=0.5, min_samples=5)
            self.model.fit(X)
            
        elif self.method == 'zscore':
            logger.info("Using Z-Score method (no training required)")
            # Z-score method doesn't require fitting a model
            pass
            
        elif self.method == 'ensemble':
            logger.info("Training ensemble of anomaly detection models")
            # Create multiple models for ensemble
            self.models = {
                'isolation_forest': IsolationForest(contamination=ANOMALY_THRESHOLD, random_state=42),
                'lof': LocalOutlierFactor(n_neighbors=20, contamination=ANOMALY_THRESHOLD, novelty=True)
            }
            
            # Fit each model
            for name, model in self.models.items():
                model.fit(X)
        else:
            raise ValueError(f"Unsupported anomaly detection method: {self.method}")
        
        # Fit PCA for visualization
        self.pca = PCA(n_components=2)
        self.pca.fit(X)
        
        # Save the model
        self.save_model()
        
        logger.info(f"Anomaly detection model trained successfully using {self.method}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in the data
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Binary array indicating anomalies (1: normal, -1: anomaly)
        """
        if self.model is None and self.method != 'zscore' and self.method != 'ensemble':
            self.load_model()
            
            if self.model is None:
                raise ValueError("No model available for prediction. Train or load a model first.")
        
        if self.method == 'isolation_forest':
            predictions = self.model.predict(X)
            
        elif self.method == 'lof':
            predictions = self.model.predict(X)
            
        elif self.method == 'dbscan':
            cluster_labels = self.model.fit_predict(X)
            # In DBSCAN, -1 indicates noise points (potential anomalies)
            predictions = np.where(cluster_labels == -1, -1, 1)
            
        elif self.method == 'zscore':
            # Calculate Z-scores for each feature
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            # Mark as anomaly if any feature has z-score > 3
            predictions = np.where(np.any(z_scores > 3, axis=1), -1, 1)
            
        elif self.method == 'ensemble':
            # Get predictions from each model
            pred_if = self.models['isolation_forest'].predict(X)
            pred_lof = self.models['lof'].predict(X)
            
            # Combine predictions (majority vote)
            stacked = np.vstack([pred_if, pred_lof])
            predictions = np.apply_along_axis(lambda x: -1 if np.sum(x == -1) >= len(self.models)/2 else 1, 0, stacked)
            
        else:
            raise ValueError(f"Unsupported anomaly detection method: {self.method}")
        
        logger.info(f"Predicted {np.sum(predictions == -1)} anomalies out of {len(predictions)} records")
        return predictions
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for the data (higher score = more anomalous)
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Anomaly scores
        """
        if self.model is None and self.method != 'zscore' and self.method != 'ensemble':
            self.load_model()
            
            if self.model is None:
                raise ValueError("No model available for prediction. Train or load a model first.")
        
        if self.method == 'isolation_forest':
            # Negating because decision_function returns lower values for anomalies
            scores = -self.model.decision_function(X)
            
        elif self.method == 'lof':
            # Negating because decision_function returns lower values for anomalies
            scores = -self.model.decision_function(X)
            
        elif self.method == 'dbscan':
            cluster_labels = self.model.fit_predict(X)
            # In DBSCAN we don't get scores directly, so we'll use binary classification
            scores = np.where(cluster_labels == -1, 1.0, 0.0)
            
        elif self.method == 'zscore':
            # Calculate Z-scores for each feature and take the maximum
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            scores = np.max(z_scores, axis=1)
            
        elif self.method == 'ensemble':
            # Get scores from each model
            scores_if = -self.models['isolation_forest'].decision_function(X)
            scores_lof = -self.models['lof'].decision_function(X)
            
            # Average the scores
            scores = np.mean(np.vstack([scores_if, scores_lof]), axis=0)
            
        else:
            raise ValueError(f"Unsupported anomaly detection method: {self.method}")
        
        # Normalize scores to [0, 1] range
        if scores.min() != scores.max():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores
    
    def get_anomaly_insights(self, X: np.ndarray, df: pd.DataFrame, anomaly_indices: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Get insights into why records were flagged as anomalies
        
        Args:
            X (np.ndarray): Feature matrix
            df (pd.DataFrame): Original dataframe with all columns
            anomaly_indices (List[int]): Indices of anomalies
            
        Returns:
            Dict[int, Dict[str, Any]]: Dictionary of insights for each anomaly
        """
        if len(anomaly_indices) == 0:
            return {}
        
        insights = {}
        
        # Calculate mean and std of all features
        feature_means = np.mean(X, axis=0)
        feature_stds = np