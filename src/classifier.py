import pandas as pd
import numpy as np
import os
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
import openai

from config import MODELS_DIR, ANOMALY_CATEGORIES, OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyClassifier:
    """
    Classifies detected anomalies into predefined categories
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the anomaly classifier
        
        Args:
            use_llm (bool): Whether to use an LLM for classification
        """
        self.use_llm = use_llm
        self.model = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """
        Fit the classifier model
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels (anomaly categories)
            feature_names (List[str]): Names of features
        """
        if not self.use_llm:
            self.feature_names = feature_names
            logger.info("Training Random Forest classifier for anomaly classification")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
            # Save the model
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(self.model, os.path.join(MODELS_DIR, "anomaly_classifier.pkl"))
            
            logger.info("Anomaly classification model trained successfully")
        else:
            logger.info("Using LLM for classification, no model training needed")
    
    def predict(self, X: np.ndarray, df: pd.DataFrame, anomaly_indices: List[int]) -> Dict[int, str]:
        """
        Predict the category of each anomaly
        
        Args:
            X (np.ndarray): Feature matrix
            df (pd.DataFrame): Original dataframe with all columns
            anomaly_indices (List[int]): Indices of anomalies
            
        Returns:
            Dict[int, str]: Dictionary mapping anomaly indices to categories
        """
        if len(anomaly_indices) == 0:
            return {}
        
        predictions = {}
        
        if self.use_llm:
            # Use LLM for classification
            if not OPENAI_API_KEY:
                logger.warning("OpenAI API key not found, falling back to rule-based classification")
                return self.rule_based_classification(X, df, anomaly_indices)
            
            try:
                # Set OpenAI API key
                openai.api_key = OPENAI_API_KEY
                
                # Process anomalies in batches to avoid API rate limits
                batch_size = 5
                for i in range(0, len(anomaly_indices), batch_size):
                    batch_indices = anomaly_indices[i:i+batch_size]
                    
                    for idx in batch_indices:
                        # Get the anomalous record
                        record = df.iloc[idx].to_dict()
                        
                        # Filter to relevant columns for classification
                        relevant_cols = [col for col in df.columns if 'difference' in col.lower() 
                                        or 'balance' in col.lower()
                                        or 'match' in col.lower()
                                        or 'zscore' in col.lower()]
                        
                        relevant_data = {k: v for k, v in record.items() if k in relevant_cols}
                        
                        # Create prompt for LLM
                        prompt = f"""
                        You are a financial reconciliation expert. Analyze this reconciliation record with anomalous values and classify it into one of these categories:
                        {', '.join(ANOMALY_CATEGORIES)}
                        
                        Record data: {relevant_data}
                        
                        Respond with only the category name from the list above. If none apply, respond with 'Other'.
                        """
                        
                        # Call OpenAI API
                        response = openai.Completion.create(
                            engine="text-davinci-003",
                            prompt=prompt,
                            max_tokens=50,
                            temperature=0.3,
                            top_p=1.0
                        )
                        
                        # Extract category from response
                        predicted_category = response.choices[0].text.strip()
                        
                        # Validate that the category is in our list
                        if predicted_category not in ANOMALY_CATEGORIES:
                            predicted_category = "Other"
                        
                        predictions[idx] = predicted_category
                
                logger.info(f"Classified {len(predictions)} anomalies using LLM")
            except Exception as e:
                logger.error(f"Error using LLM for classification: {str(e)}")
                # Fall back to rule-based classification
                return self.rule_based_classification(X, df, anomaly_indices)
        else:
            # Use trained ML model for classification
            if self.model is None:
                model_path = os.path.join(MODELS_DIR, "anomaly_classifier.pkl")
                if os.path.exists(model_path):
                    self.model = joblib.load(model_path)
                else:
                    logger.warning("No classifier model found, falling back to rule-based classification")
                    return self.rule_based_classification(X, df, anomaly_indices)
            
            # Get anomaly data
            X_anomalies = X[anomaly_indices]
            
            # Predict categories
            y_pred = self.model.predict(X_anomalies)
            
            # Map predictions to indices
            for i, idx in enumerate(anomaly_indices):
                predictions[idx] = ANOMALY_CATEGORIES[y_pred[i]] if y_pred[i] < len(ANOMALY_CATEGORIES) else "Other"
            
            logger.info(f"Classified {len(predictions)} anomalies using ML model")
        
        return predictions
    
    def rule_based_classification(self, X: np.ndarray, df: pd.DataFrame, anomaly_indices: List[int]) -> Dict[int, str]:
        """
        Classify anomalies using simple rule-based approach
        
        Args:
            X (np.ndarray): Feature matrix
            df (pd.DataFrame): Original dataframe with all columns
            anomaly_indices (List[int]): Indices of anomalies
            
        Returns:
            Dict[int, str]: Dictionary mapping anomaly indices to categories
        """
        predictions = {}
        
        for idx in anomaly_indices:
            record = df.iloc[idx]
            
            # Check for missing data
            missing_values = record.isna().sum()
            if missing_values > 0:
                predictions[idx] = "Missing Data"
                continue
            
            # Check for large balance differences
            if 'Balance Difference' in record and abs(record['Balance Difference']) > 1000:
                predictions[idx] = "Calculation Error"
                continue
                
            # Check for timing differences
            if 'As of Date' in record and record['As of Date'].dayofweek >= 5:  # Weekend
                predictions[idx] = "Timing Difference"
                continue
                
            # Check for currency issues
            if 'Currency' in record and record['Currency'] != 'USD':
                predictions[idx] = "Currency Conversion Issue"
                continue
                
            # Default category
            predictions[idx] = "System Processing Error"
        
        logger.info(f"Classified {len(predictions)} anomalies using rule-based approach")
        return predictions