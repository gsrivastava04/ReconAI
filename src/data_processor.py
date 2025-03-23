import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import joblib

from config import DATA_DIR, MODELS_DIR
from config import GL_IHUB_KEY_COLUMNS, GL_IHUB_CRITERIA_COLUMNS, GL_IHUB_DERIVED_COLUMNS
from config import CATALYST_IMPACT_KEY_COLUMNS, CATALYST_IMPACT_CRITERIA_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data preprocessing for reconciliation data including:
    - Data loading and cleaning
    - Feature engineering
    - Data transformation for model input
    """
    
    def __init__(self, recon_type: str = 'GL_IHUB'):
        """
        Initialize the data processor with specific reconciliation settings
        
        Args:
            recon_type (str): Type of reconciliation ('GL_IHUB' or 'CATALYST_IMPACT')
        """
        self.recon_type = recon_type
        
        # Set columns based on reconciliation type
        if recon_type == 'GL_IHUB':
            self.key_columns = GL_IHUB_KEY_COLUMNS
            self.criteria_columns = GL_IHUB_CRITERIA_COLUMNS
            self.derived_columns = GL_IHUB_DERIVED_COLUMNS
        elif recon_type == 'CATALYST_IMPACT':
            self.key_columns = CATALYST_IMPACT_KEY_COLUMNS
            self.criteria_columns = CATALYST_IMPACT_CRITERIA_COLUMNS
            self.derived_columns = ["Difference"] 
        else:
            raise ValueError(f"Unsupported reconciliation type: {recon_type}")
        
        self.scaler = None
        self.encoders = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file and perform basic cleaning
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info(f"Loading data from {file_path}")
        
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Basic cleaning
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(0)
        
        # Convert date columns to datetime
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                logger.warning(f"Could not convert column {col} to datetime")
        
        logger.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    
    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features based on the reconciliation type
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with added derived features
        """
        df_copy = df.copy()
        
        if self.recon_type == 'GL_IHUB':
            # Calculate balance difference
            if 'GL Balance' in df_copy.columns and 'IHub Balance' in df_copy.columns:
                df_copy['Balance Difference'] = df_copy['GL Balance'] - df_copy['IHub Balance']
                df_copy['Balance Difference Abs'] = df_copy['Balance Difference'].abs()
                df_copy['Balance Difference Percentage'] = df_copy['Balance Difference'].abs() / df_copy['GL Balance'].abs() * 100
                df_copy['Balance Difference Percentage'] = df_copy['Balance Difference Percentage'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
        elif self.recon_type == 'CATALYST_IMPACT':
            # For simplicity, we'll create dummy derived features
            # In a real scenario, you'd calculate actual differences between systems
            for col in self.criteria_columns:
                if f"Catalyst_{col}" in df_copy.columns and f"Impact_{col}" in df_copy.columns:
                    # For numeric columns
                    if df_copy[f"Catalyst_{col}"].dtype in [np.int64, np.float64] and df_copy[f"Impact_{col}"].dtype in [np.int64, np.float64]:
                        df_copy[f"{col}_Difference"] = df_copy[f"Catalyst_{col}"] - df_copy[f"Impact_{col}"]
                        df_copy[f"{col}_Difference_Abs"] = df_copy[f"{col}_Difference"].abs()
                    # For categorical columns
                    else:
                        df_copy[f"{col}_Match"] = (df_copy[f"Catalyst_{col}"] == df_copy[f"Impact_{col}"]).astype(int)
            
            # Create a consolidated difference flag
            df_copy["Has_Difference"] = 0
            for col in self.criteria_columns:
                if f"{col}_Difference" in df_copy.columns:
                    df_copy["Has_Difference"] = np.where(df_copy[f"{col}_Difference"] != 0, 1, df_copy["Has_Difference"])
                elif f"{col}_Match" in df_copy.columns:
                    df_copy["Has_Difference"] = np.where(df_copy[f"{col}_Match"] == 0, 1, df_copy["Has_Difference"])
        
        logger.info(f"Derived features calculated for {self.recon_type} reconciliation")
        return df_copy
    
    def add_historical_features(self, current_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add historical features to help with anomaly detection
        
        Args:
            current_df (pd.DataFrame): Current reconciliation data
            historical_df (pd.DataFrame): Historical reconciliation data
            
        Returns:
            pd.DataFrame: Current data with added historical features
        """
        result_df = current_df.copy()
        
        # Check if we have historical data
        if historical_df is None or historical_df.empty:
            logger.warning("No historical data provided for feature calculation")
            return result_df
        
        # Ensure date columns are present
        date_col = None
        for col in result_df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            logger.warning("No date column found for historical analysis")
            return result_df
        
        # Ensure both dataframes have the date column in datetime format
        for df in [result_df, historical_df]:
            if df[date_col].dtype != 'datetime64[ns]':
                df[date_col] = pd.to_datetime(df[date_col])
        
        # For each key identifier, calculate historical metrics
        groupby_columns = self.key_columns.copy()
        if date_col in groupby_columns:
            groupby_columns.remove(date_col)
        
        if not groupby_columns:
            logger.warning("No groupby columns available for historical analysis")
            return result_df
        
        # Calculate historical stats for relevant numeric columns
        if self.recon_type == 'GL_IHUB':
            numeric_cols = ['Balance Difference', 'GL Balance', 'IHub Balance']
            
            # Group by key columns to get historical metrics
            historical_stats = historical_df.groupby(groupby_columns).agg({
                'Balance Difference': ['mean', 'std', 'count', 'min', 'max'],
                'GL Balance': ['mean', 'std'],
                'IHub Balance': ['mean', 'std']
            })
            
            # Flatten the column hierarchy
            historical_stats.columns = ['_'.join(col).strip() for col in historical_stats.columns.values]
            historical_stats = historical_stats.reset_index()
            
            # Merge historical stats with current data
            result_df = pd.merge(result_df, historical_stats, on=groupby_columns, how='left')
            
            # Calculate z-scores for differences
            if 'Balance Difference_mean' in result_df.columns and 'Balance Difference_std' in result_df.columns:
                result_df['Balance_Difference_ZScore'] = (result_df['Balance Difference'] - result_df['Balance Difference_mean']) / result_df['Balance Difference_std']
                result_df['Balance_Difference_ZScore'] = result_df['Balance_Difference_ZScore'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
        elif self.recon_type == 'CATALYST_IMPACT':
            # For each criteria column with a calculated difference
            for col in self.criteria_columns:
                diff_col = f"{col}_Difference"
                if diff_col in result_df.columns:
                    if result_df[diff_col].dtype in [np.int64, np.float64]:
                        # Calculate historical stats
                        hist_stats = historical_df.groupby(groupby_columns)[diff_col].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
                        
                        # Rename columns to avoid collisions
                        hist_stats.columns = [col if i < len(groupby_columns) else f"{diff_col}_{col}" 
                                            for i, col in enumerate(hist_stats.columns)]
                        
                        # Merge with current data
                        result_df = pd.merge(result_df, hist_stats, on=groupby_columns, how='left')
                        
                        # Calculate z-score
                        result_df[f"{diff_col}_ZScore"] = (result_df[diff_col] - result_df[f"{diff_col}_mean"]) / result_df[f"{diff_col}_std"]
                        result_df[f"{diff_col}_ZScore"] = result_df[f"{diff_col}_ZScore"].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Fill missing values from the merge
        for col in result_df.columns:
            if '_mean' in col or '_std' in col or '_count' in col or '_min' in col or '_max' in col:
                result_df[col] = result_df[col].fillna(0)
        
        logger.info(f"Historical features added, resulting in {result_df.shape[1]} total columns")
        return result_df
    
    def prepare_features_for_model(self, df: pd.DataFrame, train: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for model training or inference
        
        Args:
            df (pd.DataFrame): Input dataframe
            train (bool): Whether this is for training (will fit transformers) or inference
            
        Returns:
            tuple: (np.ndarray of features, list of feature names)
        """
        # Select numeric features first
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove any key columns from features
        feature_cols = [col for col in numeric_cols if col not in self.key_columns and 'ID' not in col]
        
        # Remove date columns
        feature_cols = [col for col in feature_cols if 'date' not in col.lower() and 'time' not in col.lower()]
        
        # Also remove any target columns if they exist
        exclude_cols = ['Anomaly', 'Is_Anomaly', 'Anomaly_Type', 'Anomaly_Category']
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        # Create a DataFrame with just the features we want
        X = df[feature_cols].copy()
        
        # Import sklearn inside the function to avoid import errors if not installed
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        
        # Handle potential missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Scale the features
        if train:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_imputed)
            
            # Save the scaler for future use
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(self.scaler, os.path.join(MODELS_DIR, f"{self.recon_type}_scaler.pkl"))
        else:
            if self.scaler is None:
                scaler_path = os.path.join(MODELS_DIR, f"{self.recon_type}_scaler.pkl")
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                else:
                    logger.warning("No scaler found, using StandardScaler without fitting")
                    self.scaler = StandardScaler()
                    X_scaled = X_imputed  # Skip scaling if no scaler is available
            
            X_scaled = self.scaler.transform(X_imputed) if self.scaler else X_imputed
        
        logger.info(f"Prepared {X_scaled.shape[1]} features for modeling")
        return X_scaled, feature_cols
    
    def encode_categorical_variables(self, df: pd.DataFrame, cols: List[str], train: bool = False) -> pd.DataFrame:
        """
        Encode categorical variables for model consumption
        
        Args:
            df (pd.DataFrame): Input dataframe
            cols (List[str]): List of categorical columns to encode
            train (bool): Whether this is for training (will fit encoders) or inference
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        result_df = df.copy()
        
        from sklearn.preprocessing import OneHotEncoder
        
        for col in cols:
            if col in result_df.columns and result_df[col].dtype == 'object':
                if train:
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(result_df[[col]])
                    self.encoders[col] = encoder
                    
                    # Save the encoder
                    os.makedirs(MODELS_DIR, exist_ok=True)
                    joblib.dump(encoder, os.path.join(MODELS_DIR, f"{self.recon_type}_{col}_encoder.pkl"))
                else:
                    if col not in self.encoders:
                        encoder_path = os.path.join(MODELS_DIR, f"{self.recon_type}_{col}_encoder.pkl")
                        if os.path.exists(encoder_path):
                            self.encoders[col] = joblib.load(encoder_path)
                        else:
                            logger.warning(f"No encoder found for {col}, skipping encoding")
                            continue
                    
                    encoder = self.encoders[col]
                    encoded = encoder.transform(result_df[[col]])
                
                # Add encoded columns to dataframe
                encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                for i, enc_col in enumerate(encoded_cols):
                    result_df[enc_col] = encoded[:, i]
                
                # Drop original column
                result_df = result_df.drop(col, axis=1)
        
        return result_df
    
    def process_data_for_training(self, current_data_path: str, historical_data_path: str = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Process data for model training
        
        Args:
            current_data_path (str): Path to current data
            historical_data_path (str): Path to historical data (optional)
            
        Returns:
            tuple: (processed dataframe, feature matrix, feature names)
        """
        # Load current data
        df = self.load_data(current_data_path)
        
        # Calculate derived features
        df = self.calculate_derived_features(df)
        
        # Add historical features if available
        if historical_data_path:
            historical_df = self.load_data(historical_data_path)
            historical_df = self.calculate_derived_features(historical_df)
            df = self.add_historical_features(df, historical_df)
        
        # Prepare features for model
        X, feature_names = self.prepare_features_for_model(df, train=True)
        
        return df, X, feature_names
    
    def process_data_for_inference(self, current_data_path: str, historical_data_path: str = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Process data for model inference
        
        Args:
            current_data_path (str): Path to current data
            historical_data_path (str): Path to historical data (optional)
            
        Returns:
            tuple: (processed dataframe, feature matrix, feature names)
        """
        # Load current data
        df = self.load_data(current_data_path)
        
        # Calculate derived features
        df = self.calculate_derived_features(df)
        
        # Add historical features if available
        if historical_data_path:
            historical_df = self.load_data(historical_data_path)
            historical_df = self.calculate_derived_features(historical_df)
            df = self.add_historical_features(df, historical_df)
        
        # Prepare features for model
        X, feature_names = self.prepare_features_for_model(df, train=False)
        
        return df, X, feature_names