"""
Main Preprocessor Module
Coordinates data loading, feature engineering, and encoding workflow
"""

import numpy as np
import gc
import pickle
from sklearn.model_selection import train_test_split

from ..data.loader import DataLoader
from .time_features import extract_time_features
from .encoders import FeatureEncoder
from ..utils.helpers import optimize_memory


class FraudPreprocessor:
    """Fraud Detection Data Preprocessor"""
    
    def __init__(self, config, verbose=True):
        """
        Args:
            config: Configuration object
            verbose: Whether to print detailed information
        """
        self.config = config
        self.verbose = verbose
        
        self.data_loader = DataLoader(verbose=verbose)
        self.encoder = FeatureEncoder(
            rare_category_threshold=config.RARE_CATEGORY_THRESHOLD,
            verbose=verbose
        )
        
        self.drop_columns = []
    
    def _log(self, message):
        """Print log message"""
        if self.verbose:
            print(message)
    
    def _remove_high_missing_columns(self, df, target='isFraud'):
        """Remove columns with high missing rate"""
        self._log("\n" + "=" * 50)
        self._log("Removing High Missing Columns")
        self._log("=" * 50)
        
        # Calculate missing rate
        missing = df.isnull().mean()
        
        # Find columns to drop
        self.drop_columns = list(missing[missing > self.config.MISSING_THRESHOLD].index)
        
        # Ensure target column is not dropped
        if target in self.drop_columns:
            self.drop_columns.remove(target)
        
        self._log(f"Dropping {len(self.drop_columns)} columns (missing rate > {self.config.MISSING_THRESHOLD*100}%)")
        
        if self.drop_columns:
            df = df.drop(columns=self.drop_columns)
            self._log(f"Shape after dropping: {df.shape}")
        
        return df
    
    def _split_data(self, df, target='isFraud', test_size=0.2, valid_size=0.125):
        """Split dataset"""
        self._log("\n" + "=" * 50)
        self._log("Splitting Dataset")
        self._log("=" * 50)
        
        X = df[self.encoder.feature_columns].values.astype(np.float32)
        y = df[target].values.astype(np.int64)
        
        if self.config.USE_TIME_SPLIT and 'TransactionDT' in df.columns:
            self._log("Using time-based split (recommended to avoid data leakage)")
            
            # Sort by time
            time_order = df['TransactionDT'].values.argsort()
            X = X[time_order]
            y = y[time_order]
            
            # Split by time order
            n = len(y)
            train_end = int(n * (1 - test_size))
            valid_start = int(train_end * (1 - valid_size))
            
            X_train = X[:valid_start]
            y_train = y[:valid_start]
            X_valid = X[valid_start:train_end]
            y_valid = y[valid_start:train_end]
            X_test = X[train_end:]
            y_test = y[train_end:]
        else:
            self._log("Using random split (stratified sampling)")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=valid_size, random_state=42, stratify=y_train
            )
        
        self._log(f"Train set: {X_train.shape} (fraud rate: {y_train.mean()*100:.2f}%)")
        self._log(f"Valid set: {X_valid.shape} (fraud rate: {y_valid.mean()*100:.2f}%)")
        self._log(f"Test set: {X_test.shape} (fraud rate: {y_test.mean()*100:.2f}%)")
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    def fit_transform(self, transaction_path=None, identity_path=None, target='isFraud'):
        """
        Complete preprocessing workflow (training set)
        
        Args:
            transaction_path: Path to transaction data (defaults to config path)
            identity_path: Path to identity data (defaults to config path)
            target: Target column name
            
        Returns:
            dict: Contains train/valid/test data and metadata
        """
        transaction_path = transaction_path or self.config.TRAIN_TRANSACTION
        identity_path = identity_path or self.config.TRAIN_IDENTITY
        
        self._log("\n" + "=" * 60)
        self._log("       IEEE-CIS Fraud Detection Data Preprocessing")
        self._log("=" * 60)
        
        # 1. Load data
        df = self.data_loader.load_and_merge(transaction_path, identity_path)
        
        # 2. Data analysis
        self.data_loader.analyze(df, target)
        
        # 3. Remove high missing columns
        df = self._remove_high_missing_columns(df, target)
        
        # 4. Extract time features
        df = extract_time_features(df, verbose=self.verbose)
        
        # 5. Identify feature types
        self.encoder.identify_feature_types(df, target)
        
        # 6. Handle rare categories
        df = self.encoder.handle_rare_categories(df)
        
        # 7. Encode features
        df = self.encoder.fit_transform(df)
        
        # 8. Optimize memory
        df = optimize_memory(df, verbose=self.verbose)
        
        # 9. Split dataset
        X_train, X_valid, X_test, y_train, y_valid, y_test = self._split_data(df, target)
        
        # Clean up memory
        del df
        gc.collect()
        
        self._log("\n" + "=" * 50)
        self._log("✅ Preprocessing Complete!")
        self._log("=" * 50)
        
        return {
            'X_train': X_train,
            'X_valid': X_valid,
            'X_test': X_test,
            'y_train': y_train,
            'y_valid': y_valid,
            'y_test': y_test,
            'cat_idxs': self.encoder.cat_idxs,
            'cat_dims': self.encoder.cat_dims,
            'feature_columns': self.encoder.feature_columns,
        }
    
    def transform(self, transaction_path=None, identity_path=None):
        """
        Process test set data (for Kaggle submission)
        
        Args:
            transaction_path: Path to test transaction data
            identity_path: Path to test identity data
            
        Returns:
            dict: Contains X_test, transaction_ids
        """
        transaction_path = transaction_path or self.config.TEST_TRANSACTION
        identity_path = identity_path or self.config.TEST_IDENTITY
        
        self._log("\n" + "=" * 60)
        self._log("       Processing Test Set Data (Kaggle Submission)")
        self._log("=" * 60)
        
        # Check if already fitted
        if not self.encoder.label_encoders:
            raise ValueError("Please call fit_transform or load first!")
        
        # 1. Load data
        df = self.data_loader.load_and_merge(transaction_path, identity_path)
        
        # Save TransactionID for submission
        transaction_ids = df['TransactionID'].values
        
        # 2. Drop high missing columns (using columns determined during training)
        self._log("\nDropping high missing columns...")
        cols_to_drop = [c for c in self.drop_columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        self._log(f"Dropped {len(cols_to_drop)} columns")
        
        # 3. Extract time features
        df = extract_time_features(df, verbose=self.verbose)
        
        # 4. Handle missing feature columns
        for col in self.encoder.feature_columns:
            if col not in df.columns:
                self._log(f"Warning: Test set missing column {col}, filling with 0")
                df[col] = 0
        
        # 5. Encode features
        df = self.encoder.transform(df)
        
        # 6. Optimize memory
        df = optimize_memory(df, verbose=self.verbose)
        
        # 7. Extract features
        X_test = df[self.encoder.feature_columns].values.astype(np.float32)
        
        # Clean up memory
        del df
        gc.collect()
        
        self._log("\n" + "=" * 50)
        self._log("✅ Test Set Processing Complete!")
        self._log("=" * 50)
        self._log(f"X_test shape: {X_test.shape}")
        self._log(f"TransactionID count: {len(transaction_ids)}")
        
        return {
            'X_test': X_test,
            'transaction_ids': transaction_ids,
        }
    
    def save(self, path=None):
        """Save preprocessor state"""
        path = path or self.config.PREPROCESSOR_PATH
        
        state = {
            'drop_columns': self.drop_columns,
            'encoder_state': self.encoder.get_state(),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        self._log(f"Preprocessor saved to: {path}")
    
    def load(self, path=None):
        """Load preprocessor state"""
        path = path or self.config.PREPROCESSOR_PATH
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.drop_columns = state['drop_columns']
        self.encoder.load_state(state['encoder_state'])
        
        self._log(f"Preprocessor loaded from: {path}")
