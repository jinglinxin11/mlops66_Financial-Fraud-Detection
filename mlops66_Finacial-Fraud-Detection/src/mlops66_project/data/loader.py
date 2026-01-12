"""
Data Loading Module
Responsible for loading and merging IEEE-CIS dataset
"""

import pandas as pd
import gc
from pathlib import Path


class DataLoader:
    """IEEE-CIS Data Loader"""
    
    def __init__(self, verbose=True):
        """
        Args:
            verbose: Whether to print loading information
        """
        self.verbose = verbose
    
    def _log(self, message):
        """Print log message"""
        if self.verbose:
            print(message)
    
    def load_transaction(self, path):
        """
        Load transaction data
        
        Args:
            path: Path to transaction.csv file
            
        Returns:
            pd.DataFrame: Transaction data
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        self._log(f"Loading {path.name}...")
        df = pd.read_csv(path)
        self._log(f"  Shape: {df.shape}")
        
        return df
    
    def load_identity(self, path):
        """
        Load identity data
        
        Args:
            path: Path to identity.csv file
            
        Returns:
            pd.DataFrame or None: Identity data
        """
        path = Path(path)
        if not path.exists():
            self._log(f"  Identity data not found: {path.name}, skipping")
            return None
        
        self._log(f"Loading {path.name}...")
        df = pd.read_csv(path)
        self._log(f"  Shape: {df.shape}")
        
        return df
    
    def load_and_merge(self, transaction_path, identity_path=None):
        """
        Load and merge transaction and identity data
        
        Args:
            transaction_path: Path to transaction.csv file
            identity_path: Path to identity.csv file (optional)
            
        Returns:
            pd.DataFrame: Merged data
        """
        self._log("=" * 50)
        self._log("Loading Data")
        self._log("=" * 50)
        
        # Load transaction data
        df = self.load_transaction(transaction_path)
        
        # Load and merge identity data
        if identity_path:
            identity = self.load_identity(identity_path)
            if identity is not None:
                df = df.merge(identity, on='TransactionID', how='left')
                self._log(f"  Merged shape: {df.shape}")
                del identity
                gc.collect()
        
        return df
    
    def analyze(self, df, target='isFraud'):
        """
        Data quality analysis report
        
        Args:
            df: DataFrame
            target: Target column name
            
        Returns:
            pd.Series: Missing rate for each column
        """
        self._log("\n" + "=" * 50)
        self._log("Data Quality Analysis")
        self._log("=" * 50)
        
        self._log(f"\nData shape: {df.shape}")
        self._log(f"Samples: {len(df):,}")
        self._log(f"Features: {df.shape[1]}")
        
        # Target distribution
        if target in df.columns:
            fraud_rate = df[target].mean()
            self._log(f"\nTarget distribution:")
            self._log(f"  Normal transactions: {(1-fraud_rate)*100:.2f}%")
            self._log(f"  Fraud transactions: {fraud_rate*100:.2f}% ⚠️ Highly imbalanced")
        
        # Missing value analysis
        import numpy as np
        missing = df.isnull().mean()
        self._log(f"\nMissing value analysis:")
        self._log(f"  Columns with no missing: {(missing == 0).sum()}")
        self._log(f"  Columns with >50% missing: {(missing > 0.5).sum()}")
        self._log(f"  Columns with >90% missing: {(missing > 0.9).sum()}")
        
        # Data types
        self._log(f"\nData types:")
        self._log(f"  Numerical: {df.select_dtypes(include=[np.number]).shape[1]}")
        self._log(f"  Categorical: {df.select_dtypes(include=['object']).shape[1]}")
        
        return missing
