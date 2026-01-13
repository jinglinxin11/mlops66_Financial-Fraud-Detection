"""Feature Encoding Module
Handles encoding of categorical and numerical features.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder


class FeatureEncoder:
    """Feature Encoder."""
    
    def __init__(self, rare_category_threshold=100, verbose=True):
        """Args:
        rare_category_threshold: Rare category threshold
        verbose: Whether to print information.
        """
        self.rare_category_threshold = rare_category_threshold
        self.verbose = verbose
        
        self.label_encoders = {}
        self.numerical_medians = {}
        self.categorical_modes = {}
        self.categorical_columns = []
        self.numerical_columns = []
        self.feature_columns = []
        self.cat_idxs = []
        self.cat_dims = []
    
    def _log(self, message):
        """Print log message."""
        if self.verbose:
            print(message)
    
    def identify_feature_types(self, df, target='isFraud', exclude_cols=None):
        """Identify feature types.
        
        Args:
            df: DataFrame
            target: Target column name
            exclude_cols: Columns to exclude
            
        Returns:
            list: List of feature column names
        """
        self._log("\n" + "=" * 50)
        self._log("Identifying Feature Types")
        self._log("=" * 50)
        
        if exclude_cols is None:
            exclude_cols = ['TransactionID', 'TransactionDT', target]
        
        self.categorical_columns = []
        self.numerical_columns = []
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            if df[col].dtype == 'object':
                self.categorical_columns.append(col)
            elif df[col].nunique() < 50 and df[col].dtype in ['int64', 'int32']:
                # Integer columns with fewer than 50 unique values are treated as categorical
                self.categorical_columns.append(col)
            else:
                self.numerical_columns.append(col)
        
        self._log(f"Categorical features: {len(self.categorical_columns)}")
        self._log(f"Numerical features: {len(self.numerical_columns)}")
        
        # Feature order: categorical first, then numerical
        self.feature_columns = self.categorical_columns + self.numerical_columns
        
        return self.feature_columns
    
    def handle_rare_categories(self, df):
        """Handle rare categories.
        
        Args:
            df: DataFrame
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        self._log("\n" + "=" * 50)
        self._log("Handling Rare Categories")
        self._log("=" * 50)
        
        rare_count = 0
        for col in self.categorical_columns:
            value_counts = df[col].value_counts()
            rare_values = value_counts[value_counts < self.rare_category_threshold].index
            
            if len(rare_values) > 0:
                df[col] = df[col].replace(rare_values, 'RARE')
                rare_count += len(rare_values)
        
        self._log(f"Merged {rare_count} rare category values into 'RARE'")
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform features (training set).
        
        Args:
            df: DataFrame
            
        Returns:
            pd.DataFrame: Encoded DataFrame
        """
        self._log("\n" + "=" * 50)
        self._log("Feature Encoding (Training)")
        self._log("=" * 50)
        
        # Process categorical features
        self._log("Processing categorical features...")
        for col in self.categorical_columns:
            # Fill missing values
            self.categorical_modes[col] = 'MISSING'
            df[col] = df[col].fillna(self.categorical_modes[col])
            
            # LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Process numerical features
        self._log("Processing numerical features...")
        for col in self.numerical_columns:
            self.numerical_medians[col] = df[col].median()
            df[col] = df[col].fillna(self.numerical_medians[col])
        
        # Calculate cat_idxs and cat_dims
        self.cat_idxs = list(range(len(self.categorical_columns)))
        self.cat_dims = [
            len(self.label_encoders[col].classes_) 
            for col in self.categorical_columns
        ]
        
        self._log(f"cat_idxs: {self.cat_idxs[:5]}... (total {len(self.cat_idxs)})")
        self._log(f"cat_dims: {self.cat_dims[:5]}... (total {len(self.cat_dims)})")
        
        return df
    
    def transform(self, df):
        """Transform features (test set).
        
        Args:
            df: DataFrame
            
        Returns:
            pd.DataFrame: Encoded DataFrame
        """
        self._log("\nEncoding features...")
        
        for col in self.categorical_columns:
            if col not in df.columns:
                df[col] = 0
                continue
            
            # Fill missing values
            df[col] = df[col].fillna(self.categorical_modes.get(col, 'MISSING'))
            
            # Use LabelEncoder from training
            le = self.label_encoders[col]
            df[col] = df[col].astype(str)
            
            # Handle unseen categories
            unknown_mask = ~df[col].isin(le.classes_)
            if unknown_mask.any():
                self._log(f"  {col}: {unknown_mask.sum()} unknown categories")
                df.loc[unknown_mask, col] = le.classes_[0]
            
            df[col] = le.transform(df[col])
        
        for col in self.numerical_columns:
            if col not in df.columns:
                df[col] = self.numerical_medians.get(col, 0)
                continue
            df[col] = df[col].fillna(self.numerical_medians.get(col, 0))
        
        return df
    
    def get_state(self):
        """Get encoder state for saving."""
        return {
            'label_encoders': self.label_encoders,
            'numerical_medians': self.numerical_medians,
            'categorical_modes': self.categorical_modes,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'feature_columns': self.feature_columns,
            'cat_idxs': self.cat_idxs,
            'cat_dims': self.cat_dims,
        }
    
    def load_state(self, state):
        """Restore encoder from saved state."""
        self.label_encoders = state['label_encoders']
        self.numerical_medians = state['numerical_medians']
        self.categorical_modes = state['categorical_modes']
        self.categorical_columns = state['categorical_columns']
        self.numerical_columns = state['numerical_columns']
        self.feature_columns = state['feature_columns']
        self.cat_idxs = state['cat_idxs']
        self.cat_dims = state['cat_dims']
