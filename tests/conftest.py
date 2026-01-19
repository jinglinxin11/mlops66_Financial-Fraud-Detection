"""
Pytest fixtures and shared test resources.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'TransactionID': range(1000, 1000 + n_samples),
        'TransactionDT': np.random.randint(86400, 86400 * 30, n_samples),
        'TransactionAmt': np.random.exponential(100, n_samples),
        'ProductCD': np.random.choice(['W', 'H', 'C', 'S', 'R'], n_samples),
        'card1': np.random.randint(1000, 9999, n_samples),
        'card2': np.random.choice([100, 200, 300, np.nan], n_samples),
        'card4': np.random.choice(['visa', 'mastercard', 'discover', np.nan], n_samples),
        'card6': np.random.choice(['credit', 'debit', np.nan], n_samples),
        'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', np.nan], n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.965, 0.035]),
    }
    
    # Add some V columns
    for i in range(1, 10):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_identity_data():
    """Create sample identity data for testing."""
    np.random.seed(42)
    n_samples = 50  # Less than transaction (simulates left join)
    
    data = {
        'TransactionID': range(1000, 1000 + n_samples),
        'id_01': np.random.randn(n_samples),
        'id_02': np.random.randn(n_samples) * 1000,
        'id_12': np.random.choice(['Found', 'NotFound', np.nan], n_samples),
        'DeviceType': np.random.choice(['desktop', 'mobile', np.nan], n_samples),
        'DeviceInfo': np.random.choice(['Windows', 'iOS Device', 'MacOS', np.nan], n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_config():
    """Create mock configuration object for testing."""
    class MockConfig:
        PROJECT_ROOT = Path(__file__).parent.parent
        DATA_DIR = PROJECT_ROOT / "data"
        TRAIN_TRANSACTION = DATA_DIR / "train_transaction.csv"
        TRAIN_IDENTITY = DATA_DIR / "train_identity.csv"
        TEST_TRANSACTION = DATA_DIR / "test_transaction.csv"
        TEST_IDENTITY = DATA_DIR / "test_identity.csv"
        CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
        MODEL_PATH = PROJECT_ROOT / "tabnet_fraud_model"
        PREPROCESSOR_PATH = PROJECT_ROOT / "ieee_cis_preprocessor.pkl"
        
        MISSING_THRESHOLD = 0.9
        RARE_CATEGORY_THRESHOLD = 100
        USE_TIME_SPLIT = False  # Use random split for testing
        
        N_D = 8
        N_A = 8
        N_STEPS = 3
        GAMMA = 1.5
        CAT_EMB_DIM = 1
        LAMBDA_SPARSE = 1e-3
        MASK_TYPE = 'entmax'
        
        MAX_EPOCHS = 2
        PATIENCE = 2
        BATCH_SIZE = 32
        VIRTUAL_BATCH_SIZE = 16
        LEARNING_RATE = 0.02
        SCHEDULER_STEP_SIZE = 10
        SCHEDULER_GAMMA = 0.9
        
        CHECKPOINT_EVERY = 1
        RESUME_TRAINING = False
        
        DEVICE = 'cpu'
        
        UNCERTAINTY_THRESHOLDS = {
            'high_risk': 0.8,
            'medium_high': 0.5,
            'medium_low': 0.3,
            'low_risk': 0.1
        }
    
    return MockConfig()


@pytest.fixture
def sample_preprocessed_data():
    """Create sample preprocessed data for model testing."""
    np.random.seed(42)
    n_train = 200
    n_valid = 50
    n_test = 50
    n_features = 20
    n_cat_features = 5
    
    return {
        'X_train': np.random.randn(n_train, n_features).astype(np.float32),
        'X_valid': np.random.randn(n_valid, n_features).astype(np.float32),
        'X_test': np.random.randn(n_test, n_features).astype(np.float32),
        'y_train': np.random.choice([0, 1], n_train, p=[0.9, 0.1]).astype(np.int64),
        'y_valid': np.random.choice([0, 1], n_valid, p=[0.9, 0.1]).astype(np.int64),
        'y_test': np.random.choice([0, 1], n_test, p=[0.9, 0.1]).astype(np.int64),
        'cat_idxs': list(range(n_cat_features)),
        'cat_dims': [10] * n_cat_features,
        'feature_columns': [f'feature_{i}' for i in range(n_features)],
    }
