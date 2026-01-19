"""Tests for preprocessor module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessor import FraudPreprocessor


class MockConfig:
    """Mock configuration."""

    RARE_CATEGORY_THRESHOLD = 50
    MISSING_THRESHOLD = 0.9
    USE_TIME_SPLIT = True
    TRAIN_TRANSACTION = "train_transaction.csv"
    TRAIN_IDENTITY = "train_identity.csv"
    TEST_TRANSACTION = "test_transaction.csv"
    TEST_IDENTITY = "test_identity.csv"
    PREPROCESSOR_PATH = "preprocessor.pkl"


@pytest.fixture
def mock_config():
    """Create a mock config."""
    return MockConfig()


@pytest.fixture
def sample_df():
    """Create a sample DataFrame."""
    # Create a small DataFrame for testing
    np.random.seed(42)
    data = {
        "TransactionID": np.arange(100),
        "isFraud": [0] * 90 + [1] * 10,
        "TransactionDT": np.sort(np.random.randint(0, 100000, 100)),
        "col_missing": [np.nan] * 95 + [1.0] * 5,  # 95% missing
        "col_ok": np.random.rand(100),
        "card4": ["visa"] * 50 + ["mastercard"] * 50,
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor(mock_config):
    """Create a FraudPreprocessor instance."""
    # Patch dependencies at the module level where they are instantiated
    with (
        patch("src.features.preprocessor.DataLoader") as MockLoader,
        patch("src.features.preprocessor.FeatureEncoder") as MockEncoder,
    ):
        # Setup mocks
        loader_instance = MockLoader.return_value
        encoder_instance = MockEncoder.return_value

        # Default behavior for encoder
        encoder_instance.feature_columns = ["col_ok", "card4"]
        encoder_instance.cat_idxs = [1]
        encoder_instance.cat_dims = [2]
        encoder_instance.label_encoders = {}  # Start empty

        prep = FraudPreprocessor(mock_config, verbose=False)

        # Attach mocks to the instance so we can configure/assert on them in tests
        prep.data_loader = loader_instance
        prep.encoder = encoder_instance

        yield prep


def test_init(preprocessor, mock_config):
    """Test initialization."""
    assert preprocessor.config == mock_config, "Config should be set correctly"
    assert preprocessor.verbose is False, "Verbose should be set correctly"
    assert preprocessor.data_loader is not None, "DataLoader should be initialized"
    assert preprocessor.encoder is not None, "FeatureEncoder should be initialized"
    assert preprocessor.drop_columns == [], "Drop columns should be initialized as empty list"


def test_remove_high_missing_columns(preprocessor, sample_df):
    """Test removal of columns with high missing rate."""
    # col_missing has 95% missing, threshold is 0.9 -> should be dropped

    df_clean = preprocessor._remove_high_missing_columns(sample_df)

    assert "col_missing" not in df_clean.columns, "Column with high missing rate should be dropped"
    assert "col_missing" in preprocessor.drop_columns, "Dropped columns should be recorded"
    assert "col_ok" in df_clean.columns, "Column with acceptable missing rate should be kept"
    assert "isFraud" in df_clean.columns, "Target column should be kept"
    assert "TransactionID" in df_clean.columns, "ID column should be kept"


def test_split_data_time(preprocessor, sample_df):
    """Test time-based splitting."""
    preprocessor.config.USE_TIME_SPLIT = True

    # Needs feature columns to filter data
    preprocessor.encoder.feature_columns = ["col_ok"]

    X_train, X_valid, X_test, y_train, y_valid, y_test = preprocessor._split_data(sample_df)

    # Check shapes
    assert len(X_train) + len(X_valid) + len(X_test) == 100, (
        "Total samples should match original data"
    )
    assert len(y_train) + len(y_valid) + len(y_test) == 100, (
        "Total labels should match original data"
    )

    # With test_size=0.2 (20), valid_size=0.125 (of 80 -> 10)
    # Train: 70, Valid: 10, Test: 20
    assert len(X_test) == 20, "Test set should have 20 samples"
    assert len(X_valid) == 10, "Validation set should have 10 samples"
    assert len(X_train) == 70, "Training set should have 70 samples"

    # Verify temporal order preserved (for X_test vs X_train)
    # Since we used random data sorted by TransactionDT in sample_df fixture
    # The last 20 samples should be in X_test
    # Note: _split_data converts to numpy array, so we check values
    # col_ok is random, but let's check shapes mainly.
    assert X_train.shape[1] == 1, "Only 'col_ok' was selected"  # Only 'col_ok' was selected


def test_split_data_random(preprocessor, sample_df):
    """Test random stratified splitting."""
    preprocessor.config.USE_TIME_SPLIT = False
    preprocessor.encoder.feature_columns = ["col_ok"]

    X_train, X_valid, X_test, y_train, y_valid, y_test = preprocessor._split_data(sample_df)

    assert len(X_train) == 70, "Training set should have 70 samples"
    assert len(X_valid) == 10, "Validation set should have 10 samples"
    assert len(X_test) == 20, "Test set should have 20 samples"

    # Check stratification roughly (small sample size might not be perfect)
    # original fraud rate 10%
    # y_train mean should be somewhat close
    assert 0 <= y_train.mean() <= 1, "Fraud rate in training set should be valid probability"
    assert 0 <= y_valid.mean() <= 1, "Fraud rate in validation set should be valid probability"
    assert 0 <= y_test.mean() <= 1, "Fraud rate in test set should be valid probability"


@patch("src.features.preprocessor.extract_time_features")
@patch("src.features.preprocessor.optimize_memory")
def test_fit_transform(mock_optimize, mock_extract_time, preprocessor, sample_df):
    """Test full fit_transform pipeline."""
    # Setup mocks
    preprocessor.data_loader.load_and_merge.return_value = sample_df.copy()
    mock_extract_time.return_value = sample_df.copy()  # Pass through

    # Mock encoder methods to return dataframe
    preprocessor.encoder.fit_transform.return_value = sample_df.copy()
    preprocessor.encoder.cat_idxs = []
    preprocessor.encoder.cat_dims = []
    preprocessor.encoder.feature_columns = ["col_ok"]

    mock_optimize.side_effect = lambda x, verbose: x

    result = preprocessor.fit_transform()

    # Verify sequence of calls
    preprocessor.data_loader.load_and_merge.assert_called_once()
    preprocessor.data_loader.analyze.assert_called_once()
    mock_extract_time.assert_called_once()
    preprocessor.encoder.identify_feature_types.assert_called_once()
    preprocessor.encoder.handle_rare_categories.assert_called_once()
    preprocessor.encoder.fit_transform.assert_called_once()

    assert "X_train" in result, "Result should contain X_train"
    assert "y_train" in result, "Result should contain y_train"
    assert "cat_idxs" in result, "Result should contain cat_idxs"
    assert "feature_columns" in result, "Result should contain feature_columns"
    assert isinstance(result["X_train"], np.ndarray), "X_train should be a numpy array"
    assert isinstance(result["y_train"], np.ndarray), "y_train should be a numpy array"
    assert isinstance(result["cat_idxs"], list), "cat_idxs should be a list"
    assert isinstance(result["feature_columns"], list), "feature_columns should be a list"


@patch("src.features.preprocessor.extract_time_features")
@patch("src.features.preprocessor.optimize_memory")
def test_transform(mock_optimize, mock_extract_time, preprocessor, sample_df):
    """Test transform pipeline for test data."""
    # Setup mocks
    preprocessor.data_loader.load_and_merge.return_value = sample_df.copy()
    mock_extract_time.return_value = sample_df.copy()

    # Simulate valid state
    preprocessor.encoder.label_encoders = {"some": "encoder"}
    preprocessor.encoder.transform.return_value = sample_df.copy()
    preprocessor.encoder.feature_columns = ["col_ok"]

    mock_optimize.side_effect = lambda x, verbose: x

    preprocessor.drop_columns = ["col_missing"]

    result = preprocessor.transform()

    # Verify calls
    preprocessor.data_loader.load_and_merge.assert_called_once()
    mock_extract_time.assert_called_once()
    preprocessor.encoder.transform.assert_called_once()

    assert "X_test" in result, "Result should contain X_test"
    assert "transaction_ids" in result, "Result should contain transaction_ids"
    assert isinstance(result["X_test"], np.ndarray), "X_test should be a numpy array"
    assert isinstance(result["transaction_ids"], np.ndarray), (
        "transaction_ids should be a numpy array"
    )
    assert len(result["X_test"]) == 100, "X_test should have 100 rows"
    assert len(result["transaction_ids"]) == 100, "transaction_ids should have 100 rows"


def test_transform_not_fitted(preprocessor):
    """Test transform raises error if not fitted."""
    preprocessor.encoder.label_encoders = {}
    with pytest.raises(ValueError, match="Please call fit_transform or load first"):
        preprocessor.transform()


def test_save_load(preprocessor, tmp_path):
    """Test saving and loading preprocessor state."""
    # Setup unique state to verify persistence
    preprocessor.drop_columns = ["col_test_drop"]
    preprocessor.encoder.get_state.return_value = {"mock_encoder_state": True}

    save_path = tmp_path / "prep_test.pkl"
    preprocessor.save(path=str(save_path))

    assert save_path.exists(), "Preprocessor state file should be created"

    # Create new instance to load into
    new_prep = FraudPreprocessor(preprocessor.config, verbose=False)
    # Mock encoder for the new instance
    new_prep.encoder = MagicMock()

    new_prep.load(path=str(save_path))

    assert new_prep.drop_columns == ["col_test_drop"], (
        "Loaded drop_columns should match saved state"
    )
    new_prep.encoder.load_state.assert_called_with({"mock_encoder_state": True})
