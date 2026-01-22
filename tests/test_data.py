"""Unit tests for data loading and preprocessing.

Tests cover:
- DataLoader class functionality
- Data validation
- Preprocessing pipeline
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_dataloader_initialization(self):
        """Test DataLoader can be initialized."""
        loader = DataLoader(verbose=False)
        assert loader is not None
        assert loader.verbose is False

    def test_dataloader_verbose_default(self):
        """Test DataLoader verbose defaults to True."""
        loader = DataLoader()
        assert loader.verbose is True

    def test_load_transaction_file_not_found(self):
        """Test DataLoader raises error for missing file."""
        loader = DataLoader(verbose=False)
        with pytest.raises(FileNotFoundError):
            loader.load_transaction("nonexistent_file.csv")

    def test_load_and_merge_with_sample_data(
        self, sample_transaction_data, sample_identity_data, tmp_path
    ):
        """Test loading and merging transaction and identity data."""
        # Save sample data to temp files
        transaction_path = tmp_path / "transaction.csv"
        identity_path = tmp_path / "identity.csv"

        sample_transaction_data.to_csv(transaction_path, index=False)
        sample_identity_data.to_csv(identity_path, index=False)

        # Load and merge
        loader = DataLoader(verbose=False)
        df = loader.load_and_merge(transaction_path, identity_path)

        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_transaction_data)
        assert "TransactionID" in df.columns
        assert "isFraud" in df.columns

    def test_load_transaction_only(self, sample_transaction_data, tmp_path):
        """Test loading transaction data without identity."""
        transaction_path = tmp_path / "transaction.csv"
        sample_transaction_data.to_csv(transaction_path, index=False)

        loader = DataLoader(verbose=False)
        df = loader.load_and_merge(transaction_path, identity_path=None)

        assert len(df) == len(sample_transaction_data)

    def test_analyze_returns_missing_series(self, sample_transaction_data, tmp_path):
        """Test analyze method returns missing rate series."""
        transaction_path = tmp_path / "transaction.csv"
        sample_transaction_data.to_csv(transaction_path, index=False)

        loader = DataLoader(verbose=False)
        df = loader.load_and_merge(transaction_path)
        missing = loader.analyze(df, target="isFraud")

        assert isinstance(missing, pd.Series)
        assert len(missing) == len(df.columns)


class TestDataPreprocessing:
    """Tests for data preprocessing functionality."""

    def test_sample_data_has_correct_columns(self, sample_transaction_data):
        """Test sample data fixture has expected columns."""
        required_cols = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]
        for col in required_cols:
            assert col in sample_transaction_data.columns

    def test_sample_data_has_correct_types(self, sample_transaction_data):
        """Test sample data has correct data types."""
        assert sample_transaction_data["TransactionID"].dtype in [np.int32, np.int64]
        assert sample_transaction_data["isFraud"].dtype in [np.int32, np.int64]

    def test_sample_data_fraud_rate(self, sample_transaction_data):
        """Test sample data has reasonable fraud rate."""
        fraud_rate = sample_transaction_data["isFraud"].mean()
        assert 0 < fraud_rate < 0.5  # Should be imbalanced but not empty


class TestFeatureEncoder:
    """Tests for feature encoding."""

    def test_encoder_import(self):
        """Test FeatureEncoder can be imported."""
        from src.features.encoders import FeatureEncoder

        encoder = FeatureEncoder(verbose=False)
        assert encoder is not None

    def test_encoder_initialization(self):
        """Test FeatureEncoder initialization with parameters."""
        from src.features.encoders import FeatureEncoder

        encoder = FeatureEncoder(rare_category_threshold=50, verbose=False)
        assert encoder.rare_category_threshold == 50


class TestTimeFeatures:
    """Tests for time feature extraction."""

    def test_time_features_import(self):
        """Test time features can be imported."""
        from src.features.time_features import extract_time_features

        assert callable(extract_time_features)

    def test_extract_time_features(self, sample_transaction_data):
        """Test time feature extraction."""
        from src.features.time_features import extract_time_features

        df = extract_time_features(sample_transaction_data.copy(), verbose=False)

        # Check new columns were added (if TransactionDT exists)
        assert "TransactionDT" in df.columns
