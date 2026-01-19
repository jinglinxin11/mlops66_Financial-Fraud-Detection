"""Tests for the DataLoader class in src.data.loader."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader


@pytest.fixture
def loader():
    """Fixture for DataLoader instance."""
    return DataLoader(verbose=True)


@pytest.fixture
def sample_data(tmp_path):
    """Create sample transaction and identity CSV files."""
    # Transaction data
    trans_df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "amount": [10.5, 20.0, 30.5],
            "card1": [1000, 1001, 1002],
        }
    )
    trans_path = tmp_path / "transaction.csv"
    trans_df.to_csv(trans_path, index=False)

    # Identity data
    id_df = pd.DataFrame({"TransactionID": [1, 2], "device_info": ["mobile", "desktop"]})
    id_path = tmp_path / "identity.csv"
    id_df.to_csv(id_path, index=False)

    return trans_path, id_path


def test_init():
    """Test DataLoader initialization."""
    loader = DataLoader(verbose=False)
    assert loader.verbose is False, (
        "DataLoader verbose flag should be False when initialized with verbose=False"
    )

    loader_verbose = DataLoader(verbose=True)
    assert loader_verbose.verbose is True, (
        "DataLoader verbose flag should be True by default or when initialized with True"
    )


def test_load_transaction_success(loader, sample_data):
    """Test successful loading of transaction data."""
    trans_path, _ = sample_data
    df = loader.load_transaction(trans_path)

    assert isinstance(df, pd.DataFrame), "load_transaction should return a pandas DataFrame"
    assert not df.empty, "Loaded transaction DataFrame should not be empty"
    assert df.shape == (3, 4), f"Expected shape (3, 4), but got {df.shape}"
    assert "TransactionID" in df.columns, "TransactionID column missing from loaded data"


def test_load_transaction_file_not_found(loader, tmp_path):
    """Test load_transaction raises FileNotFoundError for non-existent file."""
    fake_path = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError, match="File not found"):
        loader.load_transaction(fake_path)


def test_load_identity_success(loader, sample_data):
    """Test successful loading of identity data."""
    _, id_path = sample_data
    df = loader.load_identity(id_path)

    assert isinstance(df, pd.DataFrame), "load_identity should return a pandas DataFrame"
    assert df.shape == (2, 2), f"Expected shape (2, 2), but got {df.shape}"


def test_load_identity_not_found(loader, tmp_path):
    """Test load_identity returns None for non-existent file."""
    fake_path = tmp_path / "non_existent_id.csv"
    result = loader.load_identity(fake_path)

    assert result is None, "load_identity should return None when file does not exist"


def test_load_and_merge(loader, sample_data):
    """Test loading and merging of transaction and identity data."""
    trans_path, id_path = sample_data
    merged_df = loader.load_and_merge(trans_path, id_path)

    assert isinstance(merged_df, pd.DataFrame), "load_and_merge should return a DataFrame"
    assert merged_df.shape[0] == 3, f"Expected 3 rows in merged data, got {merged_df.shape[0]}"
    assert "device_info" in merged_df.columns, (
        "Merged data should contain identity columns (device_info)"
    )

    # Check left join behavior (TransactionID 3 has no identity info)
    row_3 = merged_df[merged_df["TransactionID"] == 3]
    assert pd.isna(row_3["device_info"].iloc[0]), (
        "Expected NaN for missing identity match in left join"
    )


def test_load_and_merge_no_identity(loader, sample_data):
    """Test load_and_merge without identity path."""
    trans_path, _ = sample_data
    df = loader.load_and_merge(trans_path, identity_path=None)

    assert isinstance(df, pd.DataFrame), "Should return DataFrame even without identity path"
    assert df.shape == (3, 4), f"Expected shape (3, 4) without merge, got {df.shape}"
    assert "device_info" not in df.columns, (
        "Result should not contain identity columns when identity_path is None"
    )


def test_analyze(loader, sample_data):
    """Test analyze method returns missing value statistics."""
    trans_path, _ = sample_data
    df = loader.load_transaction(trans_path)

    # Inject some missing values for testing
    df.loc[0, "amount"] = np.nan

    missing_stats = loader.analyze(df, target="isFraud")

    assert isinstance(missing_stats, pd.Series), "analyze should return a pandas Series"
    assert missing_stats["amount"] > 0, "Analyze should detect missing values in 'amount' column"
    assert missing_stats.shape[0] == df.shape[1], "Missing stats should have one entry per column"


def test_logging(capsys, sample_data):
    """Test that logs are printed when verbose is True."""
    trans_path, _ = sample_data
    loader = DataLoader(verbose=True)
    loader.load_transaction(trans_path)

    captured = capsys.readouterr()
    assert "Loading" in captured.out, "Should print 'Loading' message when verbose=True"
    assert "Shape:" in captured.out, "Should print shape information when verbose=True"


def test_no_logging(capsys, sample_data):
    """Test that logs are suppressed when verbose is False."""
    trans_path, _ = sample_data
    loader = DataLoader(verbose=False)
    loader.load_transaction(trans_path)

    captured = capsys.readouterr()
    assert captured.out == "", "Should not print anything when verbose=False"
