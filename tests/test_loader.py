# Testing src/data/loader.py
import pytest
from src.data.loader import DataLoader
from src.config.settings import Config

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def data_loader(config):
    return DataLoader(config)

def test_init(data_loader):
    assert data_loader.verbose is True

def test_log(data_loader, capsys):
    message = "Test log message"
    data_loader._log(message)
    captured = capsys.readouterr()
    assert message in captured.out

def test_load_transaction(data_loader, tmp_path):
    # Create a temporary CSV file
    transaction_file = tmp_path / "transaction.csv"
    transaction_file.write_text("TransactionID,Amount\n1,100.0\n2,150.0")
    
    df = data_loader.load_transaction(transaction_file)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["TransactionID", "Amount"]

def test_load_identity(data_loader, tmp_path):
    # Create a temporary CSV file
    identity_file = tmp_path / "identity.csv"
    identity_file.write_text("TransactionID,DeviceType\n1,Mobile\n2,Desktop")
    
    df = data_loader.load_identity(identity_file)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["TransactionID", "DeviceType"]

def test_load_and_merge(data_loader, tmp_path):
    # Create temporary CSV files
    transaction_file = tmp_path / "transaction.csv"
    transaction_file.write_text("TransactionID,Amount\n1,100.0\n2,150.0")
    
    identity_file = tmp_path / "identity.csv"
    identity_file.write_text("TransactionID,DeviceType\n1,Mobile\n2,Desktop")
    
    df_merged = data_loader.load_and_merge(transaction_file, identity_file)
    assert df_merged.shape == (2, 3)
    assert list(df_merged.columns) == ["TransactionID", "Amount", "DeviceType"]

def test_analyze(data_loader, tmp_path):
    # Create temporary CSV files
    transaction_file = tmp_path / "transaction.csv"
    transaction_file.write_text("TransactionID,Amount\n1,100.0\n2,150.0")
    
    identity_file = tmp_path / "identity.csv"
    identity_file.write_text("TransactionID,DeviceType\n1,Mobile\n2,Desktop")
    
    df_merged = data_loader.load_and_merge(transaction_file, identity_file)
    analysis = data_loader.analyze(df_merged)
    
    assert "num_rows" in analysis, "Number of rows analysis not found"
    assert analysis["num_rows"] == 2, "Incorrect number of rows in analysis"
    assert "num_columns" in analysis, "Number of columns analysis not found"
    assert analysis["num_columns"] == 3, "Incorrect number of columns in analysis"

    assert "missing_values" in analysis, "Missing values analysis not found"
    assert analysis["missing_values"] == {"TransactionID": 0, "Amount": 0, "DeviceType": 0}, "Incorrect missing values analysis"


