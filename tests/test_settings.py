"""Tests for the Config class in src.config.settings."""

from pathlib import Path

import pytest

from src.config.settings import Config


def test_default_initialization():
    """Test default configuration values."""
    config = Config()

    # Check Path Configuration
    assert isinstance(config.PROJECT_ROOT, Path), "PROJECT_ROOT should be a Path object"
    assert config.DATA_DIR == config.PROJECT_ROOT / "data", "DATA_DIR default path mismatch"
    assert config.CHECKPOINT_DIR == config.PROJECT_ROOT / "checkpoints", (
        "CHECKPOINT_DIR default path mismatch"
    )

    # Check Preprocessing Parameters
    assert config.MISSING_THRESHOLD == 0.9, "Default MISSING_THRESHOLD should be 0.9"
    assert config.USE_TIME_SPLIT is True, "Default USE_TIME_SPLIT should be True"

    # Check TabNet parameters
    assert config.N_D == 48, "Default N_D should be 48"
    assert config.N_A == 48, "Default N_A should be 48"

    # Check Training Parameters
    assert config.MAX_EPOCHS == 100, "Default MAX_EPOCHS should be 100"
    assert config.DEVICE in ["cuda", "cpu"], "DEVICE should be either 'cuda' or 'cpu'"


def test_config_override():
    """Test overriding configuration values via init."""
    overrides = {"MAX_EPOCHS": 50, "BATCH_SIZE": 1024, "MISSING_THRESHOLD": 0.5}
    config = Config(**overrides)

    assert config.MAX_EPOCHS == 50, "MAX_EPOCHS should be overridden to 50"
    assert config.BATCH_SIZE == 1024, "BATCH_SIZE should be overridden to 1024"
    assert config.MISSING_THRESHOLD == 0.5, "MISSING_THRESHOLD should be overridden to 0.5"

    # Check other values remain defaults
    assert config.N_D == 48, "Non-overridden N_D should verify default value"


def test_invalid_config_key():
    """Test raising ValueError for unknown configuration keys."""
    with pytest.raises(ValueError, match="Unknown configuration key: invalid_key"):
        Config(invalid_key=123)


def test_to_dict():
    """Test converting configuration to dictionary."""
    config = Config()
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict), "to_dict should return a dictionary"

    # Check existence of some categories
    assert "PROJECT_ROOT" in config_dict, "Dictionary should contain PROJECT_ROOT"
    assert "MISSING_THRESHOLD" in config_dict, "Dictionary should contain MISSING_THRESHOLD"
    assert "N_D" in config_dict, "Dictionary should contain TabNet parameters"
    assert "DEVICE" in config_dict, "Dictionary should contain DEVICE"

    # Check that methods/private attributes are not included
    assert "__init__" not in config_dict, "Dictionary should not contain methods"
    assert "_log" not in config_dict, "Dictionary should not contain private attributes"


def test_repr():
    """Test string representation of Config."""
    config = Config()
    repr_str = repr(config)

    assert isinstance(repr_str, str), "__repr__ should return a string"
    assert "Config(" in repr_str, "Repr string should start with Config("
    assert "DEVICE=" in repr_str, "Repr string should contain DEVICE info"
    assert "MAX_EPOCHS=" in repr_str, "Repr string should contain MAX_EPOCHS info"


def test_path_existence_check(tmp_path):
    """Test configuration with custom paths."""
    # This just ensures we can set paths dynamically, not that they exist on disk
    # (since Config doesn't create them, just defines them)
    custom_data_dir = tmp_path / "custom_data"
    config = Config(DATA_DIR=custom_data_dir)

    assert config.DATA_DIR == custom_data_dir, "DATA_DIR should be updatable"
