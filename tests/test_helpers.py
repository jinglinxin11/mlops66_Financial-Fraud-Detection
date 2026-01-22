"""Tests for helper functions."""

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from src.utils.helpers import find_latest_checkpoint, optimize_memory

# -----------------------------------------------------------------------------
# optimize_memory Tests
# -----------------------------------------------------------------------------


def test_optimize_memory_dtypes():
    """Test that data types are downcasted correctly."""
    # Create a DataFrame with 64-bit types
    df = pd.DataFrame(
        {
            "float_col": np.array([1.1, 2.2, 3.3], dtype="float64"),
            "int_col": np.array([1, 2, 3000], dtype="int64"),
            "obj_col": ["a", "b", "c"],  # Should remain untouched
        }
    )

    assert df["float_col"].dtype == "float64", "Setup error: float_col should be float64"
    assert df["int_col"].dtype == "int64", "Setup error: int_col should be int64"

    # Run optimization
    df_opt = optimize_memory(df.copy(), verbose=False)

    # Check dtypes
    assert df_opt["float_col"].dtype == "float32", (
        f"Float64 should be converted to float32. Got {df_opt['float_col'].dtype}"
    )
    assert df_opt["int_col"].dtype == "int32", (
        f"Int64 should be converted to int32. Got {df_opt['int_col'].dtype}"
    )
    assert is_object_dtype(df_opt["obj_col"]) or is_string_dtype(df_opt["obj_col"]), (
        "Object column should remain as object/string type"
    )


def test_optimize_memory_values_preserved():
    """Test that values are preserved after downcasting."""
    df = pd.DataFrame(
        {
            "float_col": np.array([1.123456, 2.987654], dtype="float64"),
            "int_col": np.array([100, 200], dtype="int64"),
        }
    )

    df_opt = optimize_memory(df.copy(), verbose=False)

    # Check values
    # Floats might have small precision loss, acceptable within float32 limits
    np.testing.assert_allclose(
        df_opt["float_col"],
        df["float_col"],
        rtol=1e-6,
        err_msg="Float values changed significantly during optimization",
    )

    np.testing.assert_array_equal(
        df_opt["int_col"],
        df["int_col"],
        err_msg="Integer values changed during optimization",
    )


# -----------------------------------------------------------------------------
# find_latest_checkpoint Tests
# -----------------------------------------------------------------------------


def test_find_latest_checkpoint_no_dir():
    """Test behavior when directory does not exist."""
    path, epoch = find_latest_checkpoint("/non/existent/path/123")
    assert path is None, f"Should return None for path if dir missing, got {path}"
    assert epoch == 0, f"Should return 0 for epoch if dir missing, got {epoch}"


def test_find_latest_checkpoint_empty_dir(tmp_path):
    """Test behavior when directory exists but is empty."""
    # tmp_path creates a dir automatically
    path, epoch = find_latest_checkpoint(str(tmp_path))
    assert path is None, "Should return None for path if dir empty"
    assert epoch == 0, "Should return 0 for epoch if dir empty"


def test_find_latest_checkpoint_no_matches(tmp_path):
    """Test behavior when directory has files but not checkpoints."""
    # Create a dummy file
    (tmp_path / "model.txt").write_text("content")

    path, epoch = find_latest_checkpoint(str(tmp_path))
    assert path is None, "Should return None for path if no matching checkpoint files"
    assert epoch == 0, "Should return 0 for epoch if no matching checkpoint files"


def test_find_latest_checkpoint_success(tmp_path):
    """Test that the latest checkpoint is correctly identified."""
    # Create mock checkpoint files
    # Format: checkpoint_epoch_X.zip
    (tmp_path / "checkpoint_epoch_5.zip").touch()
    (tmp_path / "checkpoint_epoch_10.zip").touch()
    (tmp_path / "checkpoint_epoch_2.zip").touch()

    # Create a confusing file that shouldn't match pattern exactly if strict logic
    # The glob is checkpoint_epoch_*.zip
    (tmp_path / "other_file.zip").touch()

    path, epoch = find_latest_checkpoint(str(tmp_path))

    assert epoch == 10, f"Should identify 10 as latest epoch, got {epoch}"
    assert path is not None, "Path should not be None"
    assert path.endswith("checkpoint_epoch_10.zip"), (
        f"Path should point to epoch 10 file, got {path}"
    )
    assert str(tmp_path) in path, "Path should be absolute/complete"


def test_find_latest_checkpoint_malformed_filenames(tmp_path):
    """Test resilience against files matching glob but having bad format."""
    # Matches glob 'checkpoint_epoch_*.zip' but 'invalid' is not an int
    (tmp_path / "checkpoint_epoch_invalid.zip").touch()
    # Good file
    (tmp_path / "checkpoint_epoch_1.zip").touch()

    path, epoch = find_latest_checkpoint(str(tmp_path))

    # Should ignore the invalid one and find the valid one (epoch 1)
    assert epoch == 1, f"Should skip malformed filename and find epoch 1, got {epoch}"
    assert path.endswith("checkpoint_epoch_1.zip")
