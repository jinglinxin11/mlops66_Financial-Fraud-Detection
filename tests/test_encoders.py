"""Tests for feature encoders."""

import numpy as np
import pandas as pd
import pytest

from src.features.encoders import FeatureEncoder


@pytest.fixture
def dummy_df():
    """Create a dummy DataFrame for testing."""
    data = {
        "cat1": ["A", "B", "A", "C", "A", "B", "D", "E"],
        "cat2": [1, 2, 1, 3, 1, 2, 4, 5],  # Low cardinality integer
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "num2": [10, 20, 30, 40, 50, 60, 70, 80],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
        "id": [100, 101, 102, 103, 104, 105, 106, 107],
    }
    return pd.DataFrame(data)


@pytest.fixture
def encoder():
    """Create a FeatureEncoder instance."""
    return FeatureEncoder(rare_category_threshold=2, verbose=False)


def test_identify_feature_types(encoder, dummy_df):
    """Test identification of categorical and numerical features."""
    feature_cols = encoder.identify_feature_types(
        dummy_df, target="target", exclude_cols=["id", "target"]
    )

    assert "cat1" in encoder.categorical_columns, (
        "cat1 should be identified as categorical"
    )
    assert "cat2" in encoder.categorical_columns, (
        "cat2 should be identified as categorical (low cardinality int)"
    )
    assert "num1" in encoder.numerical_columns, "num1 should be identified as numerical"
    assert "num2" in encoder.categorical_columns, (
        "num2 should be identified as categorical"
    )
    assert "id" not in feature_cols, "id should be excluded from features"
    assert "target" not in feature_cols, "target should be excluded from features"

    # Check order: categorical first
    expected_order = encoder.categorical_columns + encoder.numerical_columns
    assert feature_cols == expected_order


def test_handle_rare_categories(encoder, dummy_df):
    """Test handling of rare categories."""
    # cat1: A(3), B(2) - common; C(1), D(1), E(1) - rare (threshold=2)
    encoder.categorical_columns = ["cat1"]

    df_processed = encoder.handle_rare_categories(dummy_df.copy())

    assert "RARE" in df_processed["cat1"].values
    assert (df_processed["cat1"] == "RARE").sum() == 3  # C, D, E
    assert "A" in df_processed["cat1"].values
    assert "B" in df_processed["cat1"].values


def test_fit_transform(encoder, dummy_df):
    """Test fitting and transforming training data."""
    # Setup columns
    encoder.identify_feature_types(
        dummy_df, target="target", exclude_cols=["id", "target"]
    )

    # Introduce missing values
    df_missing = dummy_df.copy()
    df_missing.loc[0, "cat1"] = np.nan
    df_missing.loc[1, "num1"] = np.nan

    df_encoded = encoder.fit_transform(df_missing)

    # Check missing value filling
    # Categorical missing filled with 'MISSING' then encoded
    assert not df_encoded["cat1"].isnull().any()
    # Numerical missing filled with median
    assert not df_encoded["num1"].isnull().any()

    # Check encoding
    assert df_encoded["cat1"].dtype != "object"
    assert len(encoder.label_encoders) == len(encoder.categorical_columns), (
        "Label encoders should be created for all categorical columns"
    )
    assert len(encoder.numerical_medians) == len(encoder.numerical_columns), (
        "Medians should be stored for all numerical columns"
    )

    # Check cat_dims and cat_idxs
    assert len(encoder.cat_idxs) == len(encoder.categorical_columns), (
        "cat_idxs length should match number of categorical columns"
    )
    assert len(encoder.cat_dims) == len(encoder.categorical_columns), (
        "cat_dims length should match number of categorical columns"
    )


def test_transform(encoder, dummy_df):
    """Test transforming test data (including unseen categories and missing values)."""
    # Fit first
    encoder.identify_feature_types(
        dummy_df, target="target", exclude_cols=["id", "target"]
    )
    encoder.fit_transform(dummy_df.copy())

    # Create test data with new categories and missing values
    test_data = {
        "cat1": ["A", "Z", np.nan],  # 'Z' is unseen
        "cat2": [1, 99, 2],  # 99 is unseen
        "num1": [1.0, 2.0, np.nan],
        "num2": [10, 20, 30],
    }
    df_test = pd.DataFrame(test_data)

    df_transformed = encoder.transform(df_test)

    assert not df_transformed.isnull().any().any(), (
        "There should be no missing values after transform"
    )

    # Unseen 'Z' should be mapped to the first class (usually 0) or handled gracefully
    # Based on code: df.loc[unknown_mask, col] = le.classes_[0]
    le_cat1 = encoder.label_encoders["cat1"]
    assert (
        df_transformed.loc[1, "cat1"] == le_cat1.transform([le_cat1.classes_[0]])[0]
    ), "Unseen category should be mapped to first known class"

    # Numerical missing should be median from train
    expected_median = dummy_df["num1"].median()
    assert df_transformed.loc[2, "num1"] == expected_median, (
        "Missing numerical value should be filled with median from training data"
    )


def test_state_management(encoder, dummy_df):
    """Test saving and loading state."""
    encoder.identify_feature_types(
        dummy_df, target="target", exclude_cols=["id", "target"]
    )
    encoder.fit_transform(dummy_df.copy())

    state = encoder.get_state()

    new_encoder = FeatureEncoder(verbose=False)
    new_encoder.load_state(state)

    assert new_encoder.categorical_columns == encoder.categorical_columns, (
        "Categorical columns should match after loading state"
    )
    assert new_encoder.numerical_columns == encoder.numerical_columns, (
        "Numerical columns should match after loading state"
    )
    assert new_encoder.cat_dims == encoder.cat_dims, (
        "cat_dims should match after loading state"
    )
    assert new_encoder.numerical_medians["num1"] == encoder.numerical_medians["num1"], (
        "Numerical medians should match after loading state"
    )
