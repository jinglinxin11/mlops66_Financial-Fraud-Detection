"""Tests for evaluation metrics."""

import numpy as np
import pytest
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.datasets import make_classification

from src.evaluation.metrics import evaluate_model, get_feature_importance


@pytest.fixture
def dummy_data():
    """Create dummy classification data."""
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=10, n_redundant=5, random_state=42
    )
    # Ensure strict numpy types
    X = np.ascontiguousarray(X).astype(np.float32)
    y = y.astype(np.int64)
    return X, y


@pytest.fixture
def trained_model(dummy_data):
    """Create and train a dummy model."""
    X, y = dummy_data
    # Initialize with basic parameters
    model = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(step_size=10, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="sparsemax",  # strictly enforcing a standard mask
        verbose=0,
    )

    # Fit with explicit float validation
    model.fit(
        X,
        y,
        max_epochs=1,
        patience=1,
        batch_size=32,
        virtual_batch_size=16,
        num_workers=0,
        drop_last=False,
    )
    return model


def test_evaluate_model(trained_model, dummy_data):
    """Test model evaluation metrics."""
    X, y = dummy_data

    # Test without verbose
    metrics = evaluate_model(trained_model, X, y, verbose=False)

    assert isinstance(metrics, dict), "Metrics should be returned as a dictionary"
    assert "auc" in metrics, "Metrics should include AUC"
    assert 0.0 <= metrics["auc"] <= 1.0, "AUC should be between 0 and 1"
    assert "confusion_matrix" in metrics, "Metrics should include confusion matrix"
    assert metrics["proba"].shape[0] == X.shape[0], (
        "Probability predictions should match number of samples"
    )
    assert metrics["preds"].shape[0] == X.shape[0], "Predictions should match number of samples"


def test_feature_importance(trained_model, dummy_data):
    """Test feature importance calculation."""
    X, _ = dummy_data
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    importance_df = get_feature_importance(trained_model, feature_names)

    assert importance_df is not None, "Feature importance should not be None"
    assert len(importance_df) == X.shape[1], (
        "Feature importance should have an entry for each feature"
    )
    assert "importance" in importance_df.columns, (
        "Feature importance DataFrame should have 'Importance' column"
    )
    # Check if importance is numeric
    assert np.issubdtype(importance_df["importance"].dtype, np.number), (
        "Importance values should be numeric"
    )


def test_evaluate_model_integration(trained_model, dummy_data):
    """Test evaluate_model with feature columns (integration check)."""
    X, y = dummy_data
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    metrics = evaluate_model(trained_model, X, y, feature_columns=feature_names, verbose=False)

    assert "feature_importance" in metrics, "Metrics should include feature importance"
    assert metrics["feature_importance"] is not None, "Feature importance should not be None"
    assert len(metrics["feature_importance"]) == X.shape[1], (
        "Feature importance should have an entry for each feature"
    )
