"""Configuration Module.

Centralized management of all configuration parameters.
"""

from pathlib import Path

import torch


class Config:
    """Project configuration class."""

    # ============================================================
    # Path Configuration
    # ============================================================

    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # Data paths
    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_TRANSACTION = DATA_DIR / "train_transaction.csv"
    TRAIN_IDENTITY = DATA_DIR / "train_identity.csv"
    TEST_TRANSACTION = DATA_DIR / "test_transaction.csv"
    TEST_IDENTITY = DATA_DIR / "test_identity.csv"

    # Output paths
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    MODEL_PATH = PROJECT_ROOT / "tabnet_fraud_model"
    PREPROCESSOR_PATH = PROJECT_ROOT / "ieee_cis_preprocessor.pkl"
    SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

    # ============================================================
    # Preprocessing Parameters
    # ============================================================

    MISSING_THRESHOLD = 0.9  # Missing rate threshold, columns above this will be dropped
    RARE_CATEGORY_THRESHOLD = (
        100  # Rare category threshold, categories with count below this will be merged
    )
    USE_TIME_SPLIT = True  # Whether to split dataset by time

    # ============================================================
    # TabNet Model Parameters
    # ============================================================

    N_D = 48  # Decision layer width
    N_A = 48  # Attention layer width
    N_STEPS = 4  # Number of decision steps
    GAMMA = 1.5  # Feature reuse coefficient
    CAT_EMB_DIM = 2  # Categorical embedding dimension
    LAMBDA_SPARSE = 1e-3  # Sparsity regularization
    MASK_TYPE = "entmax"  # Mask type

    # ============================================================
    # Training Parameters
    # ============================================================

    MAX_EPOCHS = 100
    PATIENCE = 10
    BATCH_SIZE = 8192
    VIRTUAL_BATCH_SIZE = 256
    LEARNING_RATE = 5e-3

    # Learning rate scheduler parameters
    SCHEDULER_STEP_SIZE = 50
    SCHEDULER_GAMMA = 0.9

    # ============================================================
    # Checkpoint Parameters
    # ============================================================

    CHECKPOINT_EVERY = 10  # Save checkpoint every N epochs
    RESUME_TRAINING = True  # Whether to resume training from checkpoint

    # ============================================================
    # Device Configuration
    # ============================================================

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # Uncertainty Analysis Thresholds
    # ============================================================

    UNCERTAINTY_THRESHOLDS = {
        "high_risk": 0.8,
        "medium_high": 0.5,
        "medium_low": 0.3,
        "low_risk": 0.1,
    }

    def __init__(self, **kwargs):
        """Allow overriding default configuration via keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")

    def __repr__(self):
        """String representation of the configuration."""
        return f"Config(DEVICE={self.DEVICE}, MAX_EPOCHS={self.MAX_EPOCHS})"

    def to_dict(self):
        """Return configuration as dictionary."""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_") and key.isupper()
        }
