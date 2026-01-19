"""Tests for TabNetTrainer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.tabnet_trainer import TabNetTrainer


class MockConfig:
    """Mock configuration for testing."""

    CAT_EMB_DIM = 2
    N_D = 8
    N_A = 8
    N_STEPS = 3
    GAMMA = 1.3
    LAMBDA_SPARSE = 1e-4
    LEARNING_RATE = 0.02
    SCHEDULER_STEP_SIZE = 50
    SCHEDULER_GAMMA = 0.95
    MASK_TYPE = "sparsemax"
    DEVICE = "cpu"
    MAX_EPOCHS = 100
    BATCH_SIZE = 1024
    VIRTUAL_BATCH_SIZE = 128
    PATIENCE = 15
    CHECKPOINT_EVERY = 10
    CHECKPOINT_DIR = "checkpoints"
    MODEL_PATH = "model_output"
    RESUME_TRAINING = False


@pytest.fixture
def mock_config():
    """Create a mock config."""
    return MockConfig()


@pytest.fixture
def mock_data():
    """Create dummy data dictionary."""
    return {
        "X_train": np.zeros((100, 10)),
        "y_train": np.zeros((100,)),
        "X_valid": np.zeros((20, 10)),
        "y_valid": np.zeros((20,)),
        "cat_idxs": [0, 2],
        "cat_dims": [5, 3],
    }


@pytest.fixture
def trainer(mock_config, mock_data):
    """Create a TabNetTrainer instance."""
    return TabNetTrainer(mock_config, mock_data, verbose=False)


def test_init(trainer, mock_config, mock_data):
    """Test initialization."""
    assert trainer.config == mock_config, "Config object should be correctly stored in trainer"
    assert trainer.data == mock_data, "Data dictionary should be correctly stored in trainer"
    assert trainer.model is None, "Model should be initialized as None before creation"
    assert trainer.verbose is False, "Verbose flag should be set correctly"


@patch("src.models.tabnet_trainer.TabNetClassifier")
def test_create_model_params(MockClassifier, trainer):
    """Test that model is created with correct hyperparameters."""
    trainer._create_model()

    MockClassifier.assert_called_once()
    _, kwargs = MockClassifier.call_args

    # Check key hyperparameters
    assert kwargs["n_d"] == trainer.config.N_D, (
        f"Model initialized with wrong n_d. Expected {trainer.config.N_D}, got {kwargs['n_d']}"
    )
    assert kwargs["n_a"] == trainer.config.N_A, (
        f"Model initialized with wrong n_a. Expected {trainer.config.N_A}, got {kwargs['n_a']}"
    )
    assert kwargs["cat_idxs"] == trainer.data["cat_idxs"], "Model initialized with wrong cat_idxs"
    assert kwargs["cat_dims"] == trainer.data["cat_dims"], "Model initialized with wrong cat_dims"
    expected_mask = trainer.config.MASK_TYPE
    assert kwargs["mask_type"] == expected_mask, (
        f"Model initialized with wrong mask_type. Expected {expected_mask}"
    )


@patch("src.models.tabnet_trainer.TabNetClassifier")
@patch("src.models.tabnet_trainer.CheckpointCallback")
def test_train_from_scratch(MockCallback, MockClassifier, trainer):
    """Test standard training flow from scratch."""
    # Setup mocks
    mock_model = MockClassifier.return_value
    trainer.config.RESUME_TRAINING = False

    # Run training
    returned_model = trainer.train()

    # Assertions
    assert returned_model == mock_model, "Train method should return the model instance"

    # Verify fit was called
    mock_model.fit.assert_called_once()
    call_args = mock_model.fit.call_args[1]

    assert call_args["max_epochs"] == trainer.config.MAX_EPOCHS, (
        f"fit called with wrong max_epochs. Expected {trainer.config.MAX_EPOCHS}"
    )
    assert call_args["batch_size"] == trainer.config.BATCH_SIZE, (
        f"fit called with wrong batch_size. Expected {trainer.config.BATCH_SIZE}"
    )

    # Verify callback creation and usage
    MockCallback.assert_called_with(
        save_path=str(trainer.config.CHECKPOINT_DIR),
        save_every=trainer.config.CHECKPOINT_EVERY,
    )
    assert len(call_args["callbacks"]) > 0, "Callbacks list passed to fit should not be empty"


@patch("src.models.tabnet_trainer.find_latest_checkpoint")
@patch("src.models.tabnet_trainer.TabNetClassifier")
@patch("src.models.tabnet_trainer.CheckpointCallback")
def test_train_resume_success(MockCallback, MockClassifier, mock_find_checkpoint, trainer):
    """Test resuming training from a checkpoint."""
    # Setup
    trainer.config.RESUME_TRAINING = True
    found_epoch = 60
    mock_find_checkpoint.return_value = ("/path/to/ckpt.zip", found_epoch)

    mock_model = MockClassifier.return_value

    # Run
    trainer.train()

    # Verify loading
    mock_model.load_model.assert_called_once_with("/path/to/ckpt.zip")

    # Verify remaining epochs calculation
    # MAX_EPOCHS = 100, found_epoch = 60 -> remaining = 40
    expected_remaining = trainer.config.MAX_EPOCHS - found_epoch

    mock_model.fit.assert_called_once()
    call_args = mock_model.fit.call_args[1]

    actual_epochs = call_args["max_epochs"]
    assert actual_epochs == expected_remaining, (
        f"Resumed training should run for {expected_remaining} epochs, got {actual_epochs}"
    )
    assert call_args["warm_start"] is True, "Resumed training should set warm_start=True"


@patch("src.models.tabnet_trainer.find_latest_checkpoint")
@patch("src.models.tabnet_trainer.TabNetClassifier")
def test_train_resume_already_finished(MockClassifier, mock_find_checkpoint, trainer):
    """Test resuming when max epochs already reached."""
    trainer.config.RESUME_TRAINING = True
    mock_find_checkpoint.return_value = ("/path/to/ckpt.zip", trainer.config.MAX_EPOCHS)

    mock_model = MockClassifier.return_value

    trainer.train()

    # Verify load called but fit NOT called
    mock_model.load_model.assert_called_once()
    mock_model.fit.assert_not_called()


def test_predict_wrappers(trainer):
    """Test availability and delegation of predict methods."""
    mock_model = MagicMock()
    trainer.model = mock_model

    # Test predict
    X_test = np.random.rand(10, 10)
    trainer.predict(X_test)
    mock_model.predict.assert_called_once_with(X_test)

    # Test predict_proba
    trainer.predict_proba(X_test)
    mock_model.predict_proba.assert_called_once_with(X_test)


def test_save_load(trainer):
    """Test save and load methods."""
    mock_model = MagicMock()
    trainer.model = mock_model

    # Test Save
    trainer.save("custom_path.zip")
    mock_model.save_model.assert_called_with("custom_path.zip")

    # Test Load
    # trainer.load creates a NEW model instance, so we need to patch TabNetClassifier inside load
    with patch("src.models.tabnet_trainer.TabNetClassifier") as MockClassifier:
        # Need 'dict' access for MockClassifier().load_model call inside class
        new_mock_model = MockClassifier.return_value

        trainer.load("custom_path.zip")

        MockClassifier.assert_called_once()
        new_mock_model.load_model.assert_called_with("custom_path.zip")
        assert trainer.model == new_mock_model, (
            "Trainer.model should be updated to the loaded model instance"
        )
