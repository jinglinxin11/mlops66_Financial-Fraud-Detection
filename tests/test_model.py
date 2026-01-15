import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch # For mocking dependencies

from src.models.tabnet_trainer import TabNetTrainer
from src.utils.helpers import find_latest_checkpoint

class MockConfig:
    """Mock configuration for testing."""
    def get_best_device(self):
        if torch.cuda.is_available(): # Check for CUDA availability
            return 'cuda'
        elif torch.backends.mps.is_available(): # Check for MPS availability (macOS [M1/M2/M3])
            return 'mps'
        else:
            return 'cpu' # Fallback to CPU
        
    CAT_EMB_DIM = 1
    N_D = 8
    N_A = 8
    N_STEPS = 3
    GAMMA = 1.3
    LAMBDA_SPARSE = 1e-5
    LEARNING_RATE = 2e-2
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.9
    MASK_TYPE = 'sparsemax'
    DEVICE = get_best_device() # Adaptive device (should handle both Linux/Windows + macOS)
    MAX_EPOCHS = 5
    BATCH_SIZE = 32
    CHECKPOINT_DIR = 'checkpoints'


@pytest.fixture
def dummy_data():
    """Create dummy preprocessed data for testing."""
    X = np.random.rand(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=(100,))
    return {
        'X_train': X[:80],
        'y_train': y[:80],
        'X_valid': X[80:],
        'y_valid': y[80:],
        'cat_idxs': [],
        'cat_dims': []
    }

@pytest.fixture
def trainer(dummy_data):
    """Fixture for TabNetTrainer with mock config and dummy data."""
    config = MockConfig()
    return TabNetTrainer(config, dummy_data, verbose=False)

def test_init(trainer):
    assert trainer.model is None
    assert trainer.config.CAT_EMB_DIM == 1
    assert trainer.data['X_train'].shape == (80, 10)
    assert trainer.data['y_train'].shape == (80,)
    assert trainer.data['X_valid'].shape == (20, 10)
    assert trainer.data['y_valid'].shape == (20,)
    assert trainer.data['cat_idxs'] == []
    assert trainer.data['cat_dims'] == []


def test_create_model(trainer):
    model = trainer._create_model()
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    assert model.device == trainer.config.DEVICE

def test_train(trainer):
    with patch.object(trainer, '_create_model', wraps=trainer._create_model) as mock_create_model:
        trained_model = trainer.train()
        assert trained_model is not None
        assert isinstance(trained_model, torch.nn.Module)
        mock_create_model.assert_called_once()
        checkpoint_dir = Path(trainer.config.CHECKPOINT_DIR) / 'tabnet_model_final.zip'
        assert checkpoint_dir.exists(), f"Checkpoint not found at {checkpoint_dir}"

        if trainer.config.RESUME_TRAINING:
            checkpoint_path, last_epoch = find_latest_checkpoint(str(trainer.config.CHECKPOINT_DIR))
            if checkpoint_path:
                assert last_epoch < trainer.config.MAX_EPOCHS
                assert trained_model is not None
        else:
            assert trained_model is not None

def test_log_function(trainer, capsys):
    message = "Test log message"
    assert message not in capsys.readouterr().out
    assert trainer.config.verbose == True or trainer.config.verbose == False
    trainer._log(message)
    captured = capsys.readouterr()
    assert message in captured.out


        


            