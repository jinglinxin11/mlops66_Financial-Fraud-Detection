"""Tests for callback functionality."""

import pytest
import os
import pickle
from unittest.mock import MagicMock
from src.models.callbacks import CheckpointCallback

# Mock trainer class to simulate TabNet trainer behavior
class MockTrainer:
    """Mock trainer class to simulate TabNet trainer behavior."""
    def __init__(self):
        """Initialize mock trainer."""
        self.history = MagicMock()
        self.history.history = {'loss': [0.5, 0.4], 'auc': [0.7, 0.8]}
        self.best_cost = 0.35

    def save_model(self, path):
        """Mock save_model method."""
        pass

@pytest.fixture
def callback(tmp_path):
    """Fixture to create a callback instance with a temp dir."""
    save_path = tmp_path / "checkpoints"
    # Create instance with save_every=2 for easier testing
    cb = CheckpointCallback(str(save_path), save_every=2)
    
    # Inject mock trainer as if it were attached by pytorch-tabnet
    cb.trainer = MockTrainer() 
    # Mock the save_model method specifically to track calls
    cb.trainer.save_model = MagicMock()
    
    return cb

def test_init_creates_directory(tmp_path):
    """Test that __init__ creates the target directory."""
    subdir = tmp_path / "new_subdir"
    assert not subdir.exists()
    
    CheckpointCallback(str(subdir))
    
    assert subdir.exists(), f"Directory {subdir} should have been created by __init__"

def test_no_save_on_intermediate_epoch(callback):
    """Test that no checkpoint is saved when epoch % save_every != 0."""
    # save_every is 2. Epoch 0 (1st epoch) -> 1 % 2 != 0.
    callback.on_epoch_end(0, logs={})
    
    callback.trainer.save_model.assert_not_called()
    
    # Check that directory is empty (except potentially hidden files if any, but scandir handles that)
    files = list(os.scandir(callback.save_path))
    assert len(files) == 0, \
        f"Save directory should be empty for intermediate epochs, found: {[f.name for f in files]}"

def test_save_on_correct_epoch(callback):
    """Test that checkpoint is saved when epoch % save_every == 0."""
    # save_every is 2. Epoch 1 (2nd epoch) -> 2 % 2 == 0.
    callback.on_epoch_end(1, logs={})
    
    # 1. Check model save called with correct path
    expected_model_path = os.path.join(callback.save_path, "checkpoint_epoch_2")
    callback.trainer.save_model.assert_called_once_with(expected_model_path)
    
    # 2. Check state file created
    expected_state_file = os.path.join(callback.save_path, "training_state_epoch_2.pkl")
    assert os.path.exists(expected_state_file), \
        f"State file {expected_state_file} was not created"
        
    # 3. Check state content
    with open(expected_state_file, 'rb') as f:
        state = pickle.load(f)
    
    assert state['epoch'] == 2, \
        f"Expected epoch 2 in state file, got {state.get('epoch')}"
    assert state['best_cost'] == 0.35, \
        f"Expected best_cost 0.35, got {state.get('best_cost')}"
    assert 'history' in state, "History should be preserved in state"
    assert state['history'] == {'loss': [0.5, 0.4], 'auc': [0.7, 0.8]}, \
        "History content mismatch"

def test_save_sequence(callback):
    """Test a sequence of epochs to ensure it saves periodically."""
    # save_every=2
    
    # Epoch 1 (0): No save
    callback.on_epoch_end(0)
    assert callback.trainer.save_model.call_count == 0, \
        "Should not save on epoch 1 (index 0)"
    
    # Epoch 2 (1): Save
    callback.on_epoch_end(1)
    assert callback.trainer.save_model.call_count == 1, \
        "Should save on epoch 2 (index 1)"
    
    # Epoch 3 (2): No save
    callback.on_epoch_end(2)
    assert callback.trainer.save_model.call_count == 1, \
        "Should not save on epoch 3 (index 2)"
    
    # Epoch 4 (3): Save
    callback.on_epoch_end(3)
    assert callback.trainer.save_model.call_count == 2, \
        "Should save on epoch 4 (index 3)"
