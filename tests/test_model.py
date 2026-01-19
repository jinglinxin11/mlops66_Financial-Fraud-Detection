"""
Unit tests for model construction and training.

Tests cover:
- TabNetTrainer class
- Model creation
- Prediction functionality
- Configuration validation
"""

import pytest
import numpy as np
from pathlib import Path

from src.models.tabnet_trainer import TabNetTrainer


class TestTabNetTrainer:
    """Tests for TabNetTrainer class."""
    
    def test_trainer_initialization(self, mock_config, sample_preprocessed_data):
        """Test TabNetTrainer can be initialized."""
        trainer = TabNetTrainer(mock_config, sample_preprocessed_data, verbose=False)
        
        assert trainer is not None
        assert trainer.config == mock_config
        assert trainer.model is None  # Not trained yet
        assert trainer.verbose is False
    
    def test_trainer_create_model(self, mock_config, sample_preprocessed_data):
        """Test TabNetTrainer can create a model."""
        trainer = TabNetTrainer(mock_config, sample_preprocessed_data, verbose=False)
        model = trainer._create_model()
        
        assert model is not None


class TestModelPrediction:
    """Tests for model prediction functionality."""
    
    def test_model_predict_shape(self, mock_config, sample_preprocessed_data):
        """Test model prediction output shape."""
        trainer = TabNetTrainer(mock_config, sample_preprocessed_data, verbose=False)
        
        # Create and fit model with minimal training
        model = trainer._create_model()
        
        # Quick fit (1 epoch)
        model.fit(
            X_train=sample_preprocessed_data['X_train'],
            y_train=sample_preprocessed_data['y_train'],
            eval_set=[(sample_preprocessed_data['X_valid'], sample_preprocessed_data['y_valid'])],
            max_epochs=1,
            patience=1,
            batch_size=32,
            virtual_batch_size=16,
        )
        
        # Test prediction
        predictions = model.predict(sample_preprocessed_data['X_test'])
        
        assert len(predictions) == len(sample_preprocessed_data['X_test'])
        assert set(predictions).issubset({0, 1})
    
    def test_model_predict_proba_shape(self, mock_config, sample_preprocessed_data):
        """Test model probability prediction shape."""
        trainer = TabNetTrainer(mock_config, sample_preprocessed_data, verbose=False)
        model = trainer._create_model()
        
        # Quick fit
        model.fit(
            X_train=sample_preprocessed_data['X_train'],
            y_train=sample_preprocessed_data['y_train'],
            eval_set=[(sample_preprocessed_data['X_valid'], sample_preprocessed_data['y_valid'])],
            max_epochs=1,
            patience=1,
            batch_size=32,
            virtual_batch_size=16,
        )
        
        # Test probability prediction
        proba = model.predict_proba(sample_preprocessed_data['X_test'])
        
        assert proba.shape == (len(sample_preprocessed_data['X_test']), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1


class TestModelConfig:
    """Tests for model configuration."""
    
    def test_config_has_required_attributes(self, mock_config):
        """Test mock config has all required attributes."""
        required_attrs = [
            'N_D', 'N_A', 'N_STEPS', 'GAMMA', 'CAT_EMB_DIM',
            'LAMBDA_SPARSE', 'MASK_TYPE', 'MAX_EPOCHS', 'PATIENCE',
            'BATCH_SIZE', 'VIRTUAL_BATCH_SIZE', 'LEARNING_RATE', 'DEVICE'
        ]
        
        for attr in required_attrs:
            assert hasattr(mock_config, attr), f"Config missing: {attr}"
    
    def test_config_values_are_valid(self, mock_config):
        """Test config values are within valid ranges."""
        assert mock_config.N_D > 0
        assert mock_config.N_A > 0
        assert mock_config.N_STEPS > 0
        assert mock_config.MAX_EPOCHS > 0
        assert mock_config.BATCH_SIZE > 0
        assert 0 < mock_config.LEARNING_RATE < 1


class TestEvaluationMetrics:
    """Tests for evaluation metrics."""
    
    def test_evaluate_model_import(self):
        """Test evaluate_model can be imported."""
        from src.evaluation.metrics import evaluate_model
        assert callable(evaluate_model)
    
    def test_uncertainty_analyzer_import(self):
        """Test UncertaintyAnalyzer can be imported."""
        from src.evaluation.uncertainty import UncertaintyAnalyzer
        analyzer = UncertaintyAnalyzer()
        assert analyzer is not None


class TestCallbacks:
    """Tests for training callbacks."""
    
    def test_checkpoint_callback_import(self):
        """Test CheckpointCallback can be imported."""
        from src.models.callbacks import CheckpointCallback
        callback = CheckpointCallback(save_path="./test_checkpoints", save_every=5)
        assert callback is not None
        assert callback.save_every == 5
