"""TabNet Trainer Module.

Encapsulates TabNet model creation, training, saving, and loading logic.
"""

from typing import Any, Dict, Optional

import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from ..utils.helpers import find_latest_checkpoint
from ..utils.logger import get_logger
from .callbacks import CheckpointCallback

# Module logger
logger = get_logger(__name__)


class TabNetTrainer:
    """TabNet Model Trainer."""

    def __init__(self, config, data: Dict[str, Any], verbose: bool = True):
        """Initialize TabNetTrainer.

        Args:
        config: Configuration object
        data: Preprocessed data dictionary
        verbose: Whether to print detailed information.
        """
        self.config = config
        self.data = data
        self.verbose = verbose
        self.model: Optional[TabNetClassifier] = None

    def _log(self, message: str):
        """Log message using logging module."""
        if self.verbose:
            logger.info(message)

    def _create_model(self):
        """Create a new TabNet model."""
        return TabNetClassifier(
            cat_idxs=self.data["cat_idxs"],
            cat_dims=self.data["cat_dims"],
            cat_emb_dim=self.config.CAT_EMB_DIM,
            n_d=self.config.N_D,
            n_a=self.config.N_A,
            n_steps=self.config.N_STEPS,
            gamma=self.config.GAMMA,
            lambda_sparse=self.config.LAMBDA_SPARSE,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": self.config.LEARNING_RATE},
            scheduler_params={
                "step_size": self.config.SCHEDULER_STEP_SIZE,
                "gamma": self.config.SCHEDULER_GAMMA,
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type=self.config.MASK_TYPE,
            device_name=self.config.DEVICE,
            verbose=1 if self.verbose else 0,
        )

    def train(self, callbacks=None):
        """Train model (supports resuming from checkpoint).

        Args:
            callbacks: List of additional callbacks (e.g. for Gradio logging)

        Returns:
            TabNetClassifier: Trained model
        """
        self._log("\n" + "=" * 60)
        self._log("              Model Training")
        self._log("=" * 60)

        self._log(f"\nUsing device: {self.config.DEVICE}")
        self._log(f"Train set size: {self.data['X_train'].shape}")
        self._log(f"Valid set size: {self.data['X_valid'].shape}")

        checkpoint_dir = str(self.config.CHECKPOINT_DIR)

        # Check for resumable checkpoints
        if self.config.RESUME_TRAINING:
            checkpoint_path, last_epoch = find_latest_checkpoint(checkpoint_dir)

            if checkpoint_path:
                self._log(f"\n🔄 Found checkpoint: epoch {last_epoch}")
                self._log(f"   Path: {checkpoint_path}")

                # Load model
                self.model = TabNetClassifier()
                self.model.load_model(checkpoint_path)

                # Calculate remaining epochs
                remaining_epochs = self.config.MAX_EPOCHS - last_epoch

                if remaining_epochs <= 0:
                    self._log(
                        f"✅ Training already complete ({last_epoch}/{self.config.MAX_EPOCHS} epochs)"
                    )
                    return self.model

                self._log(f"   Continuing training: {remaining_epochs} epochs remaining")

                # Create checkpoint callback
                checkpoint_callback = CheckpointCallback(
                    save_path=checkpoint_dir, save_every=self.config.CHECKPOINT_EVERY
                )

                # Continue training
                self.model.fit(
                    X_train=self.data["X_train"],
                    y_train=self.data["y_train"],
                    eval_set=[(self.data["X_valid"], self.data["y_valid"])],
                    eval_name=["valid"],
                    eval_metric=["auc"],
                    max_epochs=remaining_epochs,
                    patience=self.config.PATIENCE,
                    batch_size=self.config.BATCH_SIZE,
                    virtual_batch_size=self.config.VIRTUAL_BATCH_SIZE,
                    weights=1,
                    num_workers=0,
                    drop_last=False,
                    callbacks=[checkpoint_callback],
                    warm_start=True,
                )

                # Save final model
                self.save()
                return self.model

        # No checkpoint, train from scratch
        self._log("\n📝 Training from scratch...")

        # Create model
        self.model = self._create_model()

        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_path=checkpoint_dir, save_every=self.config.CHECKPOINT_EVERY
        )

        # Train
        self._log("\nStarting training...")

        
        # Combine callbacks
        final_callbacks = [checkpoint_callback]
        if callbacks:
            final_callbacks.extend(callbacks)

        self.model.fit(
            X_train=self.data["X_train"],
            y_train=self.data["y_train"],
            eval_set=[(self.data["X_valid"], self.data["y_valid"])],
            eval_name=["valid"],
            eval_metric=["auc"],
            max_epochs=self.config.MAX_EPOCHS,
            patience=self.config.PATIENCE,
            batch_size=self.config.BATCH_SIZE,
            virtual_batch_size=self.config.VIRTUAL_BATCH_SIZE,
            weights=1,
            num_workers=0,
            drop_last=False,
            callbacks=final_callbacks,
        )

        # Save final model
        self.save()

        return self.model

    def save(self, path=None):
        """Save model."""
        path = path or str(self.config.MODEL_PATH)
        self.model.save_model(path)
        self._log(f"\nModel saved to: {path}")

    def load(self, path=None):
        """Load model."""
        path = path or str(self.config.MODEL_PATH) + ".zip"
        self.model = TabNetClassifier()
        self.model.load_model(path)
        self._log(f"Model loaded from: {path}")
        return self.model

    def predict(self, X):
        """Predict classes."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
