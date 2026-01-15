"""Training Callbacks Module.

Defines callback functions for TabNet training process.
"""

import os
import pickle
from pytorch_tabnet.callbacks import Callback


class CheckpointCallback(Callback):
    """Callback to save checkpoints every N epochs."""
    
    def __init__(self, save_path, save_every=10):
        """Initialize CheckpointCallback.
        
        Args:
            save_path: Checkpoint save directory
            save_every: Save checkpoint every N epochs.
        """
        super().__init__()
        self.save_path = save_path
        self.save_every = save_every
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        """Check if checkpoint should be saved at end of each epoch."""
        # epoch is 0-indexed, so epoch+1 is the actual epoch number
        actual_epoch = epoch + 1
        
        if actual_epoch % self.save_every == 0:
            # Save model checkpoint
            checkpoint_path = os.path.join(
                self.save_path, 
                f"checkpoint_epoch_{actual_epoch}"
            )
            self.trainer.save_model(checkpoint_path)
            
            # Save training state (epoch, history, etc.)
            state_path = os.path.join(
                self.save_path,
                f"training_state_epoch_{actual_epoch}.pkl"
            )
            training_state = {
                'epoch': actual_epoch,
                'history': dict(self.trainer.history.history) if hasattr(self.trainer, 'history') else {},
                'best_cost': self.trainer.best_cost if hasattr(self.trainer, 'best_cost') else None,
            }
            with open(state_path, 'wb') as f:
                pickle.dump(training_state, f)
            
            print(f"\n💾 Checkpoint saved: epoch {actual_epoch}")
