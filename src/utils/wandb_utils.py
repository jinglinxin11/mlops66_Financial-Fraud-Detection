"""
Weights & Biases Integration
Experiment tracking and logging utilities
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

# Lazy import to avoid import errors if wandb not installed
_wandb = None


def _get_wandb():
    """Lazy load wandb module."""
    global _wandb
    if _wandb is None:
        try:
            import wandb
            _wandb = wandb
        except ImportError:
            logger.warning("wandb not installed. Run: pip install wandb")
            _wandb = False
    return _wandb if _wandb else None


def init_wandb(
    cfg,
    project: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[list] = None,
) -> bool:
    """
    Initialize Weights & Biases run.
    
    Args:
        cfg: Hydra config object
        project: Project name (overrides config)
        name: Run name
        tags: Additional tags
        
    Returns:
        bool: True if initialized successfully
    """
    wandb = _get_wandb()
    if wandb is None:
        return False
    
    # Check if wandb is enabled in config
    if not cfg.wandb.get('enabled', True):
        logger.info("W&B disabled in config")
        return False
    
    try:
        # Prepare config dict for logging
        config_dict = {}
        for key in ['model', 'training', 'preprocessing']:
            if hasattr(cfg, key):
                from omegaconf import OmegaConf
                config_dict[key] = OmegaConf.to_container(cfg[key], resolve=True)
        
        # Initialize run
        wandb.init(
            project=project or cfg.wandb.get('project', 'fraud-detection'),
            entity=cfg.wandb.get('entity', None),
            name=name,
            tags=(tags or []) + cfg.wandb.get('tags', []),
            config=config_dict,
        )
        
        logger.info(f"✅ W&B initialized: {wandb.run.url}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        return False


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Log metrics to W&B.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number
    """
    wandb = _get_wandb()
    if wandb is None or wandb.run is None:
        return
    
    wandb.log(metrics, step=step)


def log_model(model_path: str, name: str = "model"):
    """
    Log model artifact to W&B.
    
    Args:
        model_path: Path to model file
        name: Artifact name
    """
    wandb = _get_wandb()
    if wandb is None or wandb.run is None:
        return
    
    artifact = wandb.Artifact(name, type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    logger.info(f"📦 Model artifact logged: {name}")


def finish_wandb():
    """Finish W&B run."""
    wandb = _get_wandb()
    if wandb is not None and wandb.run is not None:
        wandb.finish()
        logger.info("W&B run finished")


class WandbCallback:
    """
    Callback for logging training progress to W&B.
    
    Compatible with pytorch-tabnet callback interface.
    """
    
    def __init__(self):
        self.wandb = _get_wandb()
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Called at the end of each epoch."""
        if self.wandb is None or self.wandb.run is None:
            return
        
        if logs:
            self.wandb.log({
                'epoch': epoch,
                **logs
            })
    
    def on_train_end(self, logs: Dict[str, float] = None):
        """Called at the end of training."""
        if logs:
            log_metrics({'final_' + k: v for k, v in logs.items()})
