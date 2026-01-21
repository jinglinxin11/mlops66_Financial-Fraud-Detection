"""
IEEE-CIS Fraud Detection - TabNet Training Entry Script

Usage:
    python train.py                          # Run with default config
    python train.py training.max_epochs=50   # Override config
    python train.py --help                   # Show Hydra help
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.logger import get_logger, setup_logging
from src.utils.wandb_utils import finish_wandb, init_wandb, log_metrics, log_model

logger = get_logger(__name__)


def get_device(device_config: str) -> str:
    """Determine device from config."""
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


class HydraConfig:
    """Adapter class to make Hydra config compatible with existing code."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Path Configuration
        self.PROJECT_ROOT = Path.cwd()
        self.DATA_DIR = self.PROJECT_ROOT / cfg.paths.data_dir
        self.TRAIN_TRANSACTION = self.PROJECT_ROOT / cfg.paths.train_transaction
        self.TRAIN_IDENTITY = self.PROJECT_ROOT / cfg.paths.train_identity
        self.TEST_TRANSACTION = self.PROJECT_ROOT / cfg.paths.test_transaction
        self.TEST_IDENTITY = self.PROJECT_ROOT / cfg.paths.test_identity
        self.CHECKPOINT_DIR = self.PROJECT_ROOT / cfg.paths.checkpoint_dir
        self.MODEL_PATH = self.PROJECT_ROOT / cfg.paths.model_path
        self.PREPROCESSOR_PATH = self.PROJECT_ROOT / cfg.paths.preprocessor_path
        self.SUBMISSION_PATH = self.PROJECT_ROOT / cfg.paths.submission_path

        # Preprocessing Parameters
        self.MISSING_THRESHOLD = cfg.preprocessing.missing_threshold
        self.RARE_CATEGORY_THRESHOLD = cfg.preprocessing.rare_category_threshold
        self.USE_TIME_SPLIT = cfg.preprocessing.use_time_split

        # TabNet Model Parameters
        self.N_D = cfg.model.n_d
        self.N_A = cfg.model.n_a
        self.N_STEPS = cfg.model.n_steps
        self.GAMMA = cfg.model.gamma
        self.CAT_EMB_DIM = cfg.model.cat_emb_dim
        self.LAMBDA_SPARSE = cfg.model.lambda_sparse
        self.MASK_TYPE = cfg.model.mask_type

        # Training Parameters
        self.MAX_EPOCHS = cfg.training.max_epochs
        self.PATIENCE = cfg.training.patience
        self.BATCH_SIZE = cfg.training.batch_size
        self.VIRTUAL_BATCH_SIZE = cfg.training.virtual_batch_size
        self.LEARNING_RATE = cfg.training.learning_rate
        self.SCHEDULER_STEP_SIZE = cfg.training.scheduler.step_size
        self.SCHEDULER_GAMMA = cfg.training.scheduler.gamma

        # Checkpoint Parameters
        self.CHECKPOINT_EVERY = cfg.training.checkpoint_every
        self.RESUME_TRAINING = cfg.training.resume_training

        # Device
        self.DEVICE = get_device(cfg.training.device)

        # Uncertainty Thresholds
        self.UNCERTAINTY_THRESHOLDS = OmegaConf.to_container(cfg.uncertainty)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main training workflow with Hydra configuration."""

    # Setup logging
    setup_logging(cfg)

    logger.info("\n" + "=" * 60)
    logger.info("     IEEE-CIS Fraud Detection - TabNet Training")
    logger.info("=" * 60)

    # Log configuration
    logger.info(f"\n📋 Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create config adapter
    config = HydraConfig(cfg)

    logger.info("\n⚙️ Settings:")
    logger.info(f"   Device: {config.DEVICE}")
    logger.info(f"   Checkpoint directory: {config.CHECKPOINT_DIR}")
    logger.info(f"   Save checkpoint every {config.CHECKPOINT_EVERY} epochs")
    logger.info(f"   Resume from checkpoint: {config.RESUME_TRAINING}")

    # Initialize Weights & Biases
    wandb_enabled = init_wandb(cfg)

    # Import here to avoid circular imports
    from src.evaluation.metrics import evaluate_model
    from src.evaluation.uncertainty import UncertaintyAnalyzer
    from src.features.preprocessor import FraudPreprocessor
    from src.models.tabnet_trainer import TabNetTrainer

    # 1. Data preprocessing
    logger.info("\n" + "=" * 60)
    logger.info("              1. Data Preprocessing")
    logger.info("=" * 60)

    preprocessor = FraudPreprocessor(config)
    data = preprocessor.fit_transform()
    preprocessor.save()

    # 2. Train model
    logger.info("\n" + "=" * 60)
    logger.info("              2. Model Training")
    logger.info("=" * 60)

    trainer = TabNetTrainer(config, data)
    model = trainer.train()

    # 3. Evaluate model
    logger.info("\n" + "=" * 60)
    logger.info("              3. Model Evaluation")
    logger.info("=" * 60)

    results = evaluate_model(
        model=model,
        X_test=data["X_test"],
        y_test=data["y_test"],
        feature_columns=data["feature_columns"],
    )

    # Log metrics to wandb
    if wandb_enabled:
        log_metrics(
            {
                "test_auc": results["auc"],
                "test_accuracy": results.get("accuracy", 0),
            }
        )

        # Log model artifact
        model_path = str(config.MODEL_PATH) + ".zip"
        if Path(model_path).exists():
            log_model(model_path, name="tabnet_model")

    # 4. Uncertainty analysis
    analyzer = UncertaintyAnalyzer(config.UNCERTAINTY_THRESHOLDS)
    analyzer.analyze(results["proba"], data["y_test"])

    # 5. Complete
    logger.info("\n" + "=" * 60)
    logger.info("              ✅ Training Complete!")
    logger.info("=" * 60)
    logger.info(f"\nFinal AUC: {results['auc']:.4f}")
    logger.info(f"Model path: {config.MODEL_PATH}")
    logger.info(f"Preprocessor path: {config.PREPROCESSOR_PATH}")
    logger.info(f"Checkpoint directory: {config.CHECKPOINT_DIR}")

    # Finish wandb
    finish_wandb()

    # Return AUC for hyperparameter optimization
    return results["auc"]


if __name__ == "__main__":
    main()
