"""IEEE-CIS Fraud Detection - TabNet Training Entry Script.

Usage:
    python train.py [OPTIONS]

Options:
    --max-epochs INT     Maximum training epochs (default: 100)
    --batch-size INT     Batch size (default: 8192)
    --lr FLOAT           Learning rate (default: 0.005)
    --patience INT       Early stopping patience (default: 10)
    --no-resume          Don't resume from checkpoint
"""

import argparse
import logging
import warnings

warnings.filterwarnings("ignore")

from src.config.settings import Config
from src.evaluation.metrics import evaluate_model
from src.evaluation.uncertainty import UncertaintyAnalyzer
from src.features.preprocessor import FraudPreprocessor
from src.models.tabnet_trainer import TabNetTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TabNet model for fraud detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint",
    )
    return parser.parse_args()


def main():
    """Main training workflow."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("     IEEE-CIS Fraud Detection - TabNet Training")
    logger.info("=" * 60)

    # 1. Load configuration with CLI overrides
    config = Config(
        MAX_EPOCHS=args.max_epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        PATIENCE=args.patience,
        RESUME_TRAINING=not args.no_resume,
    )

    logger.info("Configuration:")
    logger.info(f"   Device: {config.DEVICE}")
    logger.info(f"   Max epochs: {config.MAX_EPOCHS}")
    logger.info(f"   Batch size: {config.BATCH_SIZE}")
    logger.info(f"   Learning rate: {config.LEARNING_RATE}")
    logger.info(f"   Checkpoint directory: {config.CHECKPOINT_DIR}")
    logger.info(f"   Resume from checkpoint: {config.RESUME_TRAINING}")

    # 2. Data preprocessing
    logger.info("=" * 60)
    logger.info("              1. Data Preprocessing")
    logger.info("=" * 60)

    preprocessor = FraudPreprocessor(config)
    data = preprocessor.fit_transform()

    # Save preprocessor
    preprocessor.save()
    logger.info("Preprocessor saved")

    # 3. Train model
    logger.info("=" * 60)
    logger.info("              2. Model Training")
    logger.info("=" * 60)

    trainer = TabNetTrainer(config, data)
    model = trainer.train()
    logger.info("Model training complete")

    # 4. Evaluate model
    logger.info("=" * 60)
    logger.info("              3. Model Evaluation")
    logger.info("=" * 60)

    results = evaluate_model(
        model=model,
        X_test=data["X_test"],
        y_test=data["y_test"],
        feature_columns=data["feature_columns"],
    )

    # 5. Uncertainty analysis
    analyzer = UncertaintyAnalyzer(config.UNCERTAINTY_THRESHOLDS)
    analyzer.analyze(results["proba"], data["y_test"])
    logger.info("Uncertainty analysis complete")

    # 6. Complete
    logger.info("=" * 60)
    logger.info("              Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Final AUC: {results['auc']:.4f}")
    logger.info(f"Model path: {config.MODEL_PATH}")
    logger.info(f"Preprocessor path: {config.PREPROCESSOR_PATH}")
    logger.info(f"Checkpoint directory: {config.CHECKPOINT_DIR}")

    return model, data, results


if __name__ == "__main__":
    model, data, results = main()
