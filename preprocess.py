"""
IEEE-CIS Fraud Detection - Data Preprocessing Entry Script

Usage:
    python preprocess.py                  # Preprocess training data
    python preprocess.py --test           # Preprocess test data (for Kaggle submission)
    python preprocess.py --analyze        # Analyze data quality only

Hydra overrides:
    python preprocess.py preprocessing.missing_threshold=0.8
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


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


def analyze_data(config):
    """Analyze data quality only."""
    from src.data.loader import DataLoader

    logger.info("\n" + "=" * 60)
    logger.info("     IEEE-CIS Fraud Detection - Data Quality Analysis")
    logger.info("=" * 60)

    loader = DataLoader()
    df = loader.load_and_merge(config.TRAIN_TRANSACTION, config.TRAIN_IDENTITY)
    loader.analyze(df)

    logger.info("\n✅ Analysis complete!")
    return df


def preprocess_train(config):
    """Preprocess training data."""
    from src.features.preprocessor import FraudPreprocessor

    logger.info("\n" + "=" * 60)
    logger.info("     IEEE-CIS Fraud Detection - Training Data Preprocessing")
    logger.info("=" * 60)

    preprocessor = FraudPreprocessor(config)
    data = preprocessor.fit_transform()
    preprocessor.save()

    logger.info("\n" + "=" * 60)
    logger.info("              ✅ Preprocessing Complete!")
    logger.info("=" * 60)
    logger.info(f"\nTraining set shape: {data['X_train'].shape}")
    logger.info(f"Validation set shape: {data['X_valid'].shape}")
    logger.info(f"Test set shape: {data['X_test'].shape}")
    logger.info(f"Number of features: {len(data['feature_columns'])}")
    logger.info(
        f"Categorical feature indices: {data['cat_idxs'][:5]}... (total {len(data['cat_idxs'])})"
    )
    logger.info(f"\nPreprocessor saved to: {config.PREPROCESSOR_PATH}")

    return data, preprocessor


def preprocess_test(config):
    """Preprocess test data (for Kaggle submission)."""
    from src.features.preprocessor import FraudPreprocessor

    logger.info("\n" + "=" * 60)
    logger.info("     IEEE-CIS Fraud Detection - Test Data Preprocessing")
    logger.info("=" * 60)

    # Load saved preprocessor
    preprocessor = FraudPreprocessor(config)
    preprocessor.load()

    # Process test data
    test_data = preprocessor.transform(
        transaction_path=config.TEST_TRANSACTION, identity_path=config.TEST_IDENTITY
    )

    logger.info("\n" + "=" * 60)
    logger.info("              ✅ Test Data Preprocessing Complete!")
    logger.info("=" * 60)
    logger.info(f"\nTest set shape: {test_data['X_test'].shape}")
    logger.info(f"Transaction ID count: {len(test_data['transaction_ids'])}")

    return test_data


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main preprocessing entry point with Hydra configuration."""

    # Setup logging
    setup_logging(cfg)

    # Parse additional command line arguments (for --test and --analyze flags)
    import sys

    do_test = "--test" in sys.argv
    do_analyze = "--analyze" in sys.argv

    # Create config adapter
    config = HydraConfig(cfg)

    logger.info(f"\n📋 Configuration:\n{OmegaConf.to_yaml(cfg.preprocessing)}")

    if do_analyze:
        analyze_data(config)
    elif do_test:
        preprocess_test(config)
    else:
        preprocess_train(config)


if __name__ == "__main__":
    main()
