"""IEEE-CIS Fraud Detection - Prediction Entry Script.

Usage:
    python predict.py

Note:
    Test set has no labels, cannot evaluate locally!
    Must upload to Kaggle to get the score.
"""

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from src.config.settings import Config
from src.features.preprocessor import FraudPreprocessor
from src.models.tabnet_trainer import TabNetTrainer


def predict_test():
    """Make predictions on test set."""
    print("\n" + "=" * 60)
    print("     Predicting on Test Set")
    print("=" * 60)

    # Load configuration
    config = Config()

    # Check if files exist
    model_path = Path(str(config.MODEL_PATH) + ".zip")

    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please run train.py first to train the model")
        return None

    if not config.PREPROCESSOR_PATH.exists():
        print(f"❌ Preprocessor not found: {config.PREPROCESSOR_PATH}")
        return None

    if not config.TEST_TRANSACTION.exists():
        print(f"❌ Test data not found: {config.TEST_TRANSACTION}")
        return None

    # 1. Load preprocessor
    print("\n1. Loading preprocessor...")
    preprocessor = FraudPreprocessor(config)
    preprocessor.load()

    # 2. Load model
    print("\n2. Loading model...")
    trainer = TabNetTrainer(config, data=None)
    model = trainer.load()
    print("✅ Model loaded successfully")

    # 3. Process test data
    print("\n3. Processing test data...")
    test_data = preprocessor.transform()

    # 4. Predict
    print("\n4. Making predictions...")
    proba = model.predict_proba(test_data["X_test"])[:, 1]

    # 5. Generate submission file
    print("\n5. Generating submission file...")
    submission = pd.DataFrame(
        {"TransactionID": test_data["transaction_ids"], "isFraud": proba}
    )

    submission.to_csv(config.SUBMISSION_PATH, index=False)

    # 6. Display statistics
    print("\n" + "=" * 60)
    print("              Prediction Complete!")
    print("=" * 60)
    print(f"\nSubmission file: {config.SUBMISSION_PATH}")
    print(f"Sample count: {len(submission):,}")

    print("\nProbability distribution:")
    print(submission["isFraud"].describe())

    print(f"\nTransactions predicted as high risk (>=0.5): {(proba >= 0.5).sum():,}")
    print(f"Transactions predicted as high risk (>=0.8): {(proba >= 0.8).sum():,}")

    # Visualize probability distribution
    print("\nProbability distribution:")
    print(f"  [0.0-0.1]: {((proba >= 0.0) & (proba < 0.1)).sum():,}")
    print(f"  [0.1-0.3]: {((proba >= 0.1) & (proba < 0.3)).sum():,}")
    print(f"  [0.3-0.5]: {((proba >= 0.3) & (proba < 0.5)).sum():,}")
    print(f"  [0.5-0.7]: {((proba >= 0.5) & (proba < 0.7)).sum():,}")
    print(f"  [0.7-0.9]: {((proba >= 0.7) & (proba < 0.9)).sum():,}")
    print(f"  [0.9-1.0]: {(proba >= 0.9).sum():,}")

    print("\n" + "=" * 60)
    print("⚠️  Note: Test set has no labels, cannot evaluate AUC locally!")
    print("    To get the actual score, upload submission.csv to Kaggle:")
    print("    https://www.kaggle.com/c/ieee-fraud-detection/submit")
    print("=" * 60)

    return submission


if __name__ == "__main__":
    submission = predict_test()
