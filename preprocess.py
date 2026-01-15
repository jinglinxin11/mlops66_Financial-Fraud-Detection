"""IEEE-CIS Fraud Detection - Data Preprocessing Entry Script.

Usage:
    python preprocess.py              # Preprocess training data
    python preprocess.py --test       # Preprocess test data (for Kaggle submission)
    python preprocess.py --analyze    # Analyze data quality only
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

from src.config.settings import Config
from src.data.loader import DataLoader
from src.features.preprocessor import FraudPreprocessor


def analyze_data(config):
    """Analyze data quality only."""
    print("\n" + "=" * 60)
    print("     IEEE-CIS Fraud Detection - Data Quality Analysis")
    print("=" * 60)
    
    loader = DataLoader()
    df = loader.load_and_merge(
        config.TRAIN_TRANSACTION,
        config.TRAIN_IDENTITY
    )
    loader.analyze(df)
    
    print("\n✅ Analysis complete!")
    return df


def preprocess_train(config):
    """Preprocess training data."""
    print("\n" + "=" * 60)
    print("     IEEE-CIS Fraud Detection - Training Data Preprocessing")
    print("=" * 60)
    
    preprocessor = FraudPreprocessor(config)
    data = preprocessor.fit_transform()
    preprocessor.save()
    
    print("\n" + "=" * 60)
    print("              ✅ Preprocessing Complete!")
    print("=" * 60)
    print(f"\nTraining set shape: {data['X_train'].shape}")
    print(f"Validation set shape: {data['X_valid'].shape}")
    print(f"Test set shape: {data['X_test'].shape}")
    print(f"Number of features: {len(data['feature_columns'])}")
    print(f"Categorical feature indices: {data['cat_idxs'][:5]}... (total {len(data['cat_idxs'])})")
    print(f"\nPreprocessor saved to: {config.PREPROCESSOR_PATH}")
    
    return data, preprocessor


def preprocess_test(config):
    """Preprocess test data (for Kaggle submission)."""
    print("\n" + "=" * 60)
    print("     IEEE-CIS Fraud Detection - Test Data Preprocessing")
    print("=" * 60)
    
    # Load saved preprocessor
    preprocessor = FraudPreprocessor(config)
    preprocessor.load()
    
    # Process test data
    test_data = preprocessor.transform(
        transaction_path=config.TEST_TRANSACTION,
        identity_path=config.TEST_IDENTITY
    )
    
    print("\n" + "=" * 60)
    print("              ✅ Test Data Preprocessing Complete!")
    print("=" * 60)
    print(f"\nTest set shape: {test_data['X_test'].shape}")
    print(f"Transaction ID count: {len(test_data['transaction_ids'])}")
    
    return test_data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='IEEE-CIS Data Preprocessing Tool')
    parser.add_argument('--test', action='store_true', 
                        help='Preprocess test data (for Kaggle submission)')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze data quality only, no preprocessing')
    args = parser.parse_args()
    
    config = Config()
    
    if args.analyze:
        analyze_data(config)
    elif args.test:
        preprocess_test(config)
    else:
        preprocess_train(config)


if __name__ == "__main__":
    main()
