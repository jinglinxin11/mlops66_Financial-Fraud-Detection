"""IEEE-CIS Fraud Detection - TabNet Training Entry Script.

Usage:
    python train.py
"""

import warnings
warnings.filterwarnings('ignore')

from src.config.settings import Config
from src.features.preprocessor import FraudPreprocessor
from src.models.tabnet_trainer import TabNetTrainer
from src.evaluation.metrics import evaluate_model
from src.evaluation.uncertainty import UncertaintyAnalyzer


def main():
    """Main training workflow."""
    print("\n" + "=" * 60)
    print("     IEEE-CIS Fraud Detection - TabNet Training")
    print("=" * 60)
    
    # 1. Load configuration
    config = Config()
    
    print(f"\n⚙️ Configuration:")
    print(f"   Device: {config.DEVICE}")
    print(f"   Checkpoint directory: {config.CHECKPOINT_DIR}")
    print(f"   Save checkpoint every {config.CHECKPOINT_EVERY} epochs")
    print(f"   Resume from checkpoint: {config.RESUME_TRAINING}")
    
    # 2. Data preprocessing
    print("\n" + "=" * 60)
    print("              1. Data Preprocessing")
    print("=" * 60)
    
    preprocessor = FraudPreprocessor(config)
    data = preprocessor.fit_transform()
    
    # Save preprocessor
    preprocessor.save()
    
    # 3. Train model
    print("\n" + "=" * 60)
    print("              2. Model Training")
    print("=" * 60)
    
    trainer = TabNetTrainer(config, data)
    model = trainer.train()
    
    # 4. Evaluate model
    print("\n" + "=" * 60)
    print("              3. Model Evaluation")
    print("=" * 60)
    
    results = evaluate_model(
        model=model,
        X_test=data['X_test'],
        y_test=data['y_test'],
        feature_columns=data['feature_columns']
    )
    
    # 5. Uncertainty analysis
    analyzer = UncertaintyAnalyzer(config.UNCERTAINTY_THRESHOLDS)
    uncertainty = analyzer.analyze(results['proba'], data['y_test'])
    
    # 6. Complete
    print("\n" + "=" * 60)
    print("              ✅ Training Complete!")
    print("=" * 60)
    print(f"\nFinal AUC: {results['auc']:.4f}")
    print(f"Model path: {config.MODEL_PATH}")
    print(f"Preprocessor path: {config.PREPROCESSOR_PATH}")
    print(f"Checkpoint directory: {config.CHECKPOINT_DIR}")
    
    return model, data, results


if __name__ == "__main__":
    model, data, results = main()
