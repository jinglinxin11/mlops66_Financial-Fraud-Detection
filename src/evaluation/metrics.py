"""Evaluation Metrics Module
Calculate model performance metrics.
"""

import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test, feature_columns=None, verbose=True):
    """Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_columns: Feature column names (for feature importance)
        verbose: Whether to print detailed information
        
    Returns:
        dict: Evaluation results
    """
    if verbose:
        print("\n" + "=" * 60)
        print("              Model Evaluation")
        print("=" * 60)
    
    # Predict
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    
    # Calculate metrics
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)
    
    if verbose:
        print(f"\nTest AUC: {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"              Pred Normal  Pred Fraud")
        print(f"Actual Normal   {cm[0,0]:8d}    {cm[0,1]:8d}")
        print(f"Actual Fraud    {cm[1,0]:8d}    {cm[1,1]:8d}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, preds, target_names=['Normal', 'Fraud']))
    
    # Feature importance
    feature_importance = None
    if feature_columns is not None:
        feature_importance = get_feature_importance(model, feature_columns)
        if verbose:
            print("\nFeature Importance Top 10:")
            print(feature_importance.head(10).to_string(index=False))
    
    return {
        'auc': auc,
        'proba': proba,
        'preds': preds,
        'confusion_matrix': cm,
        'feature_importance': feature_importance
    }


def get_feature_importance(model, feature_columns):
    """Get feature importance.
    
    Args:
        model: Trained model
        feature_columns: Feature column names
        
    Returns:
        pd.DataFrame: Feature importance table sorted by importance
    """
    return pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
