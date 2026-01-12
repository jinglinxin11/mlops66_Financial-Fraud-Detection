"""
Uncertainty Analysis Module
Analyze prediction confidence and uncertainty
"""

import numpy as np


class UncertaintyAnalyzer:
    """Uncertainty Analyzer"""
    
    def __init__(self, thresholds=None):
        """
        Args:
            thresholds: Risk threshold dictionary
        """
        self.thresholds = thresholds or {
            'high_risk': 0.8,
            'medium_high': 0.5,
            'medium_low': 0.3,
            'low_risk': 0.1
        }
    
    def analyze(self, proba, y_true=None, verbose=True):
        """
        Analyze prediction uncertainty
        
        Args:
            proba: Prediction probabilities
            y_true: True labels (optional)
            verbose: Whether to print detailed information
            
        Returns:
            dict: Analysis results
        """
        if verbose:
            print("\n" + "=" * 60)
            print("              Uncertainty Analysis")
            print("=" * 60)
        
        # Stratified statistics
        high_risk = proba >= self.thresholds['high_risk']
        medium_risk = (proba >= self.thresholds['medium_low']) & (proba < self.thresholds['high_risk'])
        low_risk = proba < self.thresholds['medium_low']
        uncertain = (proba >= 0.3) & (proba <= 0.7)
        
        if verbose:
            print("\nPrediction Confidence Stratification:")
            print("-" * 50)
            
            layers = [
                ('High Risk (>=0.8)', high_risk),
                ('Medium Risk (0.3-0.8)', medium_risk),
                ('Low Risk (<0.3)', low_risk),
                ('Uncertain Range (0.3-0.7)', uncertain)
            ]
            
            for name, mask in layers:
                count = mask.sum()
                if y_true is not None and count > 0:
                    actual_fraud = y_true[mask].sum()
                    precision = actual_fraud / count
                    print(f"{name}:")
                    print(f"  Samples: {count:,} ({count/len(proba)*100:.1f}%)")
                    print(f"  Actual Fraud: {actual_fraud:,} (Precision: {precision*100:.1f}%)")
                else:
                    print(f"{name}:")
                    print(f"  Samples: {count:,} ({count/len(proba)*100:.1f}%)")
        
        # Threshold analysis
        if y_true is not None and verbose:
            self._print_threshold_analysis(proba, y_true)
        
        return {
            'high_risk_count': int(high_risk.sum()),
            'medium_risk_count': int(medium_risk.sum()),
            'low_risk_count': int(low_risk.sum()),
            'uncertain_count': int(uncertain.sum()),
        }
    
    def _print_threshold_analysis(self, proba, y_true):
        """Print performance at different thresholds"""
        print("\nPerformance at Different Thresholds:")
        print("-" * 50)
        print(f"{'Threshold':<10} {'Pred Fraud':<12} {'Precision':<10} {'Recall':<10}")
        
        for threshold in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]:
            pred = (proba >= threshold).astype(int)
            tp = ((pred == 1) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"{threshold:<10} {(tp+fp):<12,} {precision*100:>7.1f}%   {recall*100:>7.1f}%")
    
    def get_risk_level(self, proba):
        """
        Get risk level for each sample
        
        Args:
            proba: Prediction probabilities
            
        Returns:
            np.ndarray: Risk level array
        """
        risk_levels = np.where(
            proba >= self.thresholds['high_risk'], 'HIGH',
            np.where(
                proba >= self.thresholds['medium_low'], 'MEDIUM',
                'LOW'
            )
        )
        return risk_levels
