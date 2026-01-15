import pytest
from src.evaluation.uncertainty import UncertaintyAnalyzer


def test_uncertainty_analyzer():
    analyzer = UncertaintyAnalyzer()
    proba = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    results = analyzer.analyze(proba, y_true, verbose=False)
    assert results['high_risk_count'] == 3, f"Expected 3 high risk, got {results['high_risk_count']}"
    assert results['medium_risk_count'] == 5, f"Expected 5 medium risk, got {results['medium_risk_count']}"
    assert results['low_risk_count'] == 2, f"Expected 2 low risk, got {results['low_risk_count']}"
    assert results['uncertain_count'] == 5, f"Expected 5 uncertain, got {results['uncertain_count']}"

def test_get_risk_level():
    analyzer = UncertaintyAnalyzer()
    proba = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    risk_levels = analyzer.get_risk_level(proba, y_true, verbose=False)
    assert risk_levels.tolist() == ['LOW', 'LOW', 'LOW', 'LOW', 'MEDIUM', 'MEDIUM', 'MEDIUM', 'HIGH', 'HIGH', 'HIGH'], f"Unexpected risk levels: {risk_levels.tolist()}"
    




        
