"""Tests for UncertaintyAnalyzer in src.evaluation.uncertainty."""

import pytest
import numpy as np
from src.evaluation.uncertainty import UncertaintyAnalyzer


@pytest.fixture
def default_analyzer():
    """Fixture for UncertaintyAnalyzer with default values."""
    return UncertaintyAnalyzer()


@pytest.fixture
def predictions():
    """Fixture for sample probability predictions."""
    # Values chosen to hit all risk buckets:
    # High: >= 0.8 (0.95, 0.85)
    # Medium: 0.3 <= p < 0.8 (0.60, 0.40)
    # Low: < 0.3 (0.10, 0.05)
    # Uncertain: 0.3 <= p <= 0.7 (0.60, 0.40)
    return np.array([0.95, 0.85, 0.60, 0.40, 0.10, 0.05])


@pytest.fixture
def labels():
    """Fixture for true labels corresponding to predictions."""
    return np.array([1, 1, 0, 1, 0, 0])


def test_initialization():
    """Test initialization with default and custom thresholds."""
    # Default
    analyzer = UncertaintyAnalyzer()
    assert 'high_risk' in analyzer.thresholds
    assert analyzer.thresholds['high_risk'] == 0.8, "Default high_risk threshold should be 0.8"

    # Custom
    custom_thresholds = {'high_risk': 0.9, 'medium_low': 0.2}
    analyzer_custom = UncertaintyAnalyzer(thresholds=custom_thresholds)
    assert analyzer_custom.thresholds['high_risk'] == 0.9, "Custom threshold not set correctly"


def test_analyze_structure(default_analyzer, predictions):
    """Test the structure of the returned analysis dictionary."""
    result = default_analyzer.analyze(predictions, verbose=False)
    
    expected_keys = ['high_risk_count', 'medium_risk_count', 'low_risk_count', 'uncertain_count']
    for key in expected_keys:
        assert key in result, f"Result dictionary missing key: {key}"
        assert isinstance(result[key], int), f"Value for {key} should be an integer"


def test_analyze_counts(default_analyzer, predictions):
    """Test correct counting of samples in risk categories."""
    # Based on fixture:
    # High: 2
    # Medium: 2
    # Low: 2
    # Uncertain: 2
    
    result = default_analyzer.analyze(predictions, verbose=False)
    
    assert result['high_risk_count'] == 2, f"Expected 2 high risk samples, got {result['high_risk_count']}"
    assert result['medium_risk_count'] == 2, f"Expected 2 medium risk samples, got {result['medium_risk_count']}"
    assert result['low_risk_count'] == 2, f"Expected 2 low risk samples, got {result['low_risk_count']}"
    assert result['uncertain_count'] == 2, f"Expected 2 uncertain samples, got {result['uncertain_count']}"


def test_analyze_with_labels_verbose(default_analyzer, predictions, labels, capsys):
    """Test analyze method with labels and verbose output enabled."""
    default_analyzer.analyze(predictions, y_true=labels, verbose=True)
    
    captured = capsys.readouterr()
    assert "Uncertainty Analysis" in captured.out, "Should print header when verbose=True"
    assert "Performance at Different Thresholds" in captured.out, "Should print threshold analysis when labels provided"
    assert "Prediction Confidence Stratification" in captured.out, "Should print stratification details"


def test_analyze_no_labels_verbose(default_analyzer, predictions, capsys):
    """Test analyze method without labels but with verbose output."""
    default_analyzer.analyze(predictions, y_true=None, verbose=True)
    
    captured = capsys.readouterr()
    assert "Uncertainty Analysis" in captured.out
    assert "Performance at Different Thresholds" not in captured.out, "Should NOT print threshold analysis without labels"


def test_get_risk_level(default_analyzer, predictions):
    """Test risk level categorization."""
    # predictions = [0.95, 0.85, 0.60, 0.40, 0.10, 0.05]
    # Thresholds: High >= 0.8, Medium >= 0.3, else Low
    expected_levels = ['HIGH', 'HIGH', 'MEDIUM', 'MEDIUM', 'LOW', 'LOW']
    
    risk_levels = default_analyzer.get_risk_level(predictions)
    
    assert isinstance(risk_levels, np.ndarray), "Should return numpy array"
    assert len(risk_levels) == len(predictions), "Should return one level per prediction"
    np.testing.assert_array_equal(risk_levels, expected_levels, err_msg="Risk levels do not match expected values")






