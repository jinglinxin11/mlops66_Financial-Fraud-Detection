
import pytest
import pandas as pd
import numpy as np
from src.features.time_features import extract_time_features

@pytest.fixture
def sample_df():
    """Create a sample DataFrame with various time values."""
    # Define seconds for specific scenarios
    # 0 = Day 0, 00:00:00 (Night, Not Weekend, Not Working)
    # 9 * 3600 = Day 0, 09:00:00 (Working starts)
    # 15 * 3600 = Day 0, 15:00:00 (Working)
    # 19 * 3600 = Day 0, 19:00:00 (Not Working)
    # 23 * 3600 = Day 0, 23:00:00 (Not Working)
    # 5 * 86400 = Day 5, 00:00:00 (Weekend start)
    # 6 * 86400 = Day 6 (Weekend)
    
    times = [
        0,              # 00:00:00 Day 0 (Index 0)
        9 * 3600,       # 09:00:00 Day 0 (Index 1)
        15 * 3600,      # 15:00:00 Day 0 (Index 2)
        19 * 3600,      # 19:00:00 Day 0 (Index 3)
        23 * 3600,      # 23:00:00 Day 0 (Index 4)
        5 * 86400,      # 00:00:00 Day 5 (Index 5)
        6 * 86400 + 100 # Day 6 (Index 6)
    ]
    
    return pd.DataFrame({'TransactionDT': times})

def test_extract_time_features_columns_exist(sample_df):
    """Test that all expected columns are created."""
    df = extract_time_features(sample_df.copy(), verbose=False)
    
    expected_cols = [
        'hour', 'dayofweek', 'dayofmonth', 
        'is_working_hour', 'is_weekend', 'is_night'
    ]
    
    for col in expected_cols:
        assert col in df.columns, f"Column '{col}' should be present in the output DataFrame"

def test_missing_time_column():
    """Test behavior when time column is missing."""
    df = pd.DataFrame({'A': [1, 2, 3]})
    processed_df = extract_time_features(df.copy(), time_col='TransactionDT', verbose=False)
    
    assert list(processed_df.columns) == ['A'], \
        f"DataFrame columns should remain unchanged when time column is missing. Found: {processed_df.columns}"

def test_hour_extraction(sample_df):
    """Test accuracy of hour extraction."""
    df = extract_time_features(sample_df.copy(), verbose=False)
    
    # Check specific values
    # 0 -> 0
    assert df.loc[0, 'hour'] == 0, "0 seconds should convert to hour 0"
    # 9 * 3600 -> 9
    assert df.loc[1, 'hour'] == 9, "9*3600 seconds should convert to hour 9"
    # 23 * 3600 -> 23
    assert df.loc[4, 'hour'] == 23, "23*3600 seconds should convert to hour 23"
    
    # Range check
    assert df['hour'].min() >= 0, "Hour values cannot be negative"
    assert df['hour'].max() <= 23, "Hour values cannot exceed 23"

def test_is_working_hour(sample_df):
    """Test is_working_hour logic (9 <= hour <= 18)."""
    df = extract_time_features(sample_df.copy(), verbose=False)
    
    # 9 AM -> Working (1)
    assert df.loc[1, 'is_working_hour'] == 1, \
        f"Hour {df.loc[1, 'hour']} should be considered a working hour"
    
    # 15 PM -> Working (1)
    assert df.loc[2, 'is_working_hour'] == 1, \
        f"Hour {df.loc[2, 'hour']} should be considered a working hour"
        
    # 19 PM -> Not Working (0)
    assert df.loc[3, 'is_working_hour'] == 0, \
        f"Hour {df.loc[3, 'hour']} should NOT be considered a working hour"
        
    # 00 AM -> Not Working (0)
    assert df.loc[0, 'is_working_hour'] == 0, \
        f"Hour {df.loc[0, 'hour']} should NOT be considered a working hour"

def test_day_of_week_and_weekend(sample_df):
    """Test day of week extraction and weekend flag (day 5,6 are weekends)."""
    df = extract_time_features(sample_df.copy(), verbose=False)
    
    # Day 0 (Index 0) -> 0
    assert df.loc[0, 'dayofweek'] == 0, "First timestamp should be day 0"
    assert df.loc[0, 'is_weekend'] == 0, "Day 0 should not be weekend"
    
    # Day 5 (Index 5) -> 5 (Weekend)
    assert df.loc[5, 'dayofweek'] == 5, "5th day timestamp should be day 5"
    assert df.loc[5, 'is_weekend'] == 1, "Day 5 should be weekend"
    
    # Day 6 (Index 6) -> 6 (Weekend)
    assert df.loc[6, 'dayofweek'] == 6, "6th day timestamp should be day 6"
    assert df.loc[6, 'is_weekend'] == 1, "Day 6 should be weekend"

def test_is_night(sample_df):
    """Test is_night logic (0 <= hour <= 6)."""
    df = extract_time_features(sample_df.copy(), verbose=False)
    
    # 00:00 -> Night (1)
    assert df.loc[0, 'is_night'] == 1, \
        f"Hour {df.loc[0, 'hour']} should be considered night"
        
    # 09:00 -> Day (0)
    assert df.loc[1, 'is_night'] == 0, \
        f"Hour {df.loc[1, 'hour']} should NOT be considered night"

def test_custom_time_column():
    """Test using a custom time column name."""
    df = pd.DataFrame({'custom_time': [3600, 7200]})
    processed_df = extract_time_features(df.copy(), time_col='custom_time', verbose=False)
    
    assert 'hour' in processed_df.columns, \
        "Should successfully extract features from custom time column name"
    assert processed_df.loc[0, 'hour'] == 1, \
        "Should correctly calculate hour from custom time column"
