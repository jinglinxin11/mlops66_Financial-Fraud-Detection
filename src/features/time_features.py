"""Time Feature Extraction Module.

Extract meaningful time features from TransactionDT.
"""


def extract_time_features(df, time_col="TransactionDT", verbose=True):
    """Extract time features from timestamp.

    Args:
        df: DataFrame
        time_col: Time column name
        verbose: Whether to print information

    Returns:
        pd.DataFrame: DataFrame with added time features

    """
    if time_col not in df.columns:
        if verbose:
            print(f"Warning: {time_col} column not found, skipping time feature extraction")
        return df

    if verbose:
        print("\n" + "=" * 50)
        print("Extracting Time Features")
        print("=" * 50)

    # TransactionDT is a relative timestamp (seconds)
    # Extract meaningful time features

    # Hour (0-23)
    df["hour"] = (df[time_col] // 3600) % 24

    # Day of week (0-6)
    df["dayofweek"] = (df[time_col] // 86400) % 7

    # Day of month (approximate)
    df["dayofmonth"] = (df[time_col] // 86400) % 30

    # Is working hour (9-18)
    df["is_working_hour"] = ((df["hour"] >= 9) & (df["hour"] <= 18)).astype(int)

    # Is weekend
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Is night (0-6 AM)
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 6)).astype(int)

    if verbose:
        print("New features: hour, dayofweek, dayofmonth, is_working_hour, is_weekend, is_night")

    return df
