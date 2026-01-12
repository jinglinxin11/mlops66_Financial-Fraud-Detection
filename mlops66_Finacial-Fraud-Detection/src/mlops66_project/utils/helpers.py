"""
Utility Functions Module
Common helper functions
"""

import os
import glob


def optimize_memory(df, verbose=True):
    """
    Optimize DataFrame memory usage
    
    Args:
        df: pandas DataFrame
        verbose: Whether to print information
        
    Returns:
        pd.DataFrame: Optimized DataFrame
    """
    if verbose:
        print("\n" + "=" * 50)
        print("Optimizing Memory")
        print("=" * 50)
    
    before_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    
    after_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f"Memory optimization: {before_mem:.1f}MB -> {after_mem:.1f}MB")
        print(f"Saved: {(1 - after_mem/before_mem)*100:.1f}%")
    
    return df


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint
    
    Args:
        checkpoint_dir: Checkpoint directory path
        
    Returns:
        tuple: (latest checkpoint path, epoch number) or (None, 0)
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # Find all checkpoint files
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.zip"))
    
    if not checkpoints:
        return None, 0
    
    # Extract epoch numbers and find the maximum
    epochs = []
    for cp in checkpoints:
        try:
            epoch = int(cp.split("_epoch_")[1].replace(".zip", ""))
            epochs.append((epoch, cp))
        except:
            continue
    
    if not epochs:
        return None, 0
    
    # Return the latest checkpoint
    latest_epoch, latest_checkpoint = max(epochs, key=lambda x: x[0])
    
    return latest_checkpoint, latest_epoch
