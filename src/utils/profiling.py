"""
Profiling Utilities
Tools for performance profiling and optimization
"""

import cProfile
import pstats
import functools
import time
import io
from pathlib import Path
from typing import Callable, Optional, Any

from .logger import get_logger

logger = get_logger(__name__)


def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Usage:
        @timer
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.info(f"⏱️ {func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper


def profile(output_file: Optional[str] = None, 
            sort_by: str = 'cumulative',
            lines: int = 30) -> Callable:
    """
    Decorator to profile function with cProfile.
    
    Args:
        output_file: Optional path to save profile stats
        sort_by: Sort key ('cumulative', 'time', 'calls')
        lines: Number of lines to print
        
    Usage:
        @profile(output_file='profile_stats.prof')
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            result = func(*args, **kwargs)
            
            profiler.disable()
            
            # Print stats
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats(sort_by)
            stats.print_stats(lines)
            
            logger.info(f"\n📊 Profile for {func.__name__}:\n{stream.getvalue()}")
            
            # Save to file if specified
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                stats.dump_stats(output_file)
                logger.info(f"Profile saved to: {output_file}")
            
            return result
        return wrapper
    return decorator


class Timer:
    """
    Context manager for timing code blocks.
    
    Usage:
        with Timer("data loading"):
            load_data()
    """
    
    def __init__(self, name: str = "block"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        logger.info(f"⏱️ {self.name} took {self.elapsed:.2f} seconds")


def profile_training(cfg, dry_run: bool = False) -> dict:
    """
    Profile the training pipeline.
    
    Args:
        cfg: Hydra config
        dry_run: If True, only profile data loading
        
    Returns:
        dict: Timing statistics
    """
    from ..features.preprocessor import FraudPreprocessor
    from ..config.settings import Config
    
    timings = {}
    
    # Use legacy Config for compatibility
    config = Config()
    
    # Profile preprocessing
    with Timer("Preprocessing") as t:
        preprocessor = FraudPreprocessor(config)
        if not dry_run:
            data = preprocessor.fit_transform()
        else:
            logger.info("Dry run - skipping actual preprocessing")
    timings['preprocessing'] = t.elapsed
    
    if not dry_run:
        # Profile training (first epoch only for speed)
        from ..models.tabnet_trainer import TabNetTrainer
        
        # Modify config for quick test
        config.MAX_EPOCHS = 1
        config.PATIENCE = 1
        
        with Timer("Training (1 epoch)") as t:
            trainer = TabNetTrainer(config, data)
            model = trainer.train()
        timings['training_1_epoch'] = t.elapsed
    
    logger.info(f"\n📊 Profiling Summary:")
    for name, elapsed in timings.items():
        logger.info(f"   {name}: {elapsed:.2f}s")
    
    return timings
