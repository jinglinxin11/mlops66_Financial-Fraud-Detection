"""Profile Training Script.

Run profiling on the training pipeline to identify bottlenecks.

Usage:
    python profile_training.py              # Full profiling
    python profile_training.py --dry-run    # Quick test (data loading only)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.profiling import profile_training

logger = get_logger(__name__)


def main():
    """Main profiling entry point."""
    parser = argparse.ArgumentParser(description="Profile training pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Quick test without full training")
    parser.add_argument(
        "--output",
        type=str,
        default="logs/profile_stats.prof",
        help="Output file for profile stats",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("     IEEE-CIS Fraud Detection - Training Profiler")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("Mode: DRY RUN (quick test)")
    else:
        logger.info("Mode: FULL PROFILING")

    # Import config (using legacy Config for now)
    from src.config.settings import Config

    config = Config()

    # Run profiling
    timings = profile_training(config, dry_run=args.dry_run)

    logger.info("\n" + "=" * 60)
    logger.info("              ✅ Profiling Complete!")
    logger.info("=" * 60)

    return timings


if __name__ == "__main__":
    main()
