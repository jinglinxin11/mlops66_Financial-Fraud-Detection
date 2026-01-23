"""Submit a training job to Vertex AI.

Usage:
    python vertex_train.py [OPTIONS]

Options:
    --max-epochs INT     Maximum training epochs (default: 100)
    --batch-size INT     Batch size (default: 8192)
    --machine-type STR   GCP machine type (default: n1-standard-8)
    --gpu-type STR       GPU type (default: NVIDIA_TESLA_T4)
    --gpu-count INT      Number of GPUs (default: 1)
    --no-gpu             Run without GPU (CPU only)
    --local              Build and test locally first
"""

import argparse
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime

# Configuration
PROJECT_ID = "machinelearningops66"
REGION = "europe-west1"
BUCKET = "databucketmlops66"
REPO_NAME = "fraud-detection"
IMAGE_NAME = "training"
SERVICE_ACCOUNT = None  # Uses default Compute Engine SA


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Submit training job to Vertex AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size")
    parser.add_argument(
        "--machine-type", type=str, default="n1-standard-8", help="GCP machine type"
    )
    parser.add_argument("--gpu-type", type=str, default="NVIDIA_TESLA_T4", help="GPU type")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--no-gpu", action="store_true", help="Run without GPU")
    parser.add_argument("--local", action="store_true", help="Build and test locally first")
    return parser.parse_args()


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}\n")

    # On Windows, use gcloud.cmd instead of gcloud
    if platform.system() == "Windows" and len(cmd) > 0 and cmd[0] == "gcloud":
        cmd[0] = "gcloud.cmd"

    # Use shell=True on Windows for better command execution
    use_shell = platform.system() == "Windows"
    
    result = subprocess.run(cmd, capture_output=False, shell=use_shell)
    if result.returncode != 0:
        print(f"Error: {description} failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    """Main function to submit Vertex AI training job."""
    args = parse_args()

    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"fraud-detection-training-{timestamp}"

    # Image URI
    image_uri = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/{IMAGE_NAME}:latest"

    print("\n" + "=" * 60)
    print("   Vertex AI Training Job Submission")
    print("=" * 60)
    print(f"\nProject:      {PROJECT_ID}")
    print(f"Region:       {REGION}")
    print(f"Bucket:       {BUCKET}")
    print(f"Job Name:     {job_name}")
    print(f"Image:        {image_uri}")
    print(f"Machine:      {args.machine_type}")
    if not args.no_gpu:
        print(f"GPU:          {args.gpu_count}x {args.gpu_type}")
    print(f"Max Epochs:   {args.max_epochs}")
    print(f"Batch Size:   {args.batch_size}")

    # Step 1: Build Docker image
    run_command(
        ["docker", "build", "-f", "Dockerfile.train", "-t", image_uri, "."],
        "Building training Docker image",
    )

    # Step 2: Test locally (optional)
    if args.local:
        print("\n" + "=" * 60)
        print("  Local test requested - stopping before push")
        print("=" * 60)
        print("\nTo test locally, run:")
        print(f"  docker run -e GCP_BUCKET={BUCKET} {image_uri} --max-epochs 1")
        print("\nNote: You need to mount GCP credentials for local testing")
        return

    # Step 3: Push to Artifact Registry
    run_command(
        ["docker", "push", image_uri],
        "Pushing image to Artifact Registry",
    )

    # Step 4: Submit Vertex AI job
    # Create temporary YAML config file for environment variables
    # (gcloud ai custom-jobs create doesn't support --env-vars flag)
    # Build YAML config with proper structure
    config_yaml = "workerPoolSpecs:\n"
    config_yaml += "  - machineSpec:\n"
    config_yaml += f"      machineType: {args.machine_type}\n"
    
    if not args.no_gpu:
        config_yaml += f"      acceleratorType: {args.gpu_type}\n"
        config_yaml += f"      acceleratorCount: {args.gpu_count}\n"
    
    config_yaml += "    replicaCount: 1\n"
    config_yaml += "    containerSpec:\n"
    config_yaml += f"      imageUri: {image_uri}\n"
    config_yaml += "      env:\n"
    config_yaml += "        - name: GCP_BUCKET\n"
    config_yaml += f"          value: {BUCKET}\n"
    config_yaml += "      args:\n"
    config_yaml += f"        - \"--max-epochs={args.max_epochs}\"\n"
    config_yaml += f"        - \"--batch-size={args.batch_size}\"\n"

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = f.name

    try:
        # Build the gcloud command with config file
        cmd = [
            "gcloud",
            "ai",
            "custom-jobs",
            "create",
            f"--region={REGION}",
            f"--display-name={job_name}",
            f"--config={config_file}",
        ]

        run_command(cmd, "Submitting Vertex AI training job")
    finally:
        # Clean up temporary config file
        try:
            os.unlink(config_file)
        except Exception:
            pass  # Ignore cleanup errors

    print("\n" + "=" * 60)
    print("   Job Submitted Successfully!")
    print("=" * 60)
    print("\nMonitor your job at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print("\nOr via CLI:")
    print(f"  gcloud ai custom-jobs list --region={REGION}")
    print("\nAfter completion, model will be uploaded to:")
    print(f"  gs://{BUCKET}/models/tabnet_fraud_model.zip")


if __name__ == "__main__":
    main()
