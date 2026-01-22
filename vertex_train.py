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
import subprocess
import sys
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
    parser.add_argument(
        "--gpu-type", type=str, default="NVIDIA_TESLA_T4", help="GPU type"
    )
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--no-gpu", action="store_true", help="Run without GPU")
    parser.add_argument(
        "--local", action="store_true", help="Build and test locally first"
    )
    return parser.parse_args()


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)
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
    # Build the gcloud command
    cmd = [
        "gcloud",
        "ai",
        "custom-jobs",
        "create",
        f"--region={REGION}",
        f"--display-name={job_name}",
        f"--worker-pool-spec="
        f"machine-type={args.machine_type},"
        f"replica-count=1,"
        f"container-image-uri={image_uri}",
    ]

    # Add GPU if requested
    if not args.no_gpu:
        cmd[-1] += f",accelerator-type={args.gpu_type},accelerator-count={args.gpu_count}"

    # Note: Environment variables are passed via the container
    # We need to modify the command to include env vars properly
    cmd = [
        "gcloud",
        "ai",
        "custom-jobs",
        "create",
        f"--region={REGION}",
        f"--display-name={job_name}",
    ]

    # Build worker pool spec
    worker_spec = (
        f"machine-type={args.machine_type},replica-count=1,container-image-uri={image_uri}"
    )

    if not args.no_gpu:
        worker_spec += f",accelerator-type={args.gpu_type},accelerator-count={args.gpu_count}"

    cmd.append(f"--worker-pool-spec={worker_spec}")

    # Add environment variables
    cmd.append(f"--env-vars=GCP_BUCKET={BUCKET}")

    # Add training arguments
    cmd.append(f"--args=--max-epochs={args.max_epochs},--batch-size={args.batch_size}")

    run_command(cmd, "Submitting Vertex AI training job")

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
