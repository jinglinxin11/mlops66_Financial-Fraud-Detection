#!/bin/bash
# Training entrypoint script for Vertex AI
# Downloads data from GCP bucket, runs training, uploads results

set -e

echo "=========================================="
echo "   Fraud Detection - Training Job"
echo "=========================================="

# GCP_BUCKET should be set as an environment variable
if [ -z "$GCP_BUCKET" ]; then
    echo "ERROR: GCP_BUCKET environment variable not set"
    exit 1
fi

echo "GCP Bucket: $GCP_BUCKET"

# Create data directory
mkdir -p data

# Download training data from GCP bucket
echo ""
echo "Downloading training data from GCP..."
python -c "
from google.cloud import storage
import os

bucket_name = os.environ['GCP_BUCKET']
client = storage.Client()
bucket = client.bucket(bucket_name)

# Files to download for training
files = [
    ('data/train_transaction.csv', 'data/train_transaction.csv'),
    ('data/train_identity.csv', 'data/train_identity.csv'),
]

for blob_path, local_path in files:
    print(f'  Downloading {blob_path}...')
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print(f'    -> {local_path}')

print('Download complete!')
"

echo ""
echo "Starting training..."
echo "=========================================="

# Run training with any passed arguments
python train.py "$@"

echo ""
echo "=========================================="
echo "Training complete! Uploading results..."

# Upload model and preprocessor to GCP bucket
python -c "
from google.cloud import storage
import os
from pathlib import Path

bucket_name = os.environ['GCP_BUCKET']
client = storage.Client()
bucket = client.bucket(bucket_name)

# Files to upload after training
uploads = [
    ('tabnet_fraud_model.zip', 'models/tabnet_fraud_model.zip'),
    ('ieee_cis_preprocessor.pkl', 'models/ieee_cis_preprocessor.pkl'),
]

for local_path, blob_path in uploads:
    if Path(local_path).exists():
        print(f'  Uploading {local_path} -> {blob_path}')
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)
    else:
        print(f'  Warning: {local_path} not found, skipping')

# Also upload checkpoints if they exist
checkpoint_dir = Path('checkpoints')
if checkpoint_dir.exists():
    for ckpt in checkpoint_dir.glob('*.zip'):
        blob_path = f'checkpoints/{ckpt.name}'
        print(f'  Uploading {ckpt} -> {blob_path}')
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(ckpt))

print('Upload complete!')
"

echo ""
echo "=========================================="
echo "   Training job finished successfully!"
echo "=========================================="
