#!/bin/bash
# Docker entrypoint script
# Downloads data from GCP bucket if GCP_BUCKET environment variable is set

set -e

echo "=== Fraud Detection API Container ==="

# Check if GCP bucket is configured
if [ -n "$GCP_BUCKET" ]; then
    echo "GCP_BUCKET is set: $GCP_BUCKET"
    echo "Downloading data and model from GCP..."
    
    # Run the download script
    python -c "
from google.cloud import storage
import os

bucket_name = os.environ['GCP_BUCKET']
client = storage.Client()
bucket = client.bucket(bucket_name)

# Define files to download
files_to_download = [
    ('data/test_transaction.csv', 'data/test_transaction.csv'),
    ('data/test_identity.csv', 'data/test_identity.csv'),
    ('models/tabnet_fraud_model.zip', 'tabnet_fraud_model.zip'),
    ('models/ieee_cis_preprocessor.pkl', 'ieee_cis_preprocessor.pkl'),
]

# Create directories
os.makedirs('data', exist_ok=True)

# Download files
for blob_path, local_path in files_to_download:
    try:
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        print(f'  Downloaded: {blob_path} -> {local_path}')
    except Exception as e:
        print(f'  Warning: Could not download {blob_path}: {e}')

print('Download complete!')
"
    echo "GCP download complete."
else
    echo "GCP_BUCKET not set. Expecting local volume mounts."
    echo "  Mount data with: -v ./data:/app/data"
    echo "  Mount model with: -v ./tabnet_fraud_model:/app/tabnet_fraud_model"
fi

# Check if required files exist
echo ""
echo "Checking required files..."
if [ -d "/app/data" ]; then
    echo "  [OK] data/ directory exists"
else
    echo "  [WARNING] data/ directory not found"
fi

if [ -d "/app/tabnet_fraud_model" ] || [ -f "/app/tabnet_fraud_model.zip" ]; then
    echo "  [OK] Model files exist"
else
    echo "  [WARNING] Model files not found (required for /predict_test endpoint)"
fi

echo ""
echo "Starting API server..."
exec "$@"
