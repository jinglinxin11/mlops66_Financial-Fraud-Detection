# Dockerfile for Fraud Detection API
# Supports both local development (mount volumes) and cloud deployment (GCP bucket)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (lighter for API deployment)
# RUN pip install --no-cache-dir torch torchvision torchaudio 
#--index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install dependencies 
# Unified installation of PyTorch and requirements to ensure it gets CPU version for PyTorch
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio \
    -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY api/ ./api/

# Copy entrypoint script and fix line endings (Windows -> Unix)
COPY docker-entrypoint.sh .
RUN sed -i 's/\r$//' docker-entrypoint.sh && chmod +x docker-entrypoint.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Entrypoint handles GCP download if configured
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default command: run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
