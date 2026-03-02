# LOTC Reproducibility Container
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements or install directly
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    scikit-learn \
    pyyaml \
    torchvision \
    pandas

# Copy codebase
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Default command
CMD ["python", "scripts/run_combined_v3.py"]
