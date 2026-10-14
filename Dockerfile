# Multi-stage production Dockerfile for Everyday Object Detection
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system runtime dependencies for OpenCV and audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install lightweight CPU PyTorch
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and weights
COPY . /app/

# Expose ports: 7860 for Gradio Web UI / HF Spaces, 8000 for FastAPI
EXPOSE 7860 8000

# Default entrypoint launches the Gradio interactive assistant
CMD ["python", "app.py"]
