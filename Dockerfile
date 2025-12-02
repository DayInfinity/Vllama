# Multi-stage build for Vllama - Vision Models CLI Tool
# This Dockerfile creates a containerized environment for running Vllama
# with support for ML models, vision models, and GPU computation

# Stage 1: Base image with Python
FROM python:3.11-slim as base

# Set environment variables to reduce Python bytecode generation and buffering
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Builder stage - install Python dependencies
FROM base as builder

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install requirements
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 3: Runtime stage - minimal production image
FROM base as runtime

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment PATH to use virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create directories for outputs and models
RUN mkdir -p /app/outputs \
    && mkdir -p /app/models \
    && mkdir -p /app/data

# Set the entry point to the CLI
ENTRYPOINT ["vllama"]
CMD ["--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD vllama --version || exit 1

# Labels for metadata
LABEL maintainer="Gopu Manvith <manvithgopu1394@gmail.com>" \
      description="Vllama - CLI tool for vision models and ML workflows" \
      version="0.7.2"
