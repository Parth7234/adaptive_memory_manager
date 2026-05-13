# =============================================================================
# Context-Aware Adaptive Memory Management System — Docker Container
# =============================================================================
# PURPOSE: Zero-friction reproducibility of the simulation & benchmark pipeline.
#          A Samsung judge can simply run:
#            docker build -t edge-memory-sim .
#            docker run edge-memory-sim
#          and get the full 6/6 KPI results in ~10 minutes.
#
# NOTE: This Docker image is for SIMULATION & BENCHMARKING only.
#       On a real Samsung device, the system runs as a native Android
#       Background Service with NPU inference via Samsung ONE / NNAPI,
#       NOT inside a container. Docker would prevent the Adaptive Memory
#       Manager from intercepting real kernel paging (madvise, kswapd, lmkd).
# =============================================================================

# Stage 1: Use lightweight Python base
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed by matplotlib (for chart rendering)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Copy project source
COPY run.py .
COPY simulate.py .
COPY src/ src/

# Create output directories
RUN mkdir -p data models results

# Metadata labels
LABEL maintainer="Parth Singla" \
      description="Context-Aware Adaptive Memory Management Simulation" \
      version="3.0" \
      org.opencontainers.image.source="https://github.com/Parth7234/idk_name"

# Health check: verify Python and PyTorch are functional
HEALTHCHECK --interval=30s --timeout=5s --retries=1 \
    CMD python3 -c "import torch; print('OK')" || exit 1

# Run the full pipeline: data gen → train → simulate → benchmark → visualize
CMD ["python3", "run.py"]
