# =======================================================================
# Flow-Lenia Research Environment Dockerfile
# =======================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app \
    JAX_PLATFORMS=cpu

WORKDIR /app

# Install system dependencies (FFmpeg for video encoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for caching
COPY requirements.txt pyproject.toml README.md /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY core/ /app/core/
COPY experiments/ /app/experiments/
COPY configs/ /app/configs/
COPY tests/ /app/tests/
COPY run_experiment.py /app/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Default command runs the test suite
CMD ["python", "-m", "unittest", "discover", "tests", "-v"]
