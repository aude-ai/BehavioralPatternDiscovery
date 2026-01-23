FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY cloud/hetzner/ ./cloud/hetzner/

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["uvicorn", "cloud.hetzner.main:app", "--host", "0.0.0.0", "--port", "8000"]
