# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create logs directory (other directories created by api.py on startup from config)
RUN mkdir -p data/logs

# Expose API port
EXPOSE 5000

# Default command - run FastAPI server
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]
