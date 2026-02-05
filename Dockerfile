# House Price Prediction API - Dockerfile
# Build version: 3 - scikit-learn fix
FROM python:3.11-slim

# Cache bust: v3
ARG CACHEBUST=3

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Force cache invalidation for requirements
RUN echo "Cache bust: v4 - scikit-learn 1.7.0 fix"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (scikit-learn>=1.5.0 required)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY ml/ ./ml/
COPY artifacts/ ./artifacts/

# Create non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port (Railway will override with $PORT)
EXPOSE $PORT

# Start the application (Railway provides PORT env variable)
CMD ["python", "-c", "import os; import uvicorn; uvicorn.run('app.main:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))"]
