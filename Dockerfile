FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

WORKDIR /app

# Install system dependencies (for PyMuPDF and Pillow rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmupdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure data and chroma_db directories exist
RUN mkdir -p data .chroma_db

EXPOSE 8080

# Run FastAPI production server
CMD ["uvicorn", "src.generation_service.app:app", "--host", "0.0.0.0", "--port", "8080"]
