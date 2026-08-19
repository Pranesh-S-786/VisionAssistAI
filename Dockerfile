FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    TORCH_HOME=/app/weights \
    YOLO_CONFIG_DIR=/app/weights

# Create a non-root user for Hugging Face Spaces
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user:user backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir ftfy regex git+https://github.com/ultralytics/CLIP.git

# Copy project files
COPY --chown=user:user . .

# Expose Hugging Face default port 7860
EXPOSE 7860

# Run FastAPI backend with Uvicorn on port 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
