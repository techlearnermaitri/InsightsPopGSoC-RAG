FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by some Python packages
# (e.g. pypdf, sentence-transformers may need libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache optimisation)
COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy the full project into the container
COPY . .

# Set PYTHONPATH to project root so `from server.xxx import` resolves correctly
ENV PYTHONPATH=/app

# Expose the backend port
EXPOSE 8000

# Run the FastAPI app via uvicorn
# Entry point: server/main.py → `app = FastAPI(...)`
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

