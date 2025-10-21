# Use slim Python base image
FROM python:3.12-slim

# System deps sometimes needed for scientific wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model artifacts
COPY main.py ./main.py
COPY best_model/ ./best_model/

# Expose (for docs only); Render still injects PORT
EXPOSE 8000

# Start FastAPI (use PORT from env or default to 8000 locally)
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
