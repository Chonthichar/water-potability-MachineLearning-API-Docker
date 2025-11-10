# Use slim Python base image
FROM python:3.12-slim

# (Optional) system deps for scientific wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./main.py
COPY best_model/ ./best_model/

# Spaces expects the app to listen on 7860
EXPOSE 7860

# Use HF's injected PORT (defaults to 7860 locally)
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
