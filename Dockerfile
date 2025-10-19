# Use slim Python base image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency list
COPY ../requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model into the image
COPY ../app/main.py model.joblib model_meta.json ./

# Expose port 8000
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]