FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and artifacts
COPY inference ./inference
COPY feature_columns.txt .
COPY mlruns ./mlruns
COPY mlflow.db .

EXPOSE 8000

CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
