FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python backend/data_generator.py && python backend/ml_engine.py
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
