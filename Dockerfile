FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY data/ ./data/

RUN mkdir -p /app/models

CMD ["python", "train.py"]
