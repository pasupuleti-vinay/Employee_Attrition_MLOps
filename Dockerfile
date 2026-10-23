FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY orchestration/ ./orchestration/
COPY WA_Fn-UseC_-HR-Employee-Attrition.csv .

CMD ["python", "src/train.py"]
