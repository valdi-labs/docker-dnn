FROM python:3.10-slim
COPY . .
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir data
CMD ["python", "main.py"]