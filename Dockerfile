FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data.csv assignment.md ./

RUN pip install --no-cache-dir ".[deep-learning]"

EXPOSE 8000
CMD ["python", "-m", "microgcc", "serve", "--artifacts", "artifacts", "--host", "0.0.0.0", "--port", "8000"]

