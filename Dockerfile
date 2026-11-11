# Multi-arch friendly: every dependency has linux/amd64 and linux/arm64 wheels.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY features.py ./
COPY sim ./sim
COPY edge ./edge
COPY models ./models

RUN useradd --system --uid 10001 --no-create-home app
USER app

ENTRYPOINT ["python", "-m", "edge.infer"]
CMD ["--sim", "--loop", "--interval", "1"]
