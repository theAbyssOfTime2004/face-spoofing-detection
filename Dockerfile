FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --target=/install


FROM python:3.11-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-base AS model-cache

ENV INSIGHTFACE_HOME=/root/.insightface

COPY --from=builder /install /usr/local/lib/python3.11/site-packages

RUN python - <<'PY'
from insightface.app import FaceAnalysis

# Pre-download and validate buffalo_l at build time (fail-fast).
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1)
print("buffalo_l model cache ready at /root/.insightface")
PY


FROM runtime-base AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INSIGHTFACE_HOME=/root/.insightface \
    PYTHONPATH=/app

WORKDIR /app

COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY --from=model-cache /root/.insightface /root/.insightface

COPY app/ /app/app/
COPY src/ /app/src/
COPY config/ /app/config/
COPY models/ /app/models/
COPY requirements.txt /app/requirements.txt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
