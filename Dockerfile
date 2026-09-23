FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    U2NET_HOME=/app/.cache/models NUMBA_CACHE_DIR=/app/.cache/numba \
    CPU_THREADS=4 PRELOAD_MODEL=true PORT=8000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY vehicle_pipeline ./vehicle_pipeline
RUN pip install --no-cache-dir .
# Fetch and checksum weights at build time; do not load the large inference session.
RUN python -c "from rembg.sessions.birefnet_general import BiRefNetSessionGeneral; BiRefNetSessionGeneral.download_models()"
COPY backgrounds/parking-lots ./backgrounds/parking-lots
COPY backgrounds/backgrounds ./backgrounds/backgrounds
RUN useradd --create-home --uid 10001 vehicle && chown -R vehicle:vehicle /app
USER vehicle
EXPOSE 8000
# A small connection limit bounds waiting requests; excess load gets HTTP 503.
CMD ["sh", "-c", "exec uvicorn vehicle_pipeline.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --limit-concurrency 4 --timeout-keep-alive 5"]
