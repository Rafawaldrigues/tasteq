# Web study image. Build from the repository root:
#   docker build -f web/Dockerfile -t tasteq-study .
# Mount a persistent volume at /data with the music in /data/music.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TASTEQ_DATA_DIR=/data \
    TASTEQ_MUSIC_DIR=/data/music \
    TASTEQ_TORCH_THREADS=2 \
    PORT=8000

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# CPU-only torch keeps the image small; nothing here needs a GPU.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[ml,audio,web]"
COPY web ./web

RUN useradd --create-home --uid 1000 study && mkdir -p /data && chown study /data
USER study
EXPOSE 8000

# One process on purpose: participants live in memory between requests, so
# several workers would each hold their own copy. Requests run in threads.
# No access log, so client IP addresses are never written anywhere.
CMD python -m web.prepare && \
    exec uvicorn web.app:create_app --factory --host 0.0.0.0 --port "$PORT" \
        --workers 1 --no-access-log --proxy-headers
