# Hugging Face Space image (Docker SDK, free CPU tier).
#
# Assembled and uploaded by `python -m web.deploy_space`; the music ships
# inside the image and the template is built at build time, so a restart
# serves in seconds. The free tier has no persistent disk: participants are
# restored from, and synced to, a private dataset (see web/backup.py).
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/study \
    TASTEQ_DATA_DIR=/app/data \
    TASTEQ_MUSIC_DIR=/app/music \
    TASTEQ_TORCH_THREADS=2 \
    PORT=7860

RUN useradd --create-home --uid 1000 study
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# CPU-only torch keeps the image small; nothing here needs a GPU.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[ml,audio,web]"
COPY web ./web
COPY music ./music
RUN python -m web.prepare && chown -R study /app/data

USER study
EXPOSE 7860

# One process on purpose: participants live in memory between requests.
# No access log, so client IP addresses are never written anywhere.
CMD exec uvicorn web.app:create_app --factory --host 0.0.0.0 --port "$PORT" \
    --workers 1 --no-access-log --proxy-headers
