# Server image: Ubuntu 24.04 (Python 3.12) + FFmpeg.
# NOTE: written but not built in the authoring sandbox — build it once before relying on it.
#
#   docker build -t video-pipeline .
#   docker run --rm --env-file .env \
#     -v "$PWD/assets:/app/assets" -v "$PWD/output:/app/output" \
#     -v "$PWD/state:/app/state"   -v "$PWD/cache:/app/cache" \
#     -v "$PWD/scripts:/app/scripts" \
#     video-pipeline run scripts/my_script.json
#
# YouTube upload from the container: add  -v "$PWD/secrets:/app/secrets"  and run `upload ...`.
# The first Google login needs a browser, so do it once on the host (`python -m pipeline upload ...`)
# and let the container reuse secrets/token_<channel_id>.json. secrets/ is never copied into the image.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv ffmpeg fonts-dejavu-core fontconfig ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install -r requirements.txt
ENV PATH="/opt/venv/bin:$PATH"

COPY pipeline ./pipeline
COPY tests ./tests
COPY validate_script.py video_script.schema.json video_script.example.json pytest.ini ./
RUN mkdir -p assets/music assets/sfx assets/fonts output state cache secrets

# .env is NOT copied into the image: pass keys with --env-file or -e.
ENTRYPOINT ["python", "-m", "pipeline"]
CMD ["--help"]
