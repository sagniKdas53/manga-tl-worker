# AUDIT-D2: pinned by digest, not just by tag — python:3.13-slim is republished on every
# patch release, so an untagged rebuild silently changed the interpreter and the base
# userland underneath a 1.9 GB image. Bump this deliberately.
FROM python:3.13-slim@sha256:99569264a52f7665899b7bc0fb48e72a2712b850b129f63c4733af1e939accfb

# Install system libraries needed by opencv / easyocr
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    fonts-comic-neue \
    fonts-ipafont-gothic \
    fonts-wqy-microhei \
    fonts-nanum \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Install additional fonts (Bangers, Luckiest Guy, Arial, Courier New)
RUN mkdir -p /usr/share/fonts/truetype/google && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget -q -O /usr/share/fonts/truetype/google/Bangers-Regular.ttf "https://github.com/google/fonts/raw/main/ofl/bangers/Bangers-Regular.ttf" && \
    wget -q -O /usr/share/fonts/truetype/google/LuckiestGuy-Regular.ttf "https://github.com/google/fonts/raw/main/apache/luckiestguy/LuckiestGuy-Regular.ttf" && \
    wget -q -O /usr/share/fonts/truetype/google/Arial.ttf "https://raw.githubusercontent.com/root-project/root/master/fonts/arial.ttf" && \
    wget -q -O /usr/share/fonts/truetype/google/CourierNew.ttf "https://raw.githubusercontent.com/jfmdev/TuringFonts/master/fonts/Courier%20New.ttf" && \
    apt-get purge -y wget && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    fc-cache -f


# AUDIT-D2: run as an unprivileged user. The UID is fixed at 10001 rather than left to
# useradd because the model caches are bind-mounted from the host, and bind mounts carry
# host ownership straight through — a UID that drifts between rebuilds would silently lose
# write access to 374 MB of downloaded models.
RUN useradd --create-home --uid 10001 --user-group worker

WORKDIR /app

# Dependencies install as root into system site-packages, which stays read-only at runtime.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=worker:worker src/worker/ ./worker/
COPY --chown=worker:worker app.py .

# The two cache roots have to exist and be owned before the bind mounts land on them, or
# Docker creates the intermediate directories as root and the worker cannot write.
RUN mkdir -p /home/worker/.cache/huggingface /home/worker/.paddlex /app/rendered_cache \
    && chown -R worker:worker /home/worker /app

ENV HOME=/home/worker
USER worker

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["python", "app.py"]
