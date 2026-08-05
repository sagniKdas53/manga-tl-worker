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
    # AUDIT-D2: libxrender1, not libxrender-dev. Nothing in this image compiles against
    # Xrender — every wheel installed below is prebuilt — so the -dev package was shipping
    # headers, a static archive and a pkg-config file into a runtime image for nothing.
    libxrender1 \
    fonts-comic-neue \
    # AUDIT-D2: Liberation Sans and Mono are metric-compatible substitutes for Arial and
    # Courier New, which used to be wget'd from third-party repos that have no right to
    # redistribute them. See the font stanza below.
    fonts-liberation \
    fonts-ipafont-gothic \
    fonts-wqy-microhei \
    fonts-nanum \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Install the two display fonts that Debian does not package (Bangers, Luckiest Guy).
#
# AUDIT-D2: these used to be fetched from `main`/`master`, so the image content depended
# on what those branches happened to hold at build time — an unreproducible build that
# could also change rendered output without a single line of this repo changing. Both are
# now pinned to the commit that last touched the file, and both are checksum-verified so a
# pin that silently starts serving different bytes fails the build instead of shipping.
#
# Arial and Courier New used to be pulled from root-project/root and jfmdev/TuringFonts.
# Neither font is theirs to redistribute — both are Monotype's — which is a licensing
# problem for an image published to GHCR. They are replaced by Liberation Sans and
# Liberation Mono (installed via fonts-liberation above), which are designed to be
# metric-compatible substitutes: same advance widths, so line breaking and fitting are
# unchanged. As a bonus, Liberation ships real bold/italic faces, where the old files had
# one weight mapped to all four styles.
ARG BANGERS_COMMIT=cf67eacb4b4c70430d1c02e55ba0d02232e85fa1
ARG BANGERS_SHA256=4160a7311de9342674cce9160cde9fcbb30f48190397d86ff1b70b455af65824
ARG LUCKIEST_GUY_COMMIT=9a936674760330d42e94ba85eec8cd15b8fb9766
ARG LUCKIEST_GUY_SHA256=cfbdd68a039f92df51cf3721506af6242e64594c6325fe0bedbeff3fe385d980
RUN mkdir -p /usr/share/fonts/truetype/google && \
    apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
    wget -q -O /usr/share/fonts/truetype/google/Bangers-Regular.ttf \
      "https://raw.githubusercontent.com/google/fonts/${BANGERS_COMMIT}/ofl/bangers/Bangers-Regular.ttf" && \
    wget -q -O /usr/share/fonts/truetype/google/LuckiestGuy-Regular.ttf \
      "https://raw.githubusercontent.com/google/fonts/${LUCKIEST_GUY_COMMIT}/apache/luckiestguy/LuckiestGuy-Regular.ttf" && \
    echo "${BANGERS_SHA256}  /usr/share/fonts/truetype/google/Bangers-Regular.ttf" | sha256sum -c - && \
    echo "${LUCKIEST_GUY_SHA256}  /usr/share/fonts/truetype/google/LuckiestGuy-Regular.ttf" | sha256sum -c - && \
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
#
# AUDIT-D2: a BuildKit cache mount, matching what the backend already does for Maven and
# npm. --no-cache-dir is dropped deliberately: it existed to keep pip's downloads out of
# the image layer, and the cache mount does that better, because a mount is never part of
# the layer at all. 1.53 GB of ML wheels no longer get re-downloaded on every rebuild.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY --chown=worker:worker src/worker/ ./worker/
COPY --chown=worker:worker app.py .

# The two cache roots have to exist and be owned before the bind mounts land on them, or
# Docker creates the intermediate directories as root and the worker cannot write.
RUN mkdir -p /home/worker/.cache/huggingface /home/worker/.paddlex /app/rendered_cache \
    && chown -R worker:worker /home/worker /app

ENV HOME=/home/worker
# AUDIT-D2: without this, stdout is block-buffered whenever it is a pipe rather than a
# tty — which is exactly what it is under `docker compose logs` — so log lines arrive in
# 8 KB bursts, or not at all if the process dies holding a partial buffer. That is the
# reason the worker's print() calls carry flush=True; this makes the flags redundant
# rather than load-bearing, so they can be removed as the files are next touched.
ENV PYTHONUNBUFFERED=1
USER worker

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["python", "app.py"]
