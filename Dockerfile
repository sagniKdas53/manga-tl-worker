FROM python:3.13-slim

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


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/worker/ ./worker/
COPY app.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["python", "app.py"]
