FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    ffmpeg \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /opt/models /workspace/output /workspace/models \
    && wget -O /opt/models/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

RUN python -c "import os, numpy, cv2, mediapipe; print('OK okolje'); print('numpy', numpy.__version__); print('opencv', cv2.__version__); print('mediapipe', mediapipe.__version__); print('model', os.path.exists('/opt/models/hand_landmarker.task'))"

CMD ["bash"]
