FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

COPY docker/notebook/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install \
      torch==2.7.1 \
      torchvision==0.22.1 \
      torchaudio==2.7.1 \
      --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r /tmp/requirements.txt

EXPOSE 8888

ENTRYPOINT ["tini", "--"]

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.root_dir=/workspace", "--ServerApp.token=", "--ServerApp.password="]
