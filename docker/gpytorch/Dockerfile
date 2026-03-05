FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
ARG TARGETARCH
ENV CONDA_DIR=/opt/conda
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    else \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    fi && \
    curl -fsSL "$MINICONDA_URL" -o /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -f /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create environment
RUN conda create -n gpytorch python=3.12 -y

# Install PyTorch + deps in env
SHELL ["conda", "run", "-n", "gpytorch", "/bin/bash", "-c"]
RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
RUN python -m pip install --no-cache-dir gpytorch matplotlib pandas numpy scikit-learn jupyterlab

# Auto-activate conda env for interactive shells
RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> /root/.bashrc && \
    echo 'conda activate gpytorch' >> /root/.bashrc

WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate gpytorch && cd /workspace && if [ $# -eq 0 ] || [ \"$1\" = \"/bin/bash\" ]; then exec /bin/bash -i; else exec \"$@\"; fi", "--"]
CMD ["/bin/bash"]
