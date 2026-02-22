FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.4)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention 2
RUN pip install flash-attn --no-build-isolation

# Copy and install requirements
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy project
COPY . /app
WORKDIR /app

# Create data directory
RUN mkdir -p /app/data/images /app/outputs

# Default command
CMD ["python", "src/train.py", "--config", "configs/train_config.yaml"]
