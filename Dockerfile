FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace
ENV HF_HOME=/workspace/.cache/huggingface

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /workspace

# Создание виртуального окружения
RUN python3.10 -m venv venv
ENV PATH="/workspace/venv/bin:$PATH"

# Обновление pip
RUN pip install --upgrade pip setuptools wheel

# Установка PyTorch с CUDA 12.1 (совместимый с CUDA 12.2)
RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Клонирование AI-Toolkit
RUN git clone https://github.com/ostris/ai-toolkit.git

# Установка зависимостей AI-Toolkit
WORKDIR /workspace/ai-toolkit
RUN pip install -r requirements.txt

# Установка Flash Attention для оптимизации
RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.13/flash_attn-2.8.1+cu128torch2.7-cp310-cp310-linux_x86_64.whl

# Создание директорий для данных
RUN mkdir -p /workspace/training_data /workspace/output /workspace/models

# Копирование конфигурации тренировки
COPY train_config.yaml /workspace/
COPY train_face_lora.py /workspace/

# Установка переменных окружения для HuggingFace
ENV TRANSFORMERS_CACHE=/workspace/.cache/transformers
ENV HF_DATASETS_CACHE=/workspace/.cache/datasets

# Создание пользователя для безопасности
RUN useradd -m -u 1000 trainer && chown -R trainer:trainer /workspace
USER trainer

# Точка входа
WORKDIR /workspace
CMD ["python", "train_face_lora.py"]
